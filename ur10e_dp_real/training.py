"""Supervised adaptation with explicit total optimizer updates and resumable RNG."""

import copy
import json
import math
import random
import time
from pathlib import Path
import numpy as np


def configure_normalizer(model, trainset, settings, *, from_scratch, resume):
    """Fit target statistics before creating EMA; resumed weights retain saved stats."""
    mode = settings.get(
        "normalizer", "real_train_only" if from_scratch else "preserved_from_source"
    )
    if mode not in ("real_train_only", "preserved_from_source"):
        raise ValueError("normalizer must be real_train_only or preserved_from_source")
    if from_scratch and mode != "real_train_only":
        raise ValueError("Random initialization requires real_train_only normalization")
    if mode == "real_train_only" and not resume:
        model.set_normalizer(trainset.get_normalizer())
        model.to(settings["device"])
    return mode


def train(source, data_root, output, settings, resume=None, policy_config=None):
    import torch
    from .model import load_policy, new_policy, sha256
    from .data import Episodes, discover, split_episodes
    from diffusion_policy.model.diffusion.ema_model import EMAModel

    output = Path(output)
    output.mkdir(parents=True, exist_ok=bool(resume))
    seed = settings["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if settings["max_updates"] < 1 or settings["batch_size"] < 1:
        raise ValueError("Training budget and batch must be positive")
    if settings["scheduler"] not in ("constant", "cosine"):
        raise ValueError("scheduler must be constant or cosine")
    if resume:
        model, config, payload = load_policy(resume, settings["device"], weights="raw")
        if payload["settings"] != settings:
            raise ValueError(
                "Resume settings differ; start a new adaptation run to change the schedule"
            )
        split = payload["split"]
        train_paths, val_paths = [Path(p) for p in split["train"]], [
            Path(p) for p in split["val"]
        ]
    else:
        if policy_config:
            model, config = new_policy(policy_config, settings["device"])
        else:
            model, config, _ = load_policy(source, settings["device"])
        train_paths, val_paths = split_episodes(
            discover(data_root), settings["val_fraction"], seed
        )
        split = {
            "train": [str(p.resolve()) for p in train_paths],
            "val": [str(p.resolve()) for p in val_paths],
        }
        payload = None
    if settings.get("split_manifest"):
        locked = json.loads(Path(settings["split_manifest"]).read_text())
        actual = {key: [str(p.resolve()) for p in paths] for key, paths in
                  (("train", train_paths), ("val", val_paths))}
        if any(actual[key] != locked[key] for key in actual):
            raise ValueError("Dataset split differs from the locked split manifest")
    dataset = Episodes(train_paths, model.horizon, model.n_obs_steps)
    valset = Episodes(val_paths, model.horizon, model.n_obs_steps)
    if valset.contract is not None and dataset.contract != valset.contract:
        raise ValueError("Train/validation coordinate or timing contracts differ")
    updates_per_epoch = math.ceil(len(dataset) / settings["batch_size"])
    if "epochs" in settings and settings["max_updates"] != settings["epochs"] * updates_per_epoch:
        raise ValueError("max_updates must equal epochs * ceil(train_windows / batch_size)")
    normalizer_mode = configure_normalizer(
        model, dataset, settings, from_scratch=bool(policy_config), resume=bool(resume)
    )
    ema = copy.deepcopy(model)
    ema.requires_grad_(False)
    if settings["freeze_encoder"]:
        model.obs_encoder.requires_grad_(False)
    encoder = list(model.obs_encoder.parameters())
    ids = {id(p) for p in encoder}
    other = [p for p in model.parameters() if id(p) not in ids and p.requires_grad]
    groups = [{"params": other, "lr": settings["learning_rate"]}]
    if not settings["freeze_encoder"]:
        groups.append({"params": encoder, "lr": settings["encoder_learning_rate"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=settings["weight_decay"])

    def factor(step):
        warm = settings["warmup_updates"]
        if step < warm:
            return (step + 1) / max(1, warm)
        if settings["scheduler"] == "constant":
            return 1.0
        progress = (step - warm) / max(1, settings["max_updates"] - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    ema_update = EMAModel(ema, power=0.75)
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=settings["batch_size"], shuffle=False, num_workers=0
    )
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(dataset), generator=generator)
    cursor, step, seen_windows = 0, 0, 0

    def fingerprint(path):
        import hashlib

        digest = hashlib.sha256()
        for item in sorted(path.rglob("*")):
            if item.is_file():
                digest.update(str(item.relative_to(path)).encode())
                digest.update(bytes.fromhex(sha256(item)))
        return digest.hexdigest()

    fingerprints = {str(p.resolve()): fingerprint(p) for p in train_paths + val_paths}
    if payload:
        if payload["dataset_hashes"] != fingerprints:
            raise ValueError("Episode data changed since saved checkpoint")
        ema.load_state_dict(payload["ema_model"])
        # The normalizer recreates parameters while loading; move them back to GPU.
        ema.to(settings["device"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        ema_update.optimization_step = payload["ema_updates"]
        order, cursor, step = payload["order"], payload["cursor"], payload["updates"]
        seen_windows = payload["seen_windows"]
        generator.set_state(payload["sampler_rng"])
        torch.set_rng_state(payload["torch_rng"])
        if torch.cuda.is_available() and payload["cuda_rng"]:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        np.random.set_state(payload["numpy_rng"])
        random.setstate(payload["python_rng"])
    manifest = {
        "source_checkpoint": str(Path(source).resolve()) if source else None,
        "source_sha256": sha256(source) if source else None,
        "initialization": "random" if policy_config else "source_ema",
        "policy_config": config,
        "split": split,
        "settings": settings,
        "train_windows": len(dataset),
        "val_windows": len(valset),
        "contract": dataset.contract,
        "normalizer": normalizer_mode,
        "updates_per_epoch": updates_per_epoch,
        "dataset_hashes": fingerprints,
    }
    (output / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")

    best_val = payload.get("best_val_loss", float("inf")) if payload else float("inf")

    def save(name=None):
        state = dict(
            policy_config=config,
            model=model.state_dict(),
            ema_model=ema.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            updates=step,
            seen_windows=seen_windows,
            settings=settings,
            split=split,
            dataset_hashes=fingerprints,
            contract=dataset.contract,
            order=order,
            cursor=cursor,
            sampler_rng=generator.get_state(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=(
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
            numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),
            ema_updates=ema_update.optimization_step,
            best_val_loss=best_val,
        )
        dest = output / (name or "update_{:06d}.pt".format(step))
        temp = dest.with_suffix(".tmp")
        torch.save(state, temp)
        temp.replace(dest)
        return dest

    if not payload:
        save()
    print("Initialization:", manifest["initialization"], "| train/val episodes:",
          len(train_paths), len(val_paths), "| windows:", len(dataset), len(valset), flush=True)
    start_time, initial_step = time.perf_counter(), step
    from tqdm import tqdm
    progress = tqdm(total=settings["max_updates"], initial=step, unit="update", dynamic_ncols=True)
    with (output / "metrics.jsonl").open("a") as stream:
        while step < settings["max_updates"]:
            if cursor >= len(order):
                order = torch.randperm(len(dataset), generator=generator)
                cursor = 0
            indices = order[cursor : cursor + settings["batch_size"]].tolist()
            cursor += len(indices)
            seen_windows += len(indices)
            batch = torch.utils.data.default_collate([dataset[i] for i in indices])

            def move(value):
                return (
                    {k: move(v) for k, v in value.items()}
                    if isinstance(value, dict)
                    else value.to(settings["device"])
                )

            model.train()
            if settings["freeze_encoder"]:
                model.obs_encoder.eval()
            loss = model.compute_loss(move(batch))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            ema_update.step(model)
            step += 1
            log = {
                "updates": step,
                "equivalent_epochs": seen_windows / len(dataset),
                "train_loss": loss.item(),
                "lr": scheduler.get_last_lr(),
            }
            if len(valset) and step % settings["validate_every"] == 0:
                # Fixed validation noise/crops and EMA weights; validation must not change training RNG.
                ema.eval()
                with torch.random.fork_rng(), torch.no_grad():
                    torch.manual_seed(seed + 1)
                    losses = []
                    for i, b in enumerate(val_loader):
                        if settings["validation_batches"] > 0 and i >= settings["validation_batches"]:
                            break
                        losses.append((ema.compute_loss(move(b)).item(), len(b["action"])))
                log["ema_val_loss"] = float(sum(v*n for v,n in losses) / sum(n for _,n in losses))
            log["elapsed_seconds"] = time.perf_counter() - start_time
            log["updates_per_second"] = (step - initial_step) / log["elapsed_seconds"]
            stream.write(json.dumps(log) + "\n")
            stream.flush()
            progress.update(1)
            progress.set_postfix(epoch=round(log["equivalent_epochs"], 2), loss=round(loss.item(), 5))
            if "ema_val_loss" in log:
                print("\nValidation update", step, "epoch", round(log["equivalent_epochs"], 3),
                      "EMA loss", log["ema_val_loss"], flush=True)
                if log["ema_val_loss"] < best_val:
                    best_val = log["ema_val_loss"]
                    save("best_val.pt")
                    (output / "best_val.json").write_text(json.dumps(dict(
                        updates=step, equivalent_epochs=log["equivalent_epochs"],
                        ema_val_loss=best_val, criterion="offline EMA denoising loss; closed-loop success unverified"), indent=2) + "\n")
            if step in settings["save_updates"] or step == settings["max_updates"]:
                print("Saved", save(), "loss", loss.item(), flush=True)
            if step % settings.get("latest_every", 500) == 0:
                save("latest.pt")
    progress.close()
    (output / "completion.json").write_text(json.dumps(dict(
        updates=step, expected_updates=settings["max_updates"],
        elapsed_seconds=time.perf_counter()-start_time,
        equivalent_epochs=seen_windows/len(dataset),
        status="training_complete"), indent=2) + "\n")
    return manifest
