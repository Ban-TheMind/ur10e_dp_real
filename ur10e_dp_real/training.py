"""Supervised adaptation with explicit total optimizer updates and resumable RNG."""

import copy
import json
import math
import random
from pathlib import Path
import numpy as np


def train(source, data_root, output, settings, resume=None):
    import torch
    from .model import load_policy, sha256
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
        model, config, _ = load_policy(source, settings["device"])
        train_paths, val_paths = split_episodes(
            discover(data_root), settings["val_fraction"], seed
        )
        split = {
            "train": [str(p.resolve()) for p in train_paths],
            "val": [str(p.resolve()) for p in val_paths],
        }
        payload = None
    dataset = Episodes(train_paths, model.horizon, model.n_obs_steps)
    valset = Episodes(val_paths, model.horizon, model.n_obs_steps)
    if valset.contract is not None and dataset.contract != valset.contract:
        raise ValueError("Train/validation coordinate or timing contracts differ")
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
        "source_checkpoint": str(Path(source).resolve()),
        "source_sha256": sha256(source),
        "split": split,
        "settings": settings,
        "train_windows": len(dataset),
        "val_windows": len(valset),
        "contract": dataset.contract,
        "normalizer": "preserved_from_source",
        "dataset_hashes": fingerprints,
    }
    (output / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def save():
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
        )
        dest = output / ("update_{:06d}.pt".format(step))
        temp = dest.with_suffix(".tmp")
        torch.save(state, temp)
        temp.replace(dest)
        return dest

    if not payload:
        save()
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
                        if i >= settings["validation_batches"]:
                            break
                        losses.append(ema.compute_loss(move(b)).item())
                log["ema_val_loss"] = float(np.mean(losses))
            stream.write(json.dumps(log) + "\n")
            stream.flush()
            if step in settings["save_updates"] or step == settings["max_updates"]:
                print("Saved", save(), "loss", loss.item(), flush=True)
    return manifest
