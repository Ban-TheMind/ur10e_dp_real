"""Evaluate saved EMA policies on the locked validation episodes and export curves."""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    import torch
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ur10e_dp_real.data import Episodes
    from ur10e_dp_real.model import load_policy

    torch.set_num_threads(4)
    run = args.run.resolve()
    meta = json.loads((run / "run.json").read_text())
    completed = json.loads((run / "completion.json").read_text())
    if completed["updates"] != meta["settings"]["max_updates"]:
        raise ValueError("Training budget is incomplete")
    target = run / "analysis"
    target.mkdir(exist_ok=True)
    records = []
    for checkpoint in sorted(run.glob("update_*.pt")):
        model, _, payload = load_policy(checkpoint, "cuda")
        valset = Episodes([Path(p) for p in meta["split"]["val"]], model.horizon, model.n_obs_steps)
        loader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=False, num_workers=0)
        torch.manual_seed(meta["settings"]["seed"] + 1)
        total, count = 0., 0
        def move(value):
            return {k: move(v) for k,v in value.items()} if isinstance(value,dict) else value.cuda()
        with torch.no_grad():
            for batch in loader:
                loss = model.compute_loss(move(batch)).item()
                if not np.isfinite(loss):
                    raise ValueError("Nonfinite validation loss")
                n = len(batch["action"])
                total += loss*n
                count += n
        if count != meta["val_windows"]:
            raise ValueError("Validation set was not evaluated in full")
        row = dict(checkpoint=checkpoint.name, updates=payload["updates"],
                   equivalent_epochs=payload["seen_windows"]/meta["train_windows"],
                   ema_val_loss=total/count, validation_windows=count)
        records.append(row)
        print("Validation:", row, flush=True)
        del model, payload
    if not records:
        raise ValueError("No stage checkpoints found")
    with (target / "checkpoint_validation.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    logs = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot([x["equivalent_epochs"] for x in logs], [x["train_loss"] for x in logs], linewidth=.6)
    axes[0].set_ylabel("Raw training denoising loss")
    vals = [x for x in logs if "ema_val_loss" in x]
    axes[1].plot([x["equivalent_epochs"] for x in vals], [x["ema_val_loss"] for x in vals], label="Training-time EMA validation")
    axes[1].plot([x["equivalent_epochs"] for x in records], [x["ema_val_loss"] for x in records], label="Saved checkpoints, full validation")
    axes[1].set_ylabel("EMA validation denoising loss")
    axes[1].set_xlabel("Equivalent training epochs")
    axes[1].legend()
    fig.suptitle("Real Lift DP: " + meta["initialization"] + " (unsmoothed; offline evaluation)")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(target / "loss_curves.png", dpi=180)
    fig.savefig(target / "loss_curves.pdf")
    plt.close(fig)
    summary = dict(status="passed", split=meta["split"], validation_windows=meta["val_windows"],
                   checkpoint_count=len(records), best_saved_checkpoint=min(records,key=lambda x:x["ema_val_loss"]),
                   criterion="Full held-out EMA denoising loss. Robot task success is not measured.",
                   raw_training_curve="Unsmoothed stochastic minibatch loss", ema_validation_curve="Fixed noise seed; all held-out windows")
    (target / "summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("Charts and evaluation saved:", target, flush=True)


if __name__ == "__main__":
    main()
