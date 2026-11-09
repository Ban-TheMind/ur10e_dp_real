"""Synthetic fixture only: verify native checkpoint adaptation and exact CPU resume."""

import json
import sys
import time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_pipeline import make_episode
from ur10e_dp_real.training import train

root = Path(__file__).resolve().parents[1]
out = root / "outputs" / "smoke" / str(time.time_ns())
out.mkdir(parents=True, exist_ok=True)
data = out / "synthetic_data"
data.mkdir(exist_ok=True)
cfg = json.loads((root / "configs/site.example.json").read_text())
if not list(data.glob("episode_*")):
    for i in range(3):
        make_episode(data, cfg, i)
settings = json.loads((root / "configs/finetune.json").read_text())
settings.update(
    max_updates=2,
    batch_size=1,
    device="cpu",
    save_updates=[0, 1, 2],
    validate_every=1,
    validation_batches=1,
    warmup_updates=1,
)
torch.set_num_threads(2)
source = root / "weights/local_lift_3050.ckpt"
train(source, data, out / "uninterrupted", settings)
train(
    source,
    data,
    out / "resumed",
    settings,
    resume=out / "uninterrupted/update_000001.pt",
)
a = torch.load(
    out / "uninterrupted/update_000002.pt", weights_only=False, map_location="cpu"
)
b = torch.load(out / "resumed/update_000002.pt", weights_only=False, map_location="cpu")
matches = {
    name: all(torch.equal(a[name][k], b[name][k]) for k in a[name])
    for name in ["model", "ema_model"]
}
assert all(matches.values()), matches
report = {
    "synthetic_data": True,
    "updates": 2,
    "device": "cpu",
    "exact_resume": matches,
    "train": json.loads((out / "uninterrupted/run.json").read_text()),
    "metrics": [
        json.loads(s)
        for s in (out / "uninterrupted/metrics.jsonl").read_text().splitlines()
    ],
}
(out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"synthetic_smoke": "passed", "exact_resume": matches}))
