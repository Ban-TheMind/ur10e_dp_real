"""Episode-level split and time-resampled sequence windows for the source Lift policy."""

import json
from pathlib import Path
import numpy as np
from PIL import Image
from .geometry import PolicyFrame
from .recording import check_episode

RGB_KEYS = ("agentview_image", "robot0_eye_in_hand_image")


def image_tensor(rgb):
    return (
        np.asarray(
            Image.fromarray(rgb).resize((84, 84), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ).transpose(2, 0, 1)
        / 255
    )


def observation(sample, frame):
    result = frame.observations(sample)
    for name in RGB_KEYS:
        result[name] = image_tensor(sample["cameras"][name]["rgb"])
    return result


def discover(root):
    episodes = []
    for p in sorted(Path(root).glob("episode_*")):
        if not (p / "episode.json").exists():
            continue
        m = json.loads((p / "episode.json").read_text())
        if m["complete"] and m["kind"] == "demonstration" and m["success"] is True:
            check_episode(p)
            episodes.append(p)
    if not episodes:
        raise ValueError("No complete successful demonstrations under {}".format(root))
    return episodes


def split_episodes(paths, fraction, seed):
    if not 0 <= fraction < 1:
        raise ValueError("val_fraction must be in [0,1)")
    if fraction > 0 and len(paths) < 2:
        raise ValueError(
            "Need at least two episodes for a held-out split; use val_fraction=0 for a fitting check"
        )
    order = np.random.RandomState(seed).permutation(len(paths))
    nval = min(len(paths) - 1, max(1, round(len(paths) * fraction))) if fraction else 0
    val = set(order[:nval])
    return [p for i, p in enumerate(paths) if i not in val], [
        p for i, p in enumerate(paths) if i in val
    ]


class Episodes:
    def __init__(self, paths, horizon, n_obs_steps):
        self.episodes, self.index = [], []
        self.horizon, self.n_obs_steps = horizon, n_obs_steps
        self.contract = None
        for path in paths:
            meta = json.loads((Path(path) / "episode.json").read_text())
            cfg = meta["config"]
            contract = {
                "policy_frame": cfg["policy_frame"],
                "sample_hz": cfg["sample_hz"],
            }
            if self.contract is not None and self.contract != contract:
                raise ValueError(
                    "Mixed frame or sampling contracts; convert explicitly first"
                )
            self.contract = contract
            frame = PolicyFrame(cfg["policy_frame"])
            rows = [
                json.loads(s)
                for s in (Path(path) / "samples.jsonl").read_text().splitlines()
            ]
            times = np.array([r["sample"]["snapshot_monotonic"] for r in rows])
            grid = np.arange(times[0], times[-1] + 1e-9, 1 / cfg["sample_hz"])
            indices = np.maximum(0, np.searchsorted(times, grid, side="right") - 1)
            sampled = [rows[i] for i in indices]
            self.episodes.append((Path(path), sampled, frame))
            # Allow repeated first/last states, matching DP sequence padding semantics.
            self.index.extend((len(self.episodes) - 1, t) for t in range(len(sampled)))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        import torch

        ep, t = self.index[index]
        path, rows, frame = self.episodes[ep]
        start = t - self.n_obs_steps + 1
        sequence = [
            rows[min(max(i, 0), len(rows) - 1)]
            for i in range(start, start + self.horizon)
        ]
        observations = []
        for row in sequence[: self.n_obs_steps]:
            s = dict(row["sample"])
            s["cameras"] = {}
            for name in RGB_KEYS:
                with Image.open(
                    path / row["sample"]["cameras"][name]["image"]
                ) as image:
                    s["cameras"][name] = {"rgb": np.array(image.convert("RGB"))}
            observations.append(observation(s, frame))
        obs = {
            k: torch.from_numpy(np.stack([o[k] for o in observations]))
            for k in observations[0]
        }
        actions = np.stack(
            [
                frame.encode_action(r["target_tcp_pose"], r["target_gripper_width_m"])
                for r in sequence
            ]
        )
        return {"obs": obs, "action": torch.from_numpy(actions)}
