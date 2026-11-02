"""Append-only raw episodes; training windows are derived without touching originals."""

import json
import queue
import threading
import time
from pathlib import Path
import numpy as np
from PIL import Image


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class EpisodeWriter:
    def __init__(self, root, cfg, kind="demonstration", metadata=None):
        self.path = Path(root) / (
            "episode_" + time.strftime("%Y%m%d_%H%M%S") + "_" + str(time.time_ns())[-6:]
        )
        self.path.mkdir(parents=True, exist_ok=False)
        self.meta = {
            "schema_version": 1,
            "kind": kind,
            "config": cfg,
            "metadata": metadata or {},
            "complete": False,
            "success": None,
        }
        self._save_meta()
        self.queue = queue.Queue(maxsize=128)
        self.error, self.count = None, 0
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _save_meta(self):
        target = self.path / "episode.json"
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(jsonable(self.meta), indent=2) + "\n")
        temp.replace(target)

    def append(self, sample, target_pose, target_width, extra=None):
        if self.error:
            raise RuntimeError("Recording failed") from self.error
        item = {
            "sample": sample,
            "target_tcp_pose": np.array(target_pose),
            "target_gripper_width_m": float(target_width),
            "command_monotonic": time.monotonic(),
            "extra": extra or {},
        }
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            raise RuntimeError("Image writer backlog full; recording interrupted")

    def _run(self):
        try:
            with (self.path / "samples.jsonl").open("w") as stream:
                while True:
                    item = self.queue.get()
                    if item is None:
                        break
                    # Clone metadata; caller still needs its RGB arrays for rendering/inference.
                    item["sample"] = dict(item["sample"])
                    item["sample"]["cameras"] = {
                        k: dict(v) for k, v in item["sample"]["cameras"].items()
                    }
                    for name, camera in item["sample"]["cameras"].items():
                        rel = Path(name) / "{:06d}.png".format(self.count)
                        (self.path / name).mkdir(exist_ok=True)
                        Image.fromarray(camera.pop("rgb")).save(self.path / rel)
                        camera["image"] = str(rel)
                    stream.write(json.dumps(jsonable(item)) + "\n")
                    stream.flush()
                    self.count += 1
        except BaseException as error:
            self.error = error

    def close(self, success=None, complete=True):
        if self.worker.is_alive():
            while self.worker.is_alive():
                try:
                    self.queue.put(None, timeout=0.1)
                    break
                except queue.Full:
                    pass
            self.worker.join()
        self.meta.update(
            complete=bool(complete and self.error is None),
            success=success,
            samples=self.count,
        )
        self._save_meta()
        if self.error:
            raise RuntimeError("Recording writer failed") from self.error
        return self.path


def check_episode(path):
    path = Path(path)
    meta = json.loads((path / "episode.json").read_text())
    rows = [
        json.loads(line) for line in (path / "samples.jsonl").read_text().splitlines()
    ]
    if not meta["complete"] or not rows or meta["samples"] != len(rows):
        raise ValueError("Incomplete or empty episode {}".format(path))
    previous = -float("inf")
    ages, intervals = [], []
    for row in rows:
        s = row["sample"]
        t = s["snapshot_monotonic"]
        if t <= previous:
            raise ValueError("Non-increasing observation times")
        if np.isfinite(previous):
            intervals.append(t - previous)
        previous = t
        if (
            not np.isfinite(row["target_tcp_pose"]).all()
            or not np.isfinite(s["tcp_pose"]).all()
        ):
            raise ValueError("Non-finite pose")
        for name, camera in s["cameras"].items():
            with Image.open(path / camera["image"]) as image:
                image.verify()
            ages.append(t - camera["receive_monotonic"])
    return {
        "episode": str(path),
        "samples": len(rows),
        "success": meta["success"],
        "median_dt_s": float(np.median(intervals)) if intervals else None,
        "max_camera_age_s": max(ages),
        "kind": meta["kind"],
    }
