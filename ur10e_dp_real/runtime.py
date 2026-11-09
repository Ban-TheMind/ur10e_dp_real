"""Keyboard collection and asynchronous receding-horizon policy execution."""

import collections
import concurrent.futures
import json
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .data import observation, RGB_KEYS
from .devices import Rig
from .geometry import PolicyFrame
from .recording import EpisodeWriter


class Keyboard:
    def __init__(self):
        import pygame

        self.pg = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((800, 400))
        pygame.display.set_caption("UR10e DP | focus this window | Esc stops")
        self.font = pygame.font.Font(None, 22)

    def poll(self):
        pg = self.pg
        events = []
        for event in pg.event.get():
            if event.type == pg.QUIT:
                events.append("escape")
            elif event.type == pg.KEYDOWN:
                events.append(pg.key.name(event.key))
        held = pg.key.get_pressed() if pg.key.get_focused() else [False] * 512

        def axis(positive, negative):
            return int(held[getattr(pg, "K_" + positive)]) - int(
                held[getattr(pg, "K_" + negative)]
            )

        return (
            events,
            np.array([axis("w", "s"), axis("a", "d"), axis("q", "e")]),
            np.array([axis("i", "k"), axis("j", "l"), axis("u", "h")]),
        )

    def draw(self, sample, text):
        self.screen.fill((20, 20, 20))
        for i, name in enumerate(RGB_KEYS):
            array = sample["cameras"][name]["rgb"]
            surface = self.pg.surfarray.make_surface(array.transpose(1, 0, 2))
            self.screen.blit(self.pg.transform.scale(surface, (400, 300)), (i * 400, 0))
        for i, line in enumerate(text):
            self.screen.blit(
                self.font.render(line, True, (240, 240, 240)), (10, 310 + i * 22)
            )
        self.pg.display.flip()

    def close(self):
        self.pg.quit()


def collect(cfg, output):
    keyboard = Keyboard()
    writer = None
    save_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    saves = []
    fixed = cfg["fixed_orientation"]
    try:
        with Rig(cfg) as rig:
            width = rig.snapshot()["gripper_width_m"]
            while True:
                start = time.monotonic()
                sample = rig.snapshot()
                for saved in list(saves):
                    if saved.done():
                        print("Saved", saved.result(), flush=True)
                        saves.remove(saved)
                events, translation, rotation = keyboard.poll()
                if "escape" in events:
                    break
                if "r" in events:
                    fixed = not fixed
                if "o" in events:
                    width = 0.085
                if "p" in events:
                    width = 0.0
                if "c" in events and writer is None:
                    writer = EpisodeWriter(
                        output,
                        cfg,
                        metadata={"orientation_mode": "fixed" if fixed else "six_dof"},
                    )
                    print("Recording", writer.path, flush=True)
                for key, success in [("return", True), ("backspace", False)]:
                    if key in events and writer:
                        saves.append(save_pool.submit(writer.close, success=success))
                        writer = None
                # Each command is relative to measured pose: release does not chase an old distant target.
                target = np.asarray(sample["tcp_pose"], dtype=float)
                dt = 1 / cfg["sample_hz"]
                target[:3] += translation * cfg["translation_speed_m_s"] * dt
                if not fixed:
                    dr = Rotation.from_rotvec(
                        rotation * cfg["rotation_speed_rad_s"] * dt
                    )
                    target[3:] = (dr * Rotation.from_rotvec(target[3:])).as_rotvec()
                rig.command(target, width)
                if writer:
                    writer.append(sample, target, width, {"orientation_fixed": fixed})
                keyboard.draw(
                    sample,
                    [
                        "W/S X | A/D Y | Q/E Z | O open | P close | R rotation mode",
                        "I/K J/L U/H rotate | C record | Enter save success | Backspace save failure",
                        "Esc stop | {} | {}".format(
                            "fixed orientation" if fixed else "6DoF",
                            "RECORDING" if writer else "idle",
                        ),
                    ],
                )
                time.sleep(max(0, dt - (time.monotonic() - start)))
    finally:
        # Rig exits before image encoding drains, so stopping never waits for the writer.
        if writer:
            writer.close(complete=False)
        save_pool.shutdown(wait=True)
        for saved in saves:
            saved.result()
        keyboard.close()


def rollout(cfg, checkpoint, output, device="cuda", inference_steps=None, duration=30):
    import torch
    from .model import load_policy, sha256

    # Loading may take seconds; it happens before connecting the motion controller.
    model, _, payload = load_policy(checkpoint, device, inference_steps=inference_steps)
    frame = PolicyFrame(cfg["policy_frame"])
    if "contract" in payload:
        expected = {"policy_frame": cfg["policy_frame"], "sample_hz": cfg["sample_hz"]}
        if payload["contract"] != expected:
            raise ValueError(
                "Site frame/time contract differs from the fine-tuning data"
            )
    source_hash = sha256(checkpoint)
    keyboard = Keyboard()
    writer = None
    history = collections.deque(maxlen=model.n_obs_steps)
    plan = collections.deque()
    future = None
    fixed = cfg["fixed_orientation"]
    outcome = None

    def predict(obs):
        started = time.monotonic()
        tensors = {
            k: torch.from_numpy(np.stack([o[k] for o in obs]))[None].to(device)
            for k in obs[0]
        }
        with torch.inference_mode():
            actions = model.predict_action(tensors)["action"][0].cpu().numpy()
        return actions, time.monotonic() - started

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        with Rig(cfg) as rig:
            writer = EpisodeWriter(
                output,
                cfg,
                kind="evaluation",
                metadata={
                    "checkpoint_sha256": source_hash,
                    "inference_steps": model.num_inference_steps,
                    "timing": "asynchronous inference; commands execute after prediction completion",
                },
            )
            start = time.monotonic()
            initial = rig.snapshot()
            fixed_rotation = np.asarray(initial["tcp_pose"])[3:].copy()
            width = initial["gripper_width_m"]
            latency = 0.0
            raw_action = None
            while time.monotonic() - start < duration:
                tick = time.monotonic()
                sample = rig.snapshot()
                events, _, _ = keyboard.poll()
                if "escape" in events:
                    break
                if "return" in events or "backspace" in events:
                    outcome = "return" in events
                    break
                history.append(observation(sample, frame))
                if future is not None and future.done():
                    actions, latency = future.result()
                    plan = collections.deque(actions)
                    future = None
                if future is None and not plan:
                    obs = list(history)
                    obs = [obs[0]] * (model.n_obs_steps - len(obs)) + obs
                    future = pool.submit(predict, obs)
                target = np.asarray(sample["tcp_pose"])
                if plan:
                    raw_action = plan.popleft()
                    target, width = frame.decode_action(raw_action)
                    if fixed:
                        target[3:] = fixed_rotation
                rig.command(target, width)
                writer.append(
                    sample,
                    target,
                    width,
                    {
                        "raw_policy_action": raw_action,
                        "inference_latency_s": latency,
                        "orientation_fixed": fixed,
                    },
                )
                keyboard.draw(
                    sample,
                    [
                        "POLICY RUNNING | Esc stop | Enter success | Backspace failure",
                        "inference {:.3f}s | {:.1f}s".format(
                            latency, time.monotonic() - start
                        ),
                    ],
                )
                time.sleep(max(0, 1 / cfg["sample_hz"] - (time.monotonic() - tick)))
    finally:
        if writer:
            print("Saved", writer.close(success=outcome))
        keyboard.close()
        pool.shutdown(wait=True)


def inspect(cfg):
    # Observation command does not instantiate RTDEControl or send gripper writes.
    with Rig(cfg, motion=False) as rig:
        sample = rig.snapshot()
        sample["cameras"] = {
            k: {a: b for a, b in v.items() if a != "rgb"}
            for k, v in sample["cameras"].items()
        }
        print(json.dumps(sample, indent=2))
