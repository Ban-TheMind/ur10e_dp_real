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
    returning_home = False
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
                if returning_home:
                    # 归位不写入示教；持续检查设备、刷新 watchdog，Esc 始终可用。
                    rig.keepalive()
                    if not rig.homing:
                        returning_home = False
                        print("Home complete; ready to collect", flush=True)
                    keyboard.draw(sample, [
                        "RETURNING HOME | Esc stop" if returning_home else "HOME COMPLETE",
                        "Recording and jogging resume after Home",
                        "Gripper width unchanged | O open after arrival",
                    ])
                    time.sleep(max(0, 1 / cfg["sample_hz"] - (time.monotonic() - start)))
                    # 完成后再取一份新快照，避免用归位完成前的旧读数生成目标。
                    continue
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
                if "home" in events:
                    if writer is not None:
                        print("Save/finish the episode before pressing Home", flush=True)
                    else:
                        returning_home = rig.home()
                        if returning_home:
                            print("Returning to fixed Home joints; Esc stops", flush=True)
                            continue
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
                        "Esc stop | Home reset (idle) | {} | {}".format(
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
    # 旧实现：主循环逐点消费队列；队列为空只代表目标已下发，不代表已到达。
    # 新实现：整块交给共享时间计划，125 Hz 控制线程在推理期间继续执行。
    # plan = collections.deque()
    future = None
    dt = 1 / cfg["sample_hz"]
    lead_time = max(0.01, 1 / cfg["control_hz"])
    # 部署初值：4 个策略周期（20 Hz 下为 200 ms），不改变模型的 2/10/8 配置。
    replan_interval = 4 * dt
    next_inference = 0.0
    fixed = cfg["fixed_orientation"]
    outcome = None

    def predict(obs, obs_time):
        started = time.monotonic()
        tensors = {
            k: torch.from_numpy(np.stack([o[k] for o in obs]))[None].to(device)
            for k in obs[0]
        }
        with torch.inference_mode():
            actions = model.predict_action(tensors)["action"][0].cpu().numpy()
        return actions, obs_time, time.monotonic() - started

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        with Rig(cfg, timed=True) as rig:
            writer = EpisodeWriter(
                output,
                cfg,
                kind="evaluation",
                metadata={
                    "checkpoint_sha256": source_hash,
                    "inference_steps": model.num_inference_steps,
                    "timing": "observation-relative waypoints on a shared arm/gripper clock",
                    "replan_interval_s": replan_interval,
                    "action_lead_time_s": lead_time,
                },
            )
            start = time.monotonic()
            initial = rig.snapshot()
            fixed_rotation = np.asarray(initial["tcp_pose"])[3:].copy()
            width = initial["gripper_width_m"]
            latency = 0.0
            while time.monotonic() - start < duration:
                tick = time.monotonic()
                sample = rig.snapshot()
                events, _, _ = keyboard.poll()
                if "escape" in events:
                    break
                if "return" in events or "backspace" in events:
                    outcome = "return" in events
                    break
                obs_time = sample["snapshot_monotonic"]
                history.append((obs_time, observation(sample, frame)))

                # 旧实现：收到动作后建立本地队列，清空后才再次推理，形成块间空窗。
                # 为什么出问题：等待分支把实测 TCP 当成新目标，撤掉仍在追踪的策略目标。
                # 新实现：下面按 200 ms 重规划；等待期间不发替代命令，旧轨迹继续走。
                # if future is not None and future.done():
                #     actions, latency = future.result()
                #     plan = collections.deque(actions)
                #     future = None
                # 旧实现：必须耗尽动作队列才推理；新实现以观测时间控制重规划间隔。
                # if future is None and not plan:
                #     obs = list(history)
                #     obs = [obs[0]] * (model.n_obs_steps - len(obs)) + obs
                #     future = pool.submit(predict, obs)
                # 旧实现：等待时每轮写回实测 TCP，覆盖上一个策略目标，是已复现的错误。
                # target = np.asarray(sample["tcp_pose"])
                # 旧实现：每 50 ms 弹一个目标并立刻下发；新实现成批提交带时间的目标。
                # if plan:
                #     raw_action = plan.popleft()
                #     target, width = frame.decode_action(raw_action)
                #     if fixed:
                #         target[3:] = fixed_rotation
                # rig.command(target, width)

                accepted = None
                if future is not None and future.done():
                    actions, reference_time, latency = future.result()
                    future = None
                    points = []
                    for index, action in enumerate(actions):
                        # 本地 Transformer predict_action() 已切片 [To-1:To-1+8]；
                        # 第 0 个返回动作属于最新观测时刻，不能再加 To-1 偏移。
                        # TCP 和宽度共享该名义时间；过期动作直接丢弃，不平移补发。
                        deadline = reference_time + index * dt
                        if deadline <= time.monotonic() + lead_time:
                            continue
                        pose, target_width = frame.decode_action(action)
                        if fixed:
                            pose[3:] = fixed_rotation
                        points.append((deadline, pose, target_width))
                    accepted = rig.schedule(points, lead_time)
                if (
                    future is None
                    and len(history) == model.n_obs_steps
                    and obs_time >= next_inference
                ):
                    future = pool.submit(predict, [obs for _, obs in history], obs_time)
                    next_inference = obs_time + replan_interval

                # 这里只记录指令轨迹，不调用 rig.command()；无新块时保持最后目标。
                # 全过期块不会刷新 watchdog，超时仍由 Robot.run() 原停止路径处理。
                target, scheduled_width = rig.scheduled_target(time.monotonic())
                if target is None:
                    target = initial["tcp_pose"]
                if scheduled_width is not None:
                    width = scheduled_width
                writer.append(
                    sample,
                    target,
                    width,
                    {
                        "trajectory_accepted": accepted,
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
                time.sleep(max(0, dt - (time.monotonic() - tick)))
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
