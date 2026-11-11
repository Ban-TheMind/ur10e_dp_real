"""Device adapters. Recording and control never wait for image encoding."""

import copy
import json
import subprocess
import threading
import time
from pathlib import Path
import numpy as np
from .geometry import TimedActions, bounded_pose


class Worker:
    def __init__(self):
        self.stop_event = threading.Event()
        self.error = None
        self.lock = threading.Lock()

    def launch(self):
        self.thread = threading.Thread(target=self._guard, daemon=True)
        self.thread.start()

    def _guard(self):
        try:
            self.run()
        except BaseException as error:
            self.error = error
            self.stop_event.set()

    def check(self):
        if self.error is not None:
            raise RuntimeError(
                "Device worker failed: {}".format(self.error)
            ) from self.error

    def close(self):
        self.stop_event.set()
        if hasattr(self, "thread"):
            self.thread.join(timeout=3)


class Gripper(Worker):
    def __init__(self, cfg, timed_actions=None):
        super().__init__()
        self.cfg = cfg
        self.pending = None
        self.timed_actions = timed_actions
        self.sample = None
        self.last_width = None
        self.launch()

    def transact(self, operation, width=0, plan_version=None):
        c = self.cfg
        args = [
            str(Path(c["helper"]).resolve()),
            operation,
            c["port"],
            str(c["baudrate"]),
            str(c["slave_id"]),
            str(c["serial_timeout_s"]),
            str(c["action_timeout_s"]),
            str(c["speed_raw"]),
            str(width * 1000),
            str(c["force_raw"]),
        ]
        # 旧实现：先取出单槽宽度，再阻塞 subprocess.run；部署计划可能已取消该宽度。
        # result = subprocess.run(
        #     args, capture_output=True, text=True, timeout=2 * c["action_timeout_s"] + 10
        # )
        # 新实现：部署命令在启动发送 helper 前，持计划锁重新核对版本及当前到时宽度。
        # 锁仅覆盖核对和 Popen，不覆盖串口等待，避免阻塞轨迹替换与 125 Hz 控制。
        if plan_version is not None:
            with self.timed_actions.changed:
                _, active_width, _ = self.timed_actions.sample(time.monotonic())
                if (self.stop_event.is_set()
                        or plan_version != self.timed_actions.version
                        or active_width != width):
                    return None
                process = subprocess.Popen(args, stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE, text=True)
        else:
            process = subprocess.Popen(args, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True)
        try:
            stdout, stderr = process.communicate(timeout=2 * c["action_timeout_s"] + 10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise
        if process.returncode:
            raise IOError(stderr.strip())
        status = json.loads(stdout)
        if status["fault"]:
            raise IOError("Gripper fault {}".format(status["fault"]))
        return dict(
            status,
            width_m=(255 - status["position"]) / 255 * 0.085,
            receive_monotonic=time.monotonic(),
        )

    def command(self, width):
        width = float(width)
        if not np.isfinite(width) or not 0 <= width <= 0.085:
            raise ValueError("Robotiq width must be 0..0.085m")
        with self.lock:
            # Avoid repeatedly rewriting identical commands; actual state is still polled.
            if self.last_width is None or abs(width - self.last_width) > 0.001:
                self.pending = width
                self.last_width = width

    def run(self):
        if self.timed_actions is not None:
            self.run_timed()
            return
        # 旧单槽逻辑仍用于键盘采集：只需最近一次人工宽度，无未来动作时间轴。
        # 部署路径在上方转入 run_timed()，不读写 pending，也不按轮询周期等待动作。
        while not self.stop_event.is_set():
            with self.lock:
                width, self.pending = self.pending, None
            status = self.transact("status" if width is None else "command", width or 0)
            with self.lock:
                self.sample = status
            self.stop_event.wait(1 / self.cfg["poll_hz"])

    def run_timed(self):
        schedule = self.timed_actions
        next_status = time.monotonic()
        sent_width = None
        while not self.stop_event.is_set():
            with schedule.changed:
                now = time.monotonic()
                _, width, next_point = schedule.sample(now)
                # 旧部署也调用 command(width)，依赖 pending 单槽和状态轮询。
                # width, self.pending = self.pending, None
                # 新实现从当前有效计划取到时宽度；被替换的未来闭合/释放不会补发。
                version = schedule.version
                send = width is not None and (
                    sent_width is None or abs(width - sent_width) > 0.001
                )
                if not send and now < next_status:
                    wake = next_status if next_point is None else min(next_status, next_point)
                    # 到期时间和计划替换都能唤醒，宽度变化无需等下一次状态轮询。
                    schedule.changed.wait(max(0, wake - now))
                    continue
            if self.stop_event.is_set():
                break
            status = (self.transact("command", width, plan_version=version)
                      if send else self.transact("status"))
            if status is None:  # 发送边界发现计划已变，立即重新选择当前目标。
                continue
            if send:
                sent_width = width
            with self.lock:
                self.sample = status
            next_status = time.monotonic() + 1 / self.cfg["poll_hz"]


class Camera(Worker):
    def __init__(self, cfg):
        super().__init__()
        import pyrealsense2 as rs

        self.rs, self.sample = rs, None
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(cfg["serial"])
        config.enable_stream(
            rs.stream.color, cfg["width"], cfg["height"], rs.format.rgb8, cfg["fps"]
        )
        self.pipeline.start(config)
        self.launch()

    def run(self):
        try:
            while not self.stop_event.is_set():
                frame = self.pipeline.wait_for_frames(timeout_ms=1000).get_color_frame()
                if frame:
                    sample = {
                        "rgb": np.asanyarray(frame.get_data()).copy(),
                        "source_timestamp_ms": frame.get_timestamp(),
                        "timestamp_domain": str(frame.get_frame_timestamp_domain()),
                        "receive_monotonic": time.monotonic(),
                        "frame_number": frame.get_frame_number(),
                    }
                    with self.lock:
                        self.sample = sample
        finally:
            self.pipeline.stop()


class Robot(Worker):
    def __init__(self, cfg, motion=True, timed_actions=None):
        super().__init__()
        from rtde_receive import RTDEReceiveInterface

        self.cfg = cfg
        self.timed_actions = timed_actions
        self.receive = RTDEReceiveInterface(cfg["robot_ip"])
        self.control = None
        try:
            if motion:
                from rtde_control import RTDEControlInterface

                self.control = RTDEControlInterface(cfg["robot_ip"])
            self.target = np.asarray(self.receive.getActualTCPPose())
            self.last_command = time.monotonic()
            self.sample = None
            self.launch()
        except BaseException:
            if self.control:
                self.control.disconnect()
            self.receive.disconnect()
            raise

    def validate_target(self, pose):
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (6,) or not np.isfinite(pose).all():
            raise ValueError("Expected finite 6D TCP target")
        if np.any(pose[:3] < self.cfg["workspace_min_m"]) or np.any(
            pose[:3] > self.cfg["workspace_max_m"]
        ):
            raise ValueError(
                "TCP target outside configured workspace: {}".format(pose[:3])
            )
        return pose

    def command(self, pose):
        pose = self.validate_target(pose)
        with self.lock:
            self.target, self.last_command = pose.copy(), time.monotonic()

    def run(self):
        c = self.cfg
        dt = 1 / c["control_hz"]
        servo_pose = self.target.copy()
        try:
            while not self.stop_event.is_set():
                start = time.monotonic()
                if not self.receive.isConnected():
                    raise ConnectionError("RTDE receive connection lost")
                safety = self.receive.getSafetyStatusBits()
                # Protective stop, safeguard stop and emergency-stop bits from UR RTDE.
                if safety & ((1 << 2) | (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)):
                    raise RuntimeError(
                        "Robot is stopped; safety_status_bits={}".format(safety)
                    )
                sample = {
                    "tcp_pose": self.receive.getActualTCPPose(),
                    "joint_position": self.receive.getActualQ(),
                    "joint_velocity": self.receive.getActualQd(),
                    "controller_timestamp": self.receive.getTimestamp(),
                    "receive_monotonic": time.monotonic(),
                    "safety_status_bits": safety,
                }
                with self.lock:
                    self.sample = sample
                    target, last_command = self.target.copy(), self.last_command
                # 旧实现仅追踪 command() 的单个 target：
                # target, age = self.target.copy(), start - self.last_command
                # 新部署从共享指令轨迹插值，不用实测 TCP 重建轨迹起点。
                if self.timed_actions is not None:
                    schedule = self.timed_actions
                    with schedule.changed:
                        # 读取状态期间可能刚提交新块；用锁内当前时间，避免以早于
                        # 新交接点的 start 采样而短暂退回初始化 target。
                        scheduled, _, _ = schedule.sample(time.monotonic())
                        if scheduled is not None:
                            target = scheduled
                        if schedule.submitted_at is not None:
                            last_command = schedule.submitted_at
                age = start - last_command
                if self.control:
                    if age > c["command_timeout_s"]:
                        raise TimeoutError(
                            "No control update for {} seconds".format(age)
                        )
                    servo_pose = bounded_pose(
                        servo_pose,
                        target,
                        c["translation_speed_m_s"] * dt,
                        c["rotation_speed_rad_s"] * dt,
                    )
                    accepted = self.control.servoL(
                        servo_pose.tolist(),
                        c["translation_speed_m_s"],
                        0.2,
                        dt,
                        0.1,
                        300,
                    )
                    if not accepted:
                        raise RuntimeError("RTDE controller rejected servoL command")
                self.stop_event.wait(max(0, dt - (time.monotonic() - start)))
        finally:
            if self.control:
                try:
                    self.control.servoStop()
                finally:
                    self.control.stopScript()
                    self.control.disconnect()
            self.receive.disconnect()


class Rig:
    """Snapshot includes per-source times; no claim of camera hardware synchronization."""

    def __init__(self, cfg, motion=True, timed=False):
        self.cfg, self.devices = cfg, []
        self.timed_actions = TimedActions() if timed else None
        try:
            # Cameras/gripper are initialized before the motion watchdog starts.
            self.cameras = {}
            for name, spec in cfg["cameras"].items():
                self.cameras[name] = Camera(spec)
                self.devices.append(self.cameras[name])
            self.gripper = Gripper(cfg["gripper"], timed_actions=self.timed_actions)
            self.devices.append(self.gripper)
            self.robot = Robot(cfg, motion=motion, timed_actions=self.timed_actions)
            self.devices.append(self.robot)
            deadline = time.monotonic() + 10
            while any(d.sample is None for d in self.devices):
                for d in self.devices:
                    d.check()
                if time.monotonic() > deadline:
                    raise TimeoutError("No initial observation from a device")
                if motion:
                    self.robot.command(self.robot.target)
                time.sleep(0.01)
        except BaseException:
            self.close()
            raise

    def snapshot(self):
        now = time.monotonic()
        values = []
        for d in self.devices:
            d.check()
            with d.lock:
                sample = copy.deepcopy(d.sample)
            if (
                sample is None
                or now - sample["receive_monotonic"] > self.cfg["observation_max_age_s"]
            ):
                raise TimeoutError("Missing/stale device observation")
            values.append(sample)
        sample = values[-1]
        sample["gripper"] = values[-2]
        sample["gripper_width_m"] = values[-2]["width_m"]
        sample["cameras"] = dict(zip(self.cameras, values[:-2]))
        sample["snapshot_monotonic"] = now
        return sample

    def command(self, pose, width):
        self.robot.command(pose)
        self.gripper.command(width)

    def schedule(self, points, lead_time):
        """Atomically replace future TCP and width targets on one monotonic clock."""
        schedule = self.timed_actions
        with schedule.changed:
            now = time.monotonic()
            future = []
            # 提交边界再过滤一次，避免解码/校验期间已过期的点进入控制器。
            for deadline, pose, width in points:
                if deadline > now + lead_time:
                    pose = self.robot.validate_target(pose).copy()
                    if not np.isfinite(width) or not 0 <= width <= 0.085:
                        raise ValueError("Robotiq width must be 0..0.085m")
                    future.append((deadline, pose, float(width)))
            return schedule.submit(
                now,
                future,
                self.robot.target.copy(),
                self.cfg["translation_speed_m_s"],
                self.cfg["rotation_speed_rad_s"],
            )

    def scheduled_target(self, now):
        with self.timed_actions.changed:
            pose, width, _ = self.timed_actions.sample(now)
            return pose, width

    def close(self):
        # Stop robot first, then drain camera/gripper workers.
        for device in reversed(self.devices):
            device.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
