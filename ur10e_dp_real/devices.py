"""Device adapters. Recording and control never wait for image encoding."""

import copy
import json
import subprocess
import threading
import time
from pathlib import Path
import numpy as np
from .geometry import bounded_pose


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
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.pending = None
        self.sample = None
        self.last_width = None
        self.launch()

    def transact(self, operation, width=0):
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
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=2 * c["action_timeout_s"] + 10
        )
        if result.returncode:
            raise IOError(result.stderr.strip())
        status = json.loads(result.stdout)
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
        while not self.stop_event.is_set():
            with self.lock:
                width, self.pending = self.pending, None
            status = self.transact("status" if width is None else "command", width or 0)
            with self.lock:
                self.sample = status
            self.stop_event.wait(1 / self.cfg["poll_hz"])


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
    def __init__(self, cfg, motion=True):
        super().__init__()
        from rtde_receive import RTDEReceiveInterface

        self.cfg = cfg
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

    def command(self, pose):
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (6,) or not np.isfinite(pose).all():
            raise ValueError("Expected finite 6D TCP target")
        if np.any(pose[:3] < self.cfg["workspace_min_m"]) or np.any(
            pose[:3] > self.cfg["workspace_max_m"]
        ):
            raise ValueError(
                "TCP target outside configured workspace: {}".format(pose[:3])
            )
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
                    target, age = self.target.copy(), start - self.last_command
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

    def __init__(self, cfg, motion=True):
        self.cfg, self.devices = cfg, []
        try:
            # Cameras/gripper are initialized before the motion watchdog starts.
            self.cameras = {}
            for name, spec in cfg["cameras"].items():
                self.cameras[name] = Camera(spec)
                self.devices.append(self.cameras[name])
            self.gripper = Gripper(cfg["gripper"])
            self.devices.append(self.gripper)
            self.robot = Robot(cfg, motion=motion)
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

    def close(self):
        # Stop robot first, then drain camera/gripper workers.
        for device in reversed(self.devices):
            device.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
