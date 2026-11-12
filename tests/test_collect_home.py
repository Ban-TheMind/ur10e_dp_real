"""Home transitions through the actual collect loop and robot worker, without hardware."""

import json
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from ur10e_dp_real import devices, runtime


class CollectHomeTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.cfg = json.loads((root / "configs/site.example.json").read_text())
        self.clock = types.SimpleNamespace(now=0.0)
        self.clock.monotonic = lambda: self.clock.now
        self.receiver = mock.MagicMock()
        self.receiver.getActualTCPPose.return_value = [0.4, 0, 0.2, 0, 0, 0]
        self.receiver.getActualQ.return_value = [0] * 6
        self.receiver.getSafetyStatusBits.return_value = 0
        self.control = mock.MagicMock()
        modules = {
            "rtde_receive": types.SimpleNamespace(RTDEReceiveInterface=lambda _: self.receiver),
            "rtde_control": types.SimpleNamespace(RTDEControlInterface=lambda _: self.control),
        }
        with mock.patch.dict("sys.modules", modules), mock.patch.object(
            devices.Robot, "launch"
        ), mock.patch.object(devices, "time", self.clock):
            self.robot = devices.Robot(self.cfg)

    def run_worker(self, wait=None):
        def advance(delay):
            self.clock.now += delay

        with mock.patch.object(devices, "time", self.clock), mock.patch.object(
            self.robot.stop_event, "wait", side_effect=wait or advance
        ):
            self.robot.run()

    def request_home(self):
        with mock.patch.object(devices, "time", self.clock):
            return self.robot.request_home()

    def test_home_reuses_joint_target_and_resumes_at_arrival_pose(self):
        self.assertTrue(self.request_home())
        self.assertFalse(self.request_home())
        arrived_pose = [0.5, 0.1, 0.3, 0, 0, 0]
        calls = []

        def progress():
            calls.append("progress")
            if len(calls) == 1:
                return 0
            self.receiver.getActualQ.return_value = np.deg2rad(devices.HOME_JOINTS_DEG)
            self.receiver.getActualTCPPose.return_value = arrived_pose
            return -1

        self.control.getAsyncOperationProgress.side_effect = progress

        def servo(*args):
            self.assertEqual(len(calls), 2)
            self.assertFalse(self.robot.homing)
            self.robot.stop_event.set()
            return True

        self.control.servoL.side_effect = servo
        self.run_worker()
        self.control.moveJ.assert_called_once_with(
            np.deg2rad([-180, -90, 127, -123, 270, 0]).tolist(), 0.2, 0.3, True
        )
        self.assertEqual(self.control.method_calls[0], mock.call.servoStop())
        self.control.servoL.assert_called_once()
        np.testing.assert_allclose(self.control.servoL.call_args.args[0], arrived_pose)
        np.testing.assert_allclose(self.robot.target, arrived_pose)
        self.control.stopJ.assert_not_called()

    def test_stop_during_home_uses_stopj_and_never_servol(self):
        self.request_home()

        def progress():
            self.robot.stop_event.set()
            return 0

        self.control.getAsyncOperationProgress.side_effect = progress
        self.run_worker()
        self.control.stopJ.assert_called_once_with(0.5)
        self.control.servoL.assert_not_called()
        self.control.stopScript.assert_called_once()
        self.receiver.disconnect.assert_called_once()

    def test_home_completion_requires_actual_joint_arrival(self):
        self.request_home()
        self.control.getAsyncOperationProgress.return_value = -1
        with self.assertRaisesRegex(RuntimeError, "without reaching"):
            self.run_worker()
        self.control.stopJ.assert_called_once_with(0.5)
        self.control.servoL.assert_not_called()

    def test_rejected_home_stops_without_resuming_servo(self):
        self.request_home()
        self.control.moveJ.return_value = False
        with self.assertRaisesRegex(RuntimeError, "rejected Home"):
            self.run_worker()
        self.control.stopJ.assert_called_once_with(0.5)
        self.control.servoL.assert_not_called()

    def test_home_keeps_command_watchdog(self):
        self.request_home()
        self.control.getAsyncOperationProgress.return_value = 0

        def lose_heartbeat(delay):
            self.clock.now += self.cfg["command_timeout_s"] + 0.01

        with self.assertRaisesRegex(TimeoutError, "No control update"):
            self.run_worker(lose_heartbeat)
        self.control.stopJ.assert_called_once_with(0.5)

    def test_home_total_timeout_even_with_collect_heartbeat(self):
        self.cfg["home_timeout_s"] = 0.02
        self.request_home()
        self.control.getAsyncOperationProgress.return_value = 0

        def heartbeat(delay):
            self.clock.now += delay
            self.robot.keepalive()

        with self.assertRaisesRegex(TimeoutError, "Home movement timed out"):
            self.run_worker(heartbeat)
        self.control.stopJ.assert_called_once_with(0.5)

    def test_home_does_not_accept_keyboard_targets_or_policy_mode(self):
        old_target = self.robot.target.copy()
        self.request_home()
        self.robot.command([0.6, 0.1, 0.2, 0, 0, 0])
        np.testing.assert_array_equal(self.robot.target, old_target)
        self.robot.timed_actions = object()
        with self.assertRaisesRegex(RuntimeError, "keyboard collection"):
            self.robot.request_home()

    def test_collect_home_key_skips_recording_and_jogging_until_arrival(self):
        image = {"rgb": np.zeros((84, 84, 3), dtype=np.uint8)}
        sample = {
            "tcp_pose": np.array([0.4, 0, 0.2, 0, 0, 0]),
            "gripper_width_m": 0.04,
            "cameras": {key: image for key in runtime.RGB_KEYS},
        }
        rig = mock.MagicMock()
        rig.__enter__.return_value = rig
        rig.snapshot.side_effect = lambda: {**sample, "tcp_pose": sample["tcp_pose"].copy()}
        rig.home.return_value = True
        type(rig).homing = mock.PropertyMock(side_effect=[True, False])
        keys = mock.MagicMock()
        events = [["home"], ["c", "o"], [], ["c"], ["home"], ["return", "home"], ["escape"]]
        keys.poll.side_effect = [(e, np.zeros(3), np.zeros(3)) for e in events]
        writer = mock.MagicMock()
        with mock.patch.object(runtime, "Rig", return_value=rig), mock.patch.object(
            runtime, "Keyboard", return_value=keys
        ), mock.patch.object(runtime, "EpisodeWriter", return_value=writer) as make_writer, mock.patch.object(
            runtime.time, "sleep"
        ):
            runtime.collect(self.cfg, "unused")
        self.assertEqual(rig.home.call_count, 2)
        self.assertEqual(rig.keepalive.call_count, 2)
        # Only the two recording ticks jog/record. Home is ignored while recording.
        self.assertEqual(rig.command.call_count, 2)
        self.assertEqual(writer.append.call_count, 2)
        make_writer.assert_called_once()
        writer.close.assert_called_once_with(success=True)
        self.assertTrue(all(call.args[1] == 0.04 for call in rig.command.call_args_list))
        keys.close.assert_called_once()
        rig.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
