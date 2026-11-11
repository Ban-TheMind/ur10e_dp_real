import copy
import json
import tempfile
import unittest
from unittest import mock
import types
import concurrent.futures
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from ur10e_dp_real.geometry import PolicyFrame, TimedActions, bounded_pose, matrix6d, rotation6d
from ur10e_dp_real.recording import EpisodeWriter, check_episode
from ur10e_dp_real.data import Episodes, discover, split_episodes

ROOT = Path(__file__).resolve().parents[1]


def make_episode(root, cfg, number=0, samples=5):
    writer = EpisodeWriter(root, cfg, metadata={"synthetic_fixture": True})
    first = None
    for i in range(samples):
        t = 10 + i / 20
        pose = np.array([0.4 + number * 0.01, 0, 0.2 + i * 0.001, 0, 0, 0])
        camera = {
            "rgb": np.full((96, 128, 3), i * 30, dtype=np.uint8),
            "source_timestamp_ms": t * 1000,
            "timestamp_domain": "synthetic",
            "receive_monotonic": t - 0.005,
            "frame_number": i,
        }
        sample = {
            "tcp_pose": pose,
            "joint_position": np.zeros(6),
            "joint_velocity": np.zeros(6),
            "gripper_width_m": 0.04,
            "gripper": {"width_m": 0.04, "receive_monotonic": t - 0.01},
            "receive_monotonic": t - 0.001,
            "snapshot_monotonic": t,
            "controller_timestamp": t,
            "cameras": {
                "agentview_image": copy.deepcopy(camera),
                "robot0_eye_in_hand_image": copy.deepcopy(camera),
            },
        }
        if first is None:
            first = sample
        writer.append(sample, pose, 0.04)
    path = writer.close(success=True)
    assert "rgb" in first["cameras"]["agentview_image"]
    return path


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT / "configs/site.example.json").read_text())

    def test_rotation_roundtrip(self):
        rng = np.random.RandomState(1)
        for _ in range(40):
            r = Rotation.from_rotvec(rng.randn(3)).as_matrix()
            np.testing.assert_allclose(matrix6d(rotation6d(r)), r, atol=1e-12)

    def test_checkpoint_chunk_assembly(self):
        import hashlib
        from tools.fetch_weights import assemble

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            chunks = [b"first", b"second"]
            entry = {
                "file": "test.ckpt",
                "bytes": 11,
                "sha256": hashlib.sha256(b"".join(chunks)).hexdigest(),
                "parts": [],
            }
            for i, block in enumerate(chunks):
                name = "test.part" + str(i)
                (root / name).write_bytes(block)
                entry["parts"].append(
                    {
                        "file": name,
                        "bytes": len(block),
                        "sha256": hashlib.sha256(block).hexdigest(),
                    }
                )
            result = assemble(entry, root)
            self.assertEqual(result.read_bytes(), b"firstsecond")
            result.unlink()
            (root / "test.part1").write_bytes(b"broken")
            with self.assertRaises(ValueError):
                assemble(entry, root)
            self.assertFalse(result.exists())

    def test_bad_rotation(self):
        for a in [np.zeros(6), [1, 0, 0, 1, 0, 0], [np.nan] * 6]:
            with self.assertRaises(ValueError):
                matrix6d(a)

    def test_frame_tool_inverse(self):
        cfg = self.cfg["policy_frame"]
        cfg["base_T_policy"] = [
            [0, -1, 0, 0.1],
            [1, 0, 0, -0.2],
            [0, 0, 1, 0.05],
            [0, 0, 0, 1],
        ]
        cfg["tcp_T_policy_tool"][2][3] = 0.13
        frame = PolicyFrame(cfg)
        pose = np.array([0.4, 0.2, 0.3, 0.1, -0.2, 0.4])
        action = frame.encode_action(pose, 0.025)
        restored, width = frame.decode_action(action)
        np.testing.assert_allclose(restored, pose, atol=1e-6)
        self.assertAlmostEqual(width, 0.025, places=7)

    def test_virtual_gripper(self):
        frame = PolicyFrame(self.cfg["policy_frame"])
        obs = frame.observations({"tcp_pose": [0] * 6, "gripper_width_m": 0.085})
        np.testing.assert_allclose(obs["robot0_gripper_qpos"], [0.04, -0.04])
        self.assertEqual(frame.encode_action([0] * 6, 0)[-1], 1)

    def test_pose_limit(self):
        pose = bounded_pose(np.zeros(6), np.array([1, 1, 0, 0, 0, 2]), 0.01, 0.02)
        self.assertAlmostEqual(np.linalg.norm(pose[:3]), 0.01)
        self.assertAlmostEqual(np.linalg.norm(pose[3:]), 0.02)

    def test_record_split_windows(self):
        with tempfile.TemporaryDirectory() as root:
            for i in range(3):
                make_episode(root, self.cfg, i)
            paths = discover(root)
            train, val = split_episodes(paths, 0.2, 42)
            self.assertFalse(set(train) & set(val))
            self.assertEqual((len(train), len(val)), (2, 1))
            self.assertEqual(check_episode(paths[0])["samples"], 5)
            data = Episodes(train, 10, 2)
            batch = data[0]
            self.assertEqual(tuple(batch["action"].shape), (10, 10))
            self.assertEqual(
                tuple(batch["obs"]["agentview_image"].shape), (2, 3, 84, 84)
            )
            np.testing.assert_allclose(batch["action"][0], batch["action"][1])

    def test_failed_not_training_data(self):
        with tempfile.TemporaryDirectory() as root:
            path = make_episode(root, self.cfg)
            meta = json.loads((path / "episode.json").read_text())
            meta["success"] = False
            (path / "episode.json").write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                discover(root)

    def test_normalizer_uses_only_training_episodes(self):
        with tempfile.TemporaryDirectory() as root:
            training = make_episode(root, self.cfg, number=0)
            heldout = make_episode(root, self.cfg, number=100)
            trainset = Episodes([training], 10, 2)
            normalizer = trainset.get_normalizer()
            self.assertLess(float(normalizer["robot0_eef_pos"].get_input_stats()["max"][0]), 0.5)
            sample = Episodes([heldout], 10, 2)[0]
            action = sample["action"]
            import torch
            torch.testing.assert_close(normalizer["action"].unnormalize(normalizer["action"](action)), action)
            torch.testing.assert_close(normalizer["action"](action)[:, 3:], action[:, 3:])

    def test_finetune_target_normalizer_and_resume(self):
        from ur10e_dp_real.training import configure_normalizer

        model, trainset = mock.Mock(), mock.Mock()
        settings = {"normalizer": "real_train_only", "device": "cpu"}
        mode = configure_normalizer(
            model, trainset, settings, from_scratch=False, resume=False
        )
        self.assertEqual(mode, "real_train_only")
        model.set_normalizer.assert_called_once_with(trainset.get_normalizer.return_value)
        model.to.assert_called_once_with("cpu")
        model.reset_mock()
        trainset.reset_mock()
        configure_normalizer(model, trainset, settings, from_scratch=False, resume=True)
        trainset.get_normalizer.assert_not_called()
        model.set_normalizer.assert_not_called()

    def test_legacy_normalizer_defaults_and_invalid_mode(self):
        from ur10e_dp_real.training import configure_normalizer

        model, trainset = mock.Mock(), mock.Mock()
        self.assertEqual(configure_normalizer(
            model, trainset, {"device": "cpu"}, from_scratch=False, resume=False
        ), "preserved_from_source")
        model.set_normalizer.assert_not_called()
        self.assertEqual(configure_normalizer(
            model, trainset, {"device": "cpu"}, from_scratch=True, resume=False
        ), "real_train_only")
        for mode, scratch in (("invalid", False), ("preserved_from_source", True)):
            with self.assertRaises(ValueError):
                configure_normalizer(model, trainset, {"normalizer": mode},
                                     from_scratch=scratch, resume=False)

    def test_mixed_frames_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            a = make_episode(root, self.cfg)
            changed = copy.deepcopy(self.cfg)
            changed["policy_frame"]["base_T_policy"][0][3] = 0.1
            b = make_episode(root, changed)
            with self.assertRaises(ValueError):
                Episodes([a, b], 10, 2)

    def test_control_initialization_failure_disconnects_receive(self):
        from ur10e_dp_real.devices import Robot

        receiver = mock.MagicMock()
        modules = {
            "rtde_receive": types.SimpleNamespace(
                RTDEReceiveInterface=lambda _: receiver
            ),
            "rtde_control": types.SimpleNamespace(
                RTDEControlInterface=mock.Mock(side_effect=OSError("unavailable"))
            ),
        }
        with mock.patch.dict("sys.modules", modules), self.assertRaises(OSError):
            Robot(self.cfg)
        receiver.disconnect.assert_called_once()

    def test_timed_actions_replace_arm_and_gripper_on_one_timeline(self):
        plan = TimedActions()
        start = np.zeros(6)
        forward = np.array([0.06, 0, 0, 0, 0, 0])
        backward = np.array([0.02, 0, 0, 0, 0, 0])
        self.assertTrue(plan.submit(
            0, [(0.2, forward, 0.0), (0.3, forward, 0.085)], start, 1, 1
        ))
        with plan.changed:
            before, width, _ = plan.sample(0.1)
        self.assertIsNone(width)
        self.assertTrue(plan.submit(0.1, [(0.25, backward, 0.04)], start, 1, 1))
        self.assertFalse(plan.submit(0.15, [], start, 1, 1))
        with plan.changed:
            np.testing.assert_allclose(plan.sample(0.1)[0], before)
            self.assertIsNone(plan.sample(0.2)[1])
            np.testing.assert_allclose(plan.sample(0.3)[0], backward)
            self.assertEqual(plan.sample(0.3)[1], 0.04)
            self.assertEqual(plan.submitted_at, 0.1)

    def test_timed_actions_retime_both_devices_without_collapsing_steps(self):
        plan = TimedActions()
        start = np.zeros(6)
        target = np.array([0.1, 0, 0, 0, 0, 0])
        self.assertTrue(plan.submit(
            0, [(0.05, target, 0.0), (0.10, target, 0.085)], start, 0.5, 1
        ))
        with plan.changed:
            self.assertIsNone(plan.sample(0.19)[1])
            self.assertEqual(plan.sample(0.21)[1], 0.0)
            self.assertEqual(plan.sample(0.26)[1], 0.085)

    def test_timed_gripper_sends_due_change_before_next_status_poll(self):
        from ur10e_dp_real import devices

        plan = TimedActions()
        pose = np.zeros(6)
        plan.submit(0, [(0.05, pose, 0.0)], pose, 1, 1)
        clock = types.SimpleNamespace(now=0.0)
        clock.monotonic = lambda: clock.now
        transactions = []

        with mock.patch.object(devices.Gripper, "launch"):
            gripper = devices.Gripper({"poll_hz": 10}, timed_actions=plan)

        def advance(delay):
            clock.now += delay

        def transact(operation, width=0, **kwargs):
            transactions.append((clock.now, operation, width))
            if operation == "command":
                gripper.stop_event.set()
            return {}

        with mock.patch.object(devices, "time", clock), mock.patch.object(
            plan.changed, "wait", side_effect=advance
        ), mock.patch.object(gripper, "transact", side_effect=transact):
            gripper.run()
        self.assertEqual(transactions, [(0, "status", 0), (0.05, "command", 0)])

    def test_schedule_filters_expired_points_without_refreshing_watchdog(self):
        from ur10e_dp_real import devices

        rig = devices.Rig.__new__(devices.Rig)
        rig.cfg = self.cfg
        rig.timed_actions = TimedActions()
        pose = np.array([0.4, 0, 0.2, 0, 0, 0])
        rig.robot = types.SimpleNamespace(target=pose, validate_target=np.asarray)
        with mock.patch.object(devices.time, "monotonic", return_value=1.0):
            self.assertTrue(rig.schedule([
                (0.9, pose, 0.085), (1.01, pose, 0.085), (1.05, pose, 0.0)
            ], lead_time=0.01))
        self.assertEqual([p[0] for p in rig.timed_actions.points], [1.0, 1.05])
        version = rig.timed_actions.version
        with mock.patch.object(devices.time, "monotonic", return_value=1.2):
            self.assertFalse(rig.schedule([(1.1, pose, 0.085)], lead_time=0.01))
        self.assertEqual(rig.timed_actions.version, version)
        self.assertEqual(rig.timed_actions.submitted_at, 1.0)
        target, width = rig.scheduled_target(1.3)
        np.testing.assert_array_equal(target, pose)
        self.assertEqual(width, 0.0)

    def test_gripper_rechecks_cancelled_or_no_longer_due_target_before_launch(self):
        from ur10e_dp_real import devices

        for cancelled_width in (0.0, 0.085):
            with self.subTest(cancelled_width=cancelled_width):
                plan, pose = TimedActions(), np.zeros(6)
                plan.submit(0, [(0.05, pose, cancelled_width),
                                (0.15, pose, 0.04)], pose, 1, 1)
                version = plan.version
                with mock.patch.object(devices.Gripper, "launch"):
                    gripper = devices.Gripper(self.cfg["gripper"], timed_actions=plan)
                with mock.patch.object(devices.subprocess, "Popen") as popen:
                    # Width was selected at 0.05, but the next width became due
                    # before dispatch, even without any plan replacement.
                    with mock.patch.object(devices.time, "monotonic", return_value=0.16):
                        self.assertIsNone(gripper.transact(
                            "command", cancelled_width, plan_version=version))
                    # Replace future points after selection but before launch.
                    plan.submit(0.06, [(0.10, pose, 0.02)], pose, 1, 1)
                    with mock.patch.object(devices.time, "monotonic", return_value=0.11):
                        self.assertIsNone(gripper.transact(
                            "command", cancelled_width, plan_version=version))
                    popen.assert_not_called()
                    process = popen.return_value
                    process.returncode = 0
                    process.communicate.return_value = ('{"fault": 0, "position": 128}', '')
                    with mock.patch.object(devices.time, "monotonic", return_value=0.11):
                        self.assertIsNotNone(gripper.transact(
                            "command", 0.02, plan_version=plan.version))
                    popen.assert_called_once()

    def test_gripper_replacement_wakes_wait_and_drops_cancelled_future_close(self):
        from ur10e_dp_real import devices

        plan, pose = TimedActions(), np.zeros(6)
        plan.submit(0, [(0.05, pose, 0.0), (0.09, pose, 0.085)], pose, 1, 1)
        clock = types.SimpleNamespace(now=0.0)
        clock.monotonic = lambda: clock.now
        transactions, waits = [], []
        with mock.patch.object(devices.Gripper, "launch"):
            gripper = devices.Gripper({"poll_hz": 10}, timed_actions=plan)

        def wait(delay):
            waits.append(delay)
            if len(waits) == 1:
                # New block arrives before the old close becomes due.
                clock.now = 0.02
                with mock.patch.object(plan.changed, "notify_all") as notify:
                    plan.submit(clock.now, [(0.07, pose, 0.04)], pose, 1, 1)
                    notify.assert_called_once()
            else:
                clock.now += delay

        def transact(operation, width=0, **kwargs):
            transactions.append((clock.now, operation, width))
            if operation == "command":
                gripper.stop_event.set()
            return {}

        with mock.patch.object(devices, "time", clock), mock.patch.object(
            plan.changed, "wait", side_effect=wait
        ), mock.patch.object(gripper, "transact", side_effect=transact):
            gripper.run()
        self.assertEqual(transactions, [(0, "status", 0), (0.07, "command", 0.04)])

    def test_timed_robot_watchdog_uses_existing_stop_path(self):
        from ur10e_dp_real import devices

        receiver, control = mock.MagicMock(), mock.MagicMock()
        receiver.getActualTCPPose.return_value = [0.4, 0, 0.2, 0, 0, 0]
        receiver.getSafetyStatusBits.return_value = 0
        modules = {
            "rtde_receive": types.SimpleNamespace(RTDEReceiveInterface=lambda _: receiver),
            "rtde_control": types.SimpleNamespace(RTDEControlInterface=lambda _: control),
        }
        plan = TimedActions()
        pose = np.asarray(receiver.getActualTCPPose.return_value)
        plan.submit(0, [(0.05, pose, 0.0)], pose, 1, 1)
        with mock.patch.dict("sys.modules", modules), mock.patch.object(
            devices.Robot, "launch"
        ), mock.patch.object(devices.time, "monotonic", return_value=0):
            robot = devices.Robot(self.cfg, timed_actions=plan)
        with mock.patch.object(devices.time, "monotonic", return_value=2.01):
            with self.assertRaises(TimeoutError):
                robot.run()
        control.servoL.assert_not_called()
        control.servoStop.assert_called_once()
        control.stopScript.assert_called_once()
        control.disconnect.assert_called_once()
        receiver.disconnect.assert_called_once()

    def test_robot_reads_new_trajectory_at_current_time_after_state_read(self):
        from ur10e_dp_real import devices

        pose = np.array([0.4, 0, 0.2, 0, 0, 0])
        target = pose.copy()
        target[0] = 0.42
        plan = TimedActions()
        plan.submit(0, [(0.2, target, 0.0)], pose, 1, 1)
        receiver, control = mock.MagicMock(), mock.MagicMock()
        receiver.getActualTCPPose.return_value = pose.tolist()
        clock = types.SimpleNamespace(now=0.1)
        clock.monotonic = lambda: clock.now
        modules = {
            "rtde_receive": types.SimpleNamespace(RTDEReceiveInterface=lambda _: receiver),
            "rtde_control": types.SimpleNamespace(RTDEControlInterface=lambda _: control),
        }
        with mock.patch.dict("sys.modules", modules), mock.patch.object(
            devices.Robot, "launch"
        ), mock.patch.object(devices, "time", clock):
            robot = devices.Robot(self.cfg, timed_actions=plan)

        def read_safety():
            clock.now = 0.11
            plan.submit(clock.now, [(0.3, target, 0.04)], pose, 1, 1)
            return 0

        receiver.getSafetyStatusBits.side_effect = read_safety
        control.servoL.side_effect = lambda *args: robot.stop_event.set() or True
        with mock.patch.object(devices, "time", clock), mock.patch.object(
            robot.stop_event, "wait"
        ), mock.patch.object(devices, "bounded_pose", wraps=bounded_pose) as bounded:
            robot.run()
        # At splice time the old command is 0.411, although measured TCP is 0.4.
        self.assertAlmostEqual(bounded.call_args.args[1][0], 0.411)

    def test_rollout_holds_last_arm_and_gripper_targets_during_inference(self):
        import torch
        from ur10e_dp_real import devices, runtime

        submissions, inference_times = [], []
        clock = types.SimpleNamespace(now=0.0)
        clock.monotonic = lambda: clock.now

        def advance(seconds):
            clock.now += seconds

        clock.sleep = advance
        frame = PolicyFrame(self.cfg["policy_frame"])
        action = frame.encode_action([0.41, 0, 0.2, 0, 0, 0], 0)

        class Model:
            n_obs_steps = 2
            n_action_steps = 8
            num_inference_steps = 2

            def predict_action(self, obs):
                self_input_shape = tuple(obs["robot0_eef_pos"].shape)
                assert self_input_shape == (1, 2, 3)
                return {"action": torch.from_numpy(action)[None, None].repeat(1, 8, 1)}

        class FirstResultPool:
            def __init__(self, *a, **kw):
                self.calls = 0

            def submit(self, fn, *a):
                self.calls += 1
                inference_times.append(a[-1])
                future = concurrent.futures.Future()
                if self.calls == 1:
                    future.set_result(fn(*a))
                return future

            def shutdown(self, **kw):
                pass

        class Rig(devices.Rig):
            def __init__(self, cfg, timed):
                assert timed
                self.cfg = cfg
                self.timed_actions = TimedActions()
                self.robot = types.SimpleNamespace(
                    target=np.array([0.4, 0, 0.2, 0, 0, 0]), validate_target=np.asarray
                )

            def __exit__(self, *a):
                pass

            def snapshot(self):
                image = {"rgb": np.zeros((84, 84, 3), dtype=np.uint8)}
                return {
                    "tcp_pose": self.robot.target.copy(),
                    "gripper_width_m": 0.04,
                    "snapshot_monotonic": clock.now,
                    "cameras": {
                        "agentview_image": image,
                        "robot0_eye_in_hand_image": image,
                    },
                }

            def schedule(self, points, lead_time):
                submissions.extend(points)
                return super().schedule(points, lead_time)

            def command(self, pose, width):
                raise AssertionError("Rollout must not overwrite a scheduled target")

        class Keys:
            def poll(self):
                return [], None, None

            def draw(self, *a):
                pass

            def close(self):
                pass

        writer = mock.MagicMock()
        with mock.patch.object(runtime, "Rig", Rig), mock.patch.object(
            runtime, "Keyboard", Keys
        ), mock.patch.object(
            runtime, "EpisodeWriter", return_value=writer
        ), mock.patch.object(
            runtime.concurrent.futures, "ThreadPoolExecutor", FirstResultPool
        ), mock.patch(
            "ur10e_dp_real.model.load_policy", return_value=(Model(), {}, {})
        ), mock.patch(
            "ur10e_dp_real.model.sha256", return_value="fixture"
        ), mock.patch.object(runtime, "time", clock), mock.patch.object(
            devices, "time", clock
        ):
            runtime.rollout(self.cfg, "fixture", "fixture", device="cpu", duration=0.6)

        np.testing.assert_allclose([point[0] for point in submissions], np.arange(3, 9) / 20)
        _, target, width, _ = writer.append.call_args.args
        self.assertAlmostEqual(target[0], 0.41)
        self.assertEqual(width, 0.0)
        np.testing.assert_allclose(inference_times, [0.05, 0.25])
        self.assertAlmostEqual(writer.append.call_args_list[-1].args[0]["tcp_pose"][0], 0.4)


if __name__ == "__main__":
    unittest.main()
