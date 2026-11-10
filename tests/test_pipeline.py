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
from ur10e_dp_real.geometry import PolicyFrame, bounded_pose, matrix6d, rotation6d
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

    def test_rollout_holds_last_gripper_command_during_inference(self):
        import torch
        from ur10e_dp_real import runtime

        commands = []
        frame = PolicyFrame(self.cfg["policy_frame"])
        action = frame.encode_action([0.4, 0, 0.2, 0, 0, 0], 0)

        class Model:
            n_obs_steps = 2
            num_inference_steps = 2

            def predict_action(self, obs):
                return {"action": torch.from_numpy(action)[None, None]}

        class ImmediatePool:
            def __init__(self, *a, **kw):
                pass

            def submit(self, fn, *a):
                future = concurrent.futures.Future()
                future.set_result(fn(*a))
                return future

            def shutdown(self, **kw):
                pass

        class Rig:
            def __init__(self, *a, **kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def snapshot(self):
                image = {"rgb": np.zeros((84, 84, 3), dtype=np.uint8)}
                return {
                    "tcp_pose": [0.4, 0, 0.2, 0, 0, 0],
                    "gripper_width_m": 0.04,
                    "cameras": {
                        "agentview_image": image,
                        "robot0_eye_in_hand_image": image,
                    },
                }

            def command(self, pose, width):
                commands.append(width)

        class Keys:
            def __init__(self):
                self.calls = 0

            def poll(self):
                self.calls += 1
                return (["escape"] if self.calls == 4 else []), None, None

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
            runtime.concurrent.futures, "ThreadPoolExecutor", ImmediatePool
        ), mock.patch(
            "ur10e_dp_real.model.load_policy", return_value=(Model(), {}, {})
        ), mock.patch(
            "ur10e_dp_real.model.sha256", return_value="fixture"
        ), mock.patch.object(
            runtime.time, "sleep"
        ):
            runtime.rollout(self.cfg, "fixture", "fixture", device="cpu", duration=5)
        np.testing.assert_allclose(commands, [0.04, 0, 0])


if __name__ == "__main__":
    unittest.main()
