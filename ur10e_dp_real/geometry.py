"""One explicit transform contract shared by collection, training and execution."""

import numpy as np
from scipy.spatial.transform import Rotation


def rotation6d(matrix):
    # PyTorch3D/robomimic use the first two ROWS, not columns.
    return np.asarray(matrix)[..., :2, :].reshape(6)


def matrix6d(value):
    a, b = np.asarray(value, dtype=float).reshape(2, 3)
    if not np.isfinite([a, b]).all() or np.linalg.norm(a) < 1e-8:
        raise ValueError("Non-finite or degenerate rotation6d")
    a = a / np.linalg.norm(a)
    b = b - np.dot(a, b) * a
    if np.linalg.norm(b) < 1e-8:
        raise ValueError("Collinear rotation6d rows")
    b /= np.linalg.norm(b)
    return np.stack([a, b, np.cross(a, b)])


class PolicyFrame:
    """base_T_policy and tcp_T_policy_tool are rigid homogeneous transforms.

    policy_T_tool = inv(base_T_policy) @ base_T_tcp @ tcp_T_policy_tool.
    The tool offset can represent the grasp center rather than the active UR TCP.
    Gripper qpos is a documented virtual Panda-compatible observation, not a sensor.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.base = self._transform(cfg["base_T_policy"])
        self.tool = self._transform(cfg["tcp_T_policy_tool"])
        self.inverse_base = np.linalg.inv(self.base)
        self.inverse_tool = np.linalg.inv(self.tool)

    @staticmethod
    def _transform(value):
        t = np.asarray(value, dtype=float)
        if t.shape != (4, 4) or not np.isfinite(t).all():
            raise ValueError("Transform must be finite 4x4")
        r = t[:3, :3]
        if (
            not np.allclose(t[3], [0, 0, 0, 1])
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(r), 1)
        ):
            raise ValueError("Transform must be a rigid SE(3) transform")
        return t

    def pose_matrix(self, pose):
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (6,) or not np.isfinite(pose).all():
            raise ValueError("TCP pose must contain finite xyz and rotation vector")
        t = np.eye(4)
        t[:3, 3] = pose[:3]
        t[:3, :3] = Rotation.from_rotvec(pose[3:]).as_matrix()
        return t

    def observations(self, sample):
        t = self.inverse_base @ self.pose_matrix(sample["tcp_pose"]) @ self.tool
        fraction = np.clip(
            sample["gripper_width_m"] / self.cfg["gripper_max_width_m"], 0, 1
        )
        closed = np.asarray(self.cfg["virtual_qpos_closed"])
        opened = np.asarray(self.cfg["virtual_qpos_open"])
        return {
            "robot0_eef_pos": t[:3, 3].astype("float32"),
            "robot0_eef_quat": Rotation.from_matrix(t[:3, :3])
            .as_quat()
            .astype("float32"),
            "robot0_gripper_qpos": (closed + fraction * (opened - closed)).astype(
                "float32"
            ),
        }

    def encode_action(self, tcp_pose, width):
        t = self.inverse_base @ self.pose_matrix(tcp_pose) @ self.tool
        grip = 1 - 2 * np.clip(width / self.cfg["gripper_max_width_m"], 0, 1)
        return np.r_[t[:3, 3], rotation6d(t[:3, :3]), grip].astype("float32")

    def decode_action(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (10,) or not np.isfinite(action).all():
            raise ValueError("Expected finite absolute10 action")
        t = np.eye(4)
        t[:3, :3], t[:3, 3] = matrix6d(action[3:9]), action[:3]
        t = self.base @ t @ self.inverse_tool
        pose = np.r_[t[:3, 3], Rotation.from_matrix(t[:3, :3]).as_rotvec()]
        width = (1 - np.clip(action[9], -1, 1)) * 0.5 * self.cfg["gripper_max_width_m"]
        return pose, float(width)


def bounded_pose(current, target, max_distance, max_angle):
    current, target = np.asarray(current), np.asarray(target)
    delta = target[:3] - current[:3]
    delta *= min(1.0, max_distance / max(np.linalg.norm(delta), 1e-12))
    r0, r1 = Rotation.from_rotvec(current[3:]), Rotation.from_rotvec(target[3:])
    dr = (r1 * r0.inv()).as_rotvec()
    dr *= min(1.0, max_angle / max(np.linalg.norm(dr), 1e-12))
    return np.r_[current[:3] + delta, (Rotation.from_rotvec(dr) * r0).as_rotvec()]
