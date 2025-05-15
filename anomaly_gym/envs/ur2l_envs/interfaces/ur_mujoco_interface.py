"""Class to interface with a UR robot in Mujoco."""

import mujoco
import numpy as np
from transforms3d.euler import mat2euler
from transforms3d.quaternions import mat2quat
from typing_extensions import override

from anomaly_gym.common.mujoco_utils import (
    MujocoModelNames,
    get_site_xmat,
    get_site_xpos,
    get_site_xvelp,
    linear_map,
    set_mocap_pos,
    set_mocap_quat,
)
from anomaly_gym.envs.ur2l_envs.interfaces.ur_interface import URInterface
from anomaly_gym.envs.ur2l_envs.sim import MujocoSim


class URMujocoInterface(URInterface):
    """Abstract class that defines interactions with an UR robot."""

    def __init__(self, sim: MujocoSim, **kwargs):
        """Initialize the URMujocoInterface.

        Args:
            sim (MujocoSim): Simulation object.
            kwargs: (dict) extra arguents.
        """
        super().__init__(**kwargs)
        self.sim = sim
        self.robot_name = sim.robot_name
        self.gripper_name = sim.gripper_name

        self._model = sim.model
        self._data = sim.data

        self._joints = [
            "ur3/shoulder_pan_joint",
            "ur3/shoulder_lift_joint",
            "ur3/elbow_joint",
            "ur3/wrist_1_joint",
            "ur3/wrist_2_joint",
            "ur3/wrist_3_joint",
        ]
        pos_sensors = ["ur3/" + joint + ":pos" for joint in self._joints]
        self._pos_sensors_ids = [
            mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name) for sensor_name in pos_sensors
        ]
        vel_sensors = ["ur3/" + joint + ":vel" for joint in self._joints]
        self._vel_sensors_ids = [
            mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name) for sensor_name in vel_sensors
        ]

        # # print all geom names
        # for i in range(self._model.ngeom):
        #     print(f"Geom {i}: {mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, i)}")

        self._gripper_padding = 0.01493
        self._left_pad_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_GEOM, f"{self.robot_name}/{self.gripper_name}/left_pad1"
        )
        self._right_pad_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_GEOM, f"{self.robot_name}/{self.gripper_name}/right_pad1"
        )
        self._block_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "block")
        self.allowed_contacts_ids: list[int] = [self._left_pad_id, self._right_pad_id, self._block_id]

        self._joint_name_to_qpos_id = {
            mujoco.mj_id2name(self.sim.model, mujoco.mjtObj.mjOBJ_JOINT, j_id): self.sim.model.jnt_qposadr[j_id]
            for j_id in range(self.sim.model.njnt)
        }
        self._joint_ids = [
            mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) for joint_name in self._joints
        ]

        self._model_names = MujocoModelNames(self.sim.model)

    @property
    def n_robot_joints(self) -> int:
        """Number of joints of the robot."""
        return len(self._joint_ids)

    # === Getters ===
    @override
    @property
    def joint_position(self) -> np.ndarray:
        """Get the current joint position of the robot.

        Returns:
            np.array: Current joint position of the robot (6,).
        """
        # return self.sim.data.sensordata[self._pos_sensors_ids]
        return self.sim.data.qpos[self._joint_ids]

    @override
    @property
    def joint_velocity(self) -> np.ndarray:
        """Get the current joint velocity of the robot.

        Returns:
            np.array: Current joint velocity of the robot (6,).
        """
        return self.sim.data.qvel[self._joint_ids]

    @override
    @property
    def gripper_distance(self) -> float:
        """Get the opening of the gripper.

        Returns:
            np.array: Current gripper distance (1,).
        """
        left_pos = self.sim.data.body(f"{self.robot_name}/{self.gripper_name}/left_silicone_pad").xpos
        right_pos = self.sim.data.body(f"{self.robot_name}/{self.gripper_name}/right_silicone_pad").xpos
        gripper_distance = np.linalg.norm(left_pos - right_pos) - self._gripper_padding
        return gripper_distance

    @override
    @property
    def target_position(self) -> np.ndarray:
        """Get the target position of the robot."""
        return get_site_xpos(self.sim.model, self.sim.data, "target").copy()

    @override
    @property
    def target_quaternion(self) -> np.ndarray:
        """Get the target quaternion of the robot."""
        return np.array(mat2quat(get_site_xmat(self.sim.model, self.sim.data, "target").copy()))

    @override
    @property
    def goal_position(self) -> np.ndarray:
        """Get the goal position of the robot."""
        return get_site_xpos(self.sim.model, self.sim.data, "goal").copy()

    @override
    @property
    def goal_quaternion(self) -> np.ndarray:
        """Get the goal quaternion of the robot."""
        return np.array(mat2quat(get_site_xmat(self.sim.model, self.sim.data, "goal").copy()))

    @override
    @property
    def block_position(self) -> np.ndarray:
        """Get the block position of the robot."""
        return get_site_xpos(self.sim.model, self.sim.data, "block").copy()

    @override
    @property
    def block_quaternion(self) -> np.ndarray:
        """Get the block quaternion of pick."""
        return np.array(mat2quat(get_site_xmat(self.sim.model, self.sim.data, "block").copy()))

    @override
    @property
    def end_effector_position(self) -> np.ndarray:
        return get_site_xpos(self.sim.model, self.sim.data, f"{self.robot_name}/end_effector").copy()

    @override
    @property
    def end_effector_velocity(self) -> np.ndarray:
        return get_site_xvelp(self.sim.model, self.sim.data, f"{self.robot_name}/end_effector").copy()

    @override
    @property
    def end_effector_quaternion(self) -> np.ndarray:
        return np.array(
            mat2quat(get_site_xmat(self.sim.model, self.sim.data, f"{self.robot_name}/end_effector").copy())
        )

    @property
    def end_effector_rpy(self) -> np.ndarray:
        """Get the end-effector rotation in roll, pitch, yaw."""
        xmat = get_site_xmat(self.sim.model, self.sim.data, f"{self.robot_name}/end_effector").copy()
        return np.array(mat2euler(xmat))

    @override
    @property
    def body_collision(self) -> np.ndarray:
        for contact in self.sim.data.contact:
            if contact.geom[0] in self.allowed_contacts_ids and contact.geom[1] in self.allowed_contacts_ids:
                continue
            return True
        return False

    def set_q_ctrl(self, q_ctrl):
        q_ctrl = np.clip(q_ctrl, self.limits.q_ctrl_min, self.limits.q_ctrl_max)
        self.sim.data.ctrl += q_ctrl

    # === Setters ===
    @override
    def set_gripper_distance(self, distance: float):
        """Set the target distance of the gripper."""
        distance = np.clip(distance, self.limits.min_gripper_width, self.limits.max_gripper_width)
        self.sim.data.ctrl[-1] = linear_map(
            distance, self.limits.min_gripper_width, self.limits.max_gripper_width, 255, 0
        )

    @override
    def set_target(self, target_pos: np.ndarray, target_quat: np.ndarray):
        """Set the target target_pos and target_quat of the robot."""
        target_pos = np.clip(target_pos, self.limits.target_min, self.limits.target_max)
        set_mocap_pos(self.sim.model, self.sim.data, "target", target_pos)
        set_mocap_quat(self.sim.model, self.sim.data, "target", target_quat)
        self.sim.advance_simulation(n_steps=1)

    @override
    def set_block(self, position: np.ndarray, quaternion: np.ndarray):
        """Set the block position and quaternion for pick and place."""
        self.sim.data.joint("block_freejoint").qpos = np.concatenate([position, quaternion])
        self.sim.data.joint("block_freejoint").qvel = 6 * [0.0]

    @override
    def set_goal(self, position: np.ndarray, quaternion: np.ndarray):
        """Set the goal position and quaternion of the robot."""
        set_mocap_pos(self.sim.model, self.sim.data, "goal", position)
        set_mocap_quat(self.sim.model, self.sim.data, "goal", quaternion)
        self.sim.advance_simulation(n_steps=1)

    def get_touch_sensor_data(self):
        sensor_data = {k: self._data.sensor(k).data for k in self._model_names.sensor_names if "touch" in k}
        return sensor_data
