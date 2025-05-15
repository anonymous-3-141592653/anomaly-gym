"""Classes to transform the actions for different environment types."""

from abc import ABC, abstractmethod
from typing import Literal

import gymnasium as gym
import numpy as np
from numpy.core.multiarray import array as array
from typing_extensions import override

from anomaly_gym.envs.ur2l_envs.action_setters.gripper import get_gripper_action
from anomaly_gym.envs.ur2l_envs.interfaces import URInterface


class ActionSetter(ABC):
    """Takes the action of the agent and adapts it to the environment."""

    last_action: np.ndarray
    action_size: int

    def __init__(
        self,
        ur_interface: URInterface,
        gripper_action_type: str,
        gripper_action_cfg=None,
    ):
        """Initialize the action setter."""
        self._ur_interface = ur_interface
        self.gripper_action = get_gripper_action(gripper_action_type, gripper_action_cfg)

    @abstractmethod
    def set_action(self, action: np.ndarray):
        """Set an action based on the agent's'raw' action."""
        raise NotImplementedError

    def get_stats(self) -> list:
        """Get latest action information."""
        return np.array(self.last_action).tolist()

    def get_action_space(self) -> gym.spaces.Box:
        """Get the action space of the policy."""
        low = np.array([-1.0] * self.action_size).astype(np.float32)
        high = -low
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def get_gripper_target(self, gripper_action) -> float:
        """Get the target gripper width based on the action."""
        current_gripper_distance = self._ur_interface.gripper_distance
        return self.gripper_action(gripper_action, current_gripper_distance)


class JointActionSetter(ActionSetter):
    def __init__(self, scale_factor=0.05, **kwargs):
        super().__init__(**kwargs)
        self.scale_factor = scale_factor
        self.action_size = self._ur_interface.n_robot_joints + self.gripper_action.action_size

    def set_action(self, action: np.ndarray):
        q_action = action[: self._ur_interface.n_robot_joints]
        gripper_action = action[self._ur_interface.n_robot_joints :]
        q_ctrl = q_action * self.scale_factor
        gripper_target = self.get_gripper_target(gripper_action)
        self._ur_interface.set_q_ctrl(q_ctrl)
        self._ur_interface.set_gripper_distance(gripper_target)


class CartesianActionSetter(ActionSetter):
    """Class to set a cartesian actions in the environment."""

    def __init__(self, scale_factor=0.004, clip_delta=0.01, **kwargs):
        super().__init__(**kwargs)
        self.scale_factor = scale_factor
        self.clip_delta = clip_delta
        self.action_size = 3 + self.gripper_action.action_size

    @override
    def set_action(self, action: np.ndarray):
        self.last_action = action

        ee_action = action[:3]
        gripper_action = action[3:]

        # get new ee position
        pos_delta = np.clip(self.scale_factor * ee_action, -self.clip_delta, self.clip_delta)
        target_pos = self._ur_interface.target_position + pos_delta

        # Do not change the target quaternion, always the same.
        target_quat = self._ur_interface.target_quaternion

        gripper_target = self.get_gripper_target(gripper_action)

        self._ur_interface.set_target(target_pos=target_pos, target_quat=target_quat)
        self._ur_interface.set_gripper_distance(gripper_target)


def make_action_setter(action_type: Literal["joint", "cartesian"], **kwargs) -> ActionSetter:
    """Make an action setter object."""
    if action_type == "joint":
        return JointActionSetter(**kwargs)
    elif action_type == "cartesian":
        return CartesianActionSetter(**kwargs)
    else:
        raise NotImplementedError(f"Unknown action setter type: {action_type}.")
