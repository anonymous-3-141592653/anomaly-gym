"""Gripper observation parser."""

import gymnasium as gym
import numpy as np
from typing_extensions import override

from anomaly_gym.common.func_utils import min_max_norm
from anomaly_gym.envs.ur2l_envs.observation_parsers import ObservationParser


class GripperObservationParser(ObservationParser):
    """Observation parser for the end effector and gripper information."""

    def __init__(self, **observation_parser_config):
        """Initialize the gripper observation parser."""
        super().__init__(**observation_parser_config)

    @override
    def get_observation(self) -> np.ndarray:
        """Get the observation of the end effector state."""
        gripper_distance = self._ur_interface.gripper_distance

        gripper_distance = min_max_norm(
            gripper_distance, self._ur_interface.limits.min_gripper_width, self._ur_interface.limits.max_gripper_width
        )

        self._latest_observation = np.array([gripper_distance], dtype=np.float32)
        return self._latest_observation

    @override
    def get_observation_space(self) -> gym.spaces.Box:
        """Get the observation space."""
        gripper_size = 1
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(gripper_size,), dtype=np.float32)
