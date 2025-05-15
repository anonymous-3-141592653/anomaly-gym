"""Target observation parser."""

import gymnasium as gym
import numpy as np
from typing_extensions import override

from anomaly_gym.common.func_utils import min_max_norm
from anomaly_gym.envs.ur2l_envs.observation_parsers import ObservationParser


class TargetObservationParser(ObservationParser):
    """Observation parser for the end effector and gripper information."""

    def __init__(
        self,
        min_target_pos: float = -1.0,
        max_target_pos: float = 1.0,
        **observation_parser_config,
    ):
        """Initialize base action transformer."""
        super().__init__(**observation_parser_config)

    @override
    def get_observation(self) -> np.ndarray:
        """Get the observation of the end effector state."""
        target_pos = self._ur_interface.target_position
        target_quat = self._ur_interface.target_quaternion

        if self._normalize_observations:
            target_pos = min_max_norm(target_pos, -1.0, 1.0)
            target_quat = min_max_norm(target_quat, -1.0, 1.0)

        self._latest_observation = np.concatenate(
            [
                target_pos,
                target_quat,
            ],
            dtype=np.float32,
        )
        return self._latest_observation

    @override
    def get_observation_space(self) -> gym.spaces.Box:
        """Get the observation space."""
        target_size = 3 + 4
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(target_size,), dtype=np.float32)
