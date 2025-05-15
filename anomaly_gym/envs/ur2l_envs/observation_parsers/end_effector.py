"""End effector observation parser."""

import gymnasium as gym
import numpy as np
from typing_extensions import override

from anomaly_gym.envs.ur2l_envs.observation_parsers import ObservationParser


class EndEffectorObservationParser(ObservationParser):
    """Observation parser for the end effector and gripper information."""

    def __init__(
        self,
        use_orientation: bool = True,
        use_velocity: bool = True,
        use_rpy: bool = False,
        use_quat: bool = False,
        **observation_parser_config,
    ):
        """Initialize base action transformer."""
        super().__init__(**observation_parser_config)
        self._use_velocity = use_velocity
        self._use_rpy = use_rpy
        self._use_quat = use_quat

    @override
    def get_observation(self) -> np.ndarray:
        """Get the observation of the end effector state."""
        ee_pos = self._ur_interface.end_effector_position
        ee_vel = self._ur_interface.end_effector_velocity
        ee_quat = self._ur_interface.end_effector_quaternion
        ee_rpy = self._ur_interface.end_effector_rpy

        if self._normalize_observations:
            raise NotImplementedError

        self._latest_observation = np.concatenate(
            [
                ee_pos,
                ee_vel if self._use_velocity else [],
                ee_quat if self._use_quat else [],
                ee_rpy if self._use_rpy else [],
            ],
            dtype=np.float32,
        )
        return self._latest_observation

    @override
    def get_observation_space(self) -> gym.spaces.Box:
        """Get the observation space."""
        ee_size = 3 + (3 if self._use_velocity else 0) + (3 if self._use_rpy else 0) + (4 if self._use_quat else 0)
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(ee_size,), dtype=np.float32)
