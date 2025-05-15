"""Joints observation parser."""

import gymnasium as gym
import numpy as np
from typing_extensions import override

from anomaly_gym.envs.ur2l_envs.observation_parsers import ObservationParser


class JointsObservationParser(ObservationParser):
    """Observation parser for the joints information."""

    def __init__(self, **observation_parser_config):
        """Initialize base action transformer."""
        super().__init__(**observation_parser_config)
        self._joint_space_size = 6

    @override
    def get_observation(self) -> np.array:
        """Get the observation of the end effector state."""
        joint_pos = self._ur_interface.joint_position
        joint_vel = self._ur_interface.joint_velocity
        joint_torque = np.zeros_like(joint_pos)

        if self._normalize_observations:
            raise NotImplementedError

        self._latest_observation = np.concatenate(
            [joint_pos, joint_vel, joint_torque],
            dtype=np.float32,
        )
        return self._latest_observation

    @override
    def get_observation_space(self) -> gym.spaces.Box:
        """Get the observation space."""
        obs_size = 3 * self._joint_space_size
        # TODO: dont use -inf to inf for observation space size
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
