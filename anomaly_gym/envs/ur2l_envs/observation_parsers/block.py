"""Block observation parser."""

import gymnasium as gym
import numpy as np
from typing_extensions import override

from anomaly_gym.envs.ur2l_envs.observation_parsers import ObservationParser


class BlockObservationParser(ObservationParser):
    """ObservationParser for the block to pick and place."""

    def __init__(
        self,
        use_position: bool = True,
        use_quaternion: bool = True,
        **observation_parser_config,
    ):
        """Initialize block observation parser.

        Args:
            use_position: Whether to use the position of the block.
            use_quaternion: Whether to use the quaternion of the block.
            **observation_parser_config: (dict) extra arguements.
        """
        super().__init__(**observation_parser_config)
        self._use_position = use_position
        self._use_quaternion = use_quaternion
        self._observation_length = use_position * 3 + use_quaternion * 4

    @override
    def get_observation(self):
        if self._normalize_observations:
            raise NotImplementedError
        else:
            self._latest_observation = np.concatenate(
                [
                    self._ur_interface.block_position if self._use_position else [],
                    self._ur_interface.block_quaternion if self._use_quaternion else [],
                ],
                dtype=np.float32,
            )
        return self._latest_observation

    @override
    def get_observation_space(self):
        # TODO: dont use -inf to inf for observation space size
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._observation_length,), dtype=np.float32)
