"""Collision observation parser."""

import gymnasium as gym
from typing_extensions import override

from anomaly_gym.envs.ur2l_envs.observation_parsers import ObservationParser


class CollisionObservationParser(ObservationParser):
    """ObservationParser used to check collisions within the robot."""

    def __init__(self, **observation_parser_config):
        """Initialize the CollisionObservationParser."""
        super().__init__(**observation_parser_config)

    @override
    def is_ready(self):
        return True

    @override
    def get_observation(self):
        self._latest_observation = int(self._ur_interface.body_collision)
        return self._latest_observation

    @override
    def get_observation_space(self):
        return gym.spaces.Discrete(n=2)
