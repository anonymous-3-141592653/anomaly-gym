"""Observation parser classes."""

import logging
from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np

from anomaly_gym.envs.ur2l_envs.interfaces import URInterface

logger = logging.getLogger("ur2l_gym")


class ObservationParser(ABC):
    """Base class for parsing some data in observations used for a learning agent."""

    def __init__(self, ur_interface: URInterface, normalize_observations: bool = False):
        """Initialize the ObservationParser."""
        self._ur_interface = ur_interface
        self._normalize_observations = normalize_observations
        self._latest_observation = None

    @property
    def latest_observation(self):
        """Get the last observation received/computed."""
        return self._latest_observation

    @abstractmethod
    def get_observation_space(self) -> gym.spaces.Space:
        """Get the observation space of the observation parser."""
        raise NotImplementedError

    @abstractmethod
    def get_observation(self) -> np.ndarray:
        """Get the observation of the observation parser."""
        raise NotImplementedError
