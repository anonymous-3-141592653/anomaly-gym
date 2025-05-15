"""UR environments base class and utils functions."""

from __future__ import annotations

import typing
from abc import ABC, abstractmethod

from gymnasium import Env

from anomaly_gym.envs.ur2l_envs.action_setters.action_setter import ActionSetter
from anomaly_gym.envs.ur2l_envs.observation_parsers.observation_collection import ObservationCollection


class UREnv(ABC, Env):
    """Base class for environments based on the UR robot."""

    _obs_parser: ObservationCollection
    _action_setter: ActionSetter

    def __init__(self, ctrl_freq: float):
        """Initialize the environment."""
        self.ctrl_freq = ctrl_freq
        self.ctrl_dt = 1.0 / ctrl_freq

    @property
    def observation_parser(self) -> ObservationCollection:
        """Get the observation_parser."""
        return self._obs_parser

    @property
    def action_setter(self) -> ActionSetter:
        """Get the action_setter."""
        return self._action_setter

    @property
    @abstractmethod
    def type(self) -> str:
        """Type of the environment: real, mujoco..."""
        raise NotImplementedError

    @property
    @abstractmethod
    def time(self) -> int | float:
        """Current time in the environment."""
        raise NotImplementedError
