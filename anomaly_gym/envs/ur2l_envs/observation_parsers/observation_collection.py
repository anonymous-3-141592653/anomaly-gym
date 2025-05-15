"""Classes and functions to create a collection of different observation parsers."""

import gymnasium as gym
from typing_extensions import override

from anomaly_gym.envs.ur2l_envs.interfaces import URInterface
from anomaly_gym.envs.ur2l_envs.observation_parsers.base import ObservationParser
from anomaly_gym.envs.ur2l_envs.observation_parsers.block import BlockObservationParser
from anomaly_gym.envs.ur2l_envs.observation_parsers.collision import CollisionObservationParser
from anomaly_gym.envs.ur2l_envs.observation_parsers.end_effector import EndEffectorObservationParser
from anomaly_gym.envs.ur2l_envs.observation_parsers.goal import GoalObservationParser
from anomaly_gym.envs.ur2l_envs.observation_parsers.gripper import GripperObservationParser
from anomaly_gym.envs.ur2l_envs.observation_parsers.joints import JointsObservationParser
from anomaly_gym.envs.ur2l_envs.observation_parsers.target import TargetObservationParser


class ObservationCollection(ObservationParser):
    """Collection of different observation parsers."""

    def __init__(self, observation_parsers: dict[str, ObservationParser]):
        """Initialize the ObservationCollection.

        Args:
            observation_parsers: Dictionary of observation parsers.
        """
        self.observation_parsers = observation_parsers

    def get_observation_space(self) -> gym.spaces.Dict:
        """Return the combined observation space."""
        return gym.spaces.Dict(
            {
                obs_parser_name: obs_parser.get_observation_space()
                for obs_parser_name, obs_parser in self.observation_parsers.items()
            }
        )

    def get_observation(self) -> dict:
        """Get an observation as a combination of all observations."""
        return {
            obs_parser_name: obs_parser.get_observation()
            for obs_parser_name, obs_parser in self.observation_parsers.items()
        }

    def get_stats(self) -> dict:
        """Get statistics of each individual observation parser."""
        return {
            obs_parser_name: obs_parser.get_stats() for obs_parser_name, obs_parser in self.observation_parsers.items()
        }

    @override
    @property
    def latest_observation(self):
        return {
            obs_parser_name: obs_parser.latest_observation
            for obs_parser_name, obs_parser in self.observation_parsers.items()
        }


obs_parsers = {
    "joints": JointsObservationParser,
    "end_effector": EndEffectorObservationParser,
    "gripper": GripperObservationParser,
    "goal": GoalObservationParser,
    "collision": CollisionObservationParser,
    "target": TargetObservationParser,
    "block": BlockObservationParser,
}


def make_observation_parster(observation_parser_type: str, **kwargs) -> ObservationParser:
    """Make an action setter object."""
    _cls = obs_parsers.get(observation_parser_type)
    if _cls is not None:
        return _cls(**kwargs)
    else:
        raise NotImplementedError(f"Unknown action setter type: {observation_parser_type}.")


def make_observation_collection(ur_interface: URInterface, **observation_collection_config: dict[str, any]):
    """Make an ObservationCollection based on a config.

    Args:
        ur_interface: URInterface instance.
        observation_collection_config: dictionary for containing the config of each observation_parser.
    """
    obs_parsers_for_collection = {}
    for obs_parser_name, obs_parser_config in observation_collection_config.items():
        obs_parser = make_observation_parster(obs_parser_name, ur_interface=ur_interface, **obs_parser_config)
        obs_parsers_for_collection[obs_parser_name] = obs_parser
    return ObservationCollection(observation_parsers=obs_parsers_for_collection)
