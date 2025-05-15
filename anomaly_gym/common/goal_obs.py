from abc import ABC, abstractmethod

import numpy as np
from gymnasium import spaces


class GoalObs(ABC):
    goal_shape: tuple
    obs_shape: tuple

    @property
    def space(self) -> spaces.Space:
        """Return the observation space."""

        if not hasattr(self, "observation_space"):
            self.observation_space = spaces.Dict(
                dict(
                    observation=spaces.Box(-np.inf, np.inf, self.obs_shape, dtype=np.float32),
                    achieved_goal=spaces.Box(-np.inf, np.inf, self.goal_shape, dtype=np.float32),
                    desired_goal=spaces.Box(-np.inf, np.inf, self.goal_shape, dtype=np.float32),
                )
            )
        return self.observation_space

    def observe(self, *args, **kwargs) -> dict:
        """Return an observation of the environment state."""

        return {
            "observation": self._get_observation(*args, **kwargs),
            "achieved_goal": self._get_achived_goal(*args, **kwargs),
            "desired_goal": self._get_desired_goal(*args, **kwargs),
        }

    @abstractmethod
    def _get_observation(self, env, ego_state, sensor_data):
        pass

    @abstractmethod
    def _get_achived_goal(self, ego_state):
        pass

    @abstractmethod
    def _get_desired_goal(self):
        pass
