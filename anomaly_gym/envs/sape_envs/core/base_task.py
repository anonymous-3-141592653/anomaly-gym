from abc import ABC, abstractmethod

import numpy as np

from .entities import Agent, Goal, Hazard, Object
from .world import World


class BaseTask(ABC):
    """
    Task upon which the world is built. Defines functions to make & reset the world, the reward,
    terminal state and observation

    """

    def __init__(self) -> None:
        self.min_dist_goal = Goal.size * 5
        self.min_dist_obj = Object.size + Goal.size
        self.min_dist_haz = Hazard.size * 2
        super().__init__()

    @abstractmethod
    def make_world(self) -> World:
        """
        create world and all its entities
        """
        pass

    def reset_world(self, world, rng: np.random.Generator) -> None:
        self._init_agent(world, rng)
        self._init_goal(world, rng)
        self._init_obstacles(world, rng)
        self._init_hazards(world, rng)
        self._reset_object_callbacks(world, rng)

    @abstractmethod
    def _init_agent(self, world: World, rng: np.random.Generator):
        pass

    @abstractmethod
    def _init_goal(self, world: World, rng: np.random.Generator):
        pass

    @abstractmethod
    def _init_obstacles(self, world: World, rng: np.random.Generator):
        pass

    @abstractmethod
    def _init_hazards(self, world: World, rng: np.random.Generator):
        pass

    def _reset_object_callbacks(self, world: World, rng: np.random.Generator) -> None:
        for obj in world.objects:
            if obj.action_callback is not None:
                obj.action_callback.reset()
