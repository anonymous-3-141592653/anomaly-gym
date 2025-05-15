from itertools import cycle
from typing import Any

import numpy as np

from .entities import Action


class MoveUpDown:
    def __init__(self) -> None:
        self.action_list = [[0.0, 10.0]] * 1 + [[0.0, -10.0]] * 1
        self.reset()

    def __call__(self, *args: Any, **kwargs: Any) -> Action:
        action = Action()
        action.u = np.array(next(self.action_iter))
        return action

    def reset(self, seed=None):
        self.action_iter = cycle(self.action_list)


class MoveRandom:
    def __init__(self, scale=1.0) -> None:
        self.scale = scale
        self.reset()
        self.action = Action()

    def __call__(self, *args: Any, **kwargs: Any) -> Action:
        self.action.u = self.rng.uniform([-self.scale, -self.scale], [self.scale, self.scale])
        return self.action

    def reset(self):
        self.rng = np.random.default_rng()


class MoveRandomDirection:
    def __init__(self, scale=1.0) -> None:
        self.scale = scale
        self.action = Action()
        self.reset()

    def __call__(self, *args: Any, **kwargs: Any) -> Action:
        return self.action

    def reset(self):
        self.rng = np.random.default_rng()
        self.action.u = self.rng.uniform([-self.scale, -self.scale], [self.scale, self.scale])


class MoveCircle:
    def __init__(self, radius=0.25, scale=0.1) -> None:
        self.scale = scale
        self.radius = radius
        self.action = Action()
        self.reset()

    def __call__(self, *args: Any, **kwargs: Any) -> Action:
        self.phase += self.scale
        self.action.u = self.radius * np.array([np.sin(self.phase), np.cos(self.phase)])
        return self.action

    def reset(self, seed=None):
        self.phase = np.random.randint(0, 10)
