from typing import Callable

import numpy as np


class EntityState:
    """
    physical base state of all entities
    """

    def __init__(self):
        self.p_pos: np.ndarray = np.array([0.0, 0.0], dtype=np.float32)  # physical position
        self.p_vel: np.ndarray = np.array([0.0, 0.0], dtype=np.float32)  # physical velocity


class Action:
    """
    base class for binding actions
    """

    def __init__(self):
        self.u: None | np.ndarray = None  # physical action vector
        self.u_noise: float = 0.0  # physical motor noise vector


class Entity:
    """
    base class for physical entities in the world
    """

    collide: bool
    movable: bool
    size: float
    color: str
    symbol: str
    max_speed = 1.0
    initial_mass = 1.0
    accel = 5.0

    def __init__(self, name: str):
        """
        Args:
            name: unique identifier of the entity
        """
        self.name = name
        self.state = EntityState()
        self.action = Action()

    @property
    def mass(self) -> float:
        return self.initial_mass

    def distance_to(self, other) -> np.floating:
        return np.linalg.norm(self.state.p_pos - other.state.p_pos)


class Object(Entity):
    """
    an object is a physical entitity that other entities can collide with
    """

    collide = True
    movable = True
    size = 0.1
    color = "#8f836e"
    symbol = "O"

    def __init__(self, name: str, action_callback: None | Callable = None):
        """
        Args:
            name: unique name of the object
            action_callback: if None the object is static, otherwise this function is called on every env step
        """
        super().__init__(name=name)
        self.action_callback = action_callback


class Hazard(Entity):
    """
    an object is a physical entitity that other entities can collide with
    """

    collide = False
    movable = False
    size = 0.2
    color = "#cd7cff"
    symbol = "H"

    def __init__(self, name: str, action_callback: None | Callable = None):
        """
        Args:
            name: unique name of the object
            action_callback: if None the object is static, otherwise this function is called on every env step
        """
        super().__init__(name=name)
        self.action_callback = action_callback


class Goal(Entity):
    """
    a goal is an area that other entities cannot collide with and wich doesn't move
    """

    collide = False
    movable = False
    size = 0.2
    color = "#47BBC8"
    symbol = "G"


class Agent(Entity):
    """
    an agent is a physical entitity moving around in the world that other entities can collide with
    """

    collide = True
    movable = True
    size = 0.05  # agent default size smaller than objects/goals
    color = "#F36966"
    symbol = "A"
