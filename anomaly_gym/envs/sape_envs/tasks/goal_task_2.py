from ..core.entities import Agent, Goal, Object
from ..core.object_callbacks import MoveCircle
from ..core.world import World
from .goal_task_1 import GoalTask1


class GoalTask2(GoalTask1):
    """A simple task with 5 obstacles spawned around the goal htat are moving in circles"""

    def __init__(self) -> None:
        super().__init__()
        self.n_obstacles = 10
        self.n_hazards = 3
