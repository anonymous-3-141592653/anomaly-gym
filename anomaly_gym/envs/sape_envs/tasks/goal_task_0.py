import numpy as np

from ..core.base_task import BaseTask
from ..core.entities import Agent, Goal, Object
from ..core.world import World


class GoalTask0(BaseTask):
    """A simple task with 1 static obstacle spawned between agent and goal"""

    def make_world(self) -> World:
        world = World(
            agent=Agent("agent_0"), objects=[Object("static_obstacle_0")], goal=Goal("static_goal_0"), hazards=[]
        )
        return world

    def _init_agent(self, world, rng):
        """init agent at random position in the left half of the field"""

        world.agent.state.p_pos = rng.uniform([-1, -1], [0, 1], world.dim_p).astype(np.float32)
        world.agent.state.p_vel = np.zeros(world.dim_p).astype(np.float32)

    def _init_goal(self, world, rng):
        """init goal at random position in the left half of the field, at least init_distance away from agent"""
        while True:
            world.goal.state.p_pos = rng.uniform([-1, -1], [0, 1], world.dim_p).astype(np.float32)
            world.goal.state.p_vel = np.zeros(world.dim_p).astype(np.float32)
            # resample if distance too small
            if np.linalg.norm(world.agent.state.p_pos - world.goal.state.p_pos) > self.min_dist_goal:
                break

    def _init_obstacles(self, world, rng):
        "init obstacle between agent and goal"
        obj = world.objects[0]
        obj.state.p_pos = (world.agent.state.p_pos + world.goal.state.p_pos) / 2
        obj.state.p_vel = np.zeros(world.dim_p).astype(np.float32)

    def _init_hazards(self, world, rng):
        pass
