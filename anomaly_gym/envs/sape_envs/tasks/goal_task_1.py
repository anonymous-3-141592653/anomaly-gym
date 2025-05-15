import numpy as np

from ..core.base_task import BaseTask
from ..core.entities import Agent, Goal, Hazard, Object
from ..core.world import World


def sample_vector_2D(norm_min=0.0, norm_max=1.0, theta_min=0.0, theta_max=90.0, rng: None | np.random.Generator = None):
    if rng is None:
        rng = np.random.default_rng()
    r = np.sqrt(rng.uniform(low=norm_min**2, high=norm_max**2))
    theta = rng.uniform(low=theta_min / 360, high=theta_max / 360) * 2 * np.pi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.array([x, y], dtype=np.float32)


def sample_circle_2D(radius=1.0, rng: None | np.random.Generator = None):
    return sample_vector_2D(norm_min=0, norm_max=radius, theta_min=0, theta_max=360, rng=rng)


def sample_ring_2D(inner_radius=0.0, outer_radius=1.0, rng: None | np.random.Generator = None):
    return sample_vector_2D(norm_min=inner_radius, norm_max=outer_radius, theta_min=0, theta_max=360, rng=rng)


class GoalTask1(BaseTask):
    """A simple task with 3 static obstacles and spawned around the goal"""

    def __init__(self) -> None:
        super().__init__()
        self.n_obstacles = 3
        self.n_hazards = 1

    def make_world(self) -> World:
        world = World(
            agent=Agent("agent_0"),
            objects=[Object(f"static_obstacle_{i}") for i in range(self.n_obstacles)],
            goal=Goal("static_goal_0"),
            hazards=[Hazard(f"static_hazard_{i}") for i in range(self.n_hazards)],
        )
        return world

    def _init_agent(self, world, rng: np.random.Generator):
        """init agent at random position"""

        world.agent.state.p_pos = rng.uniform([-1, -1], [1, 1], world.dim_p).astype(np.float32)
        world.agent.state.p_vel = np.zeros(world.dim_p).astype(np.float32)

    def _init_goal(self, world, rng: np.random.Generator):
        """init goal at random position, at least init_distance away from agent"""
        while True:
            world.goal.state.p_pos = rng.uniform([-1, -1], [1, 1], world.dim_p).astype(np.float32)
            world.goal.state.p_vel = np.zeros(world.dim_p).astype(np.float32)
            # resample if distance too small
            if np.linalg.norm(world.agent.state.p_pos - world.goal.state.p_pos) > self.min_dist_goal:
                break

    def _init_obstacles(self, world, rng: np.random.Generator):
        "init obstacles with min distance to all other entities"

        other_pos = [world.agent.state.p_pos, world.goal.state.p_pos]

        # spawn obstacles around the goal
        for obj in world.objects:
            done = False
            while not done:
                pos = sample_ring_2D(inner_radius=world.goal.size + obj.size, outer_radius=1, rng=rng)
                pos += world.goal.state.p_pos
                done = all([np.linalg.norm(pos - other) > (self.min_dist_obj) for other in other_pos])
            obj.state.p_pos = pos
            obj.state.p_vel = np.zeros(world.dim_p).astype(np.float32)
            other_pos.append(pos)

    def _init_hazards(self, world: World, rng: np.random.Generator):
        # spawn obstacles around the goal

        other_pos = [world.agent.state.p_pos, world.goal.state.p_pos, *[o.state.p_pos for o in world.objects]]

        for haz in world.hazards:
            done = False
            while not done:
                pos = rng.uniform([-1, -1], [1, 1], world.dim_p).astype(np.float32)
                done = all([np.linalg.norm(pos - other) > (self.min_dist_haz) for other in other_pos])
            haz.state.p_pos = pos
            other_pos.append(pos)
