import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import seeding

from anomaly_gym.common.goal_env import GoalEnv
from anomaly_gym.envs.sape_envs.core.entities import Agent, Goal, Hazard, Object
from anomaly_gym.envs.sape_envs.core.observation import ImgObservation, LidarObservation, VectorObservation
from anomaly_gym.envs.sape_envs.core.render import PygameRenderer
from anomaly_gym.envs.sape_envs.tasks import task_registry


class SapeEnv(GoalEnv):
    """
    The gymnnasium.Env style interface for the agent to interact with the world.
    Receives the task and the world and defines basic interactions.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(
        self,
        task_id: str,
        task_config: dict | None = None,
        render_mode: str = "rgb_array",
        observation_type: str = "lidar",
        action_type: str = "continuous",
        reward_cfg: dict | None = None,
        info_level=None,
        lidar_num_bins=8,
        render_width=128,
        render_height=128,
    ):
        super().__init__()
        self.task_id = task_id
        self.task = self._get_task(task_id, task_config)
        self.render_mode = render_mode
        self.observation_type = observation_type
        self.action_type = action_type
        self.info_level = info_level
        self.lidar_num_bins = lidar_num_bins

        self.renderOn = False
        self.renderer = PygameRenderer(width=render_width, height=render_height)
        self.world = self.task.make_world()

        self.episodes = 0
        self.steps = 0
        self._init_action_space()
        self._init_observation_encoder()

        if reward_cfg is None:
            reward_cfg = {}

        # Dense reward multiplied by (squared) distance to goal
        self.reward_distance = reward_cfg.get("reward_distance", -1)
        # Sparse reward for colliding
        self.reward_critical = reward_cfg.get("reward_critical", -10)
        # Sparse reward for being inside the goal area
        self.reward_goal = reward_cfg.get("reward_goal", 0)
        # Sparse reward for being inside goal AND not colliding
        self.reward_success = reward_cfg.get("reward_success", 5)

        self.num_actors = len(self.world.actors)
        self.seed()

    @property
    def observation_space(self) -> spaces.Space:
        return self._observation_encoder.space

    def _get_task(self, task_id: str, task_config: dict | None = None):
        if task_config is None:
            task_config = {}
        task_class = task_registry[task_id]
        task = task_class(**task_config)
        return task

    def _init_observation_encoder(self) -> None:
        if self.observation_type == "vector":
            self._observation_encoder = VectorObservation(self.world)

        elif self.observation_type == "img":
            self._observation_encoder = ImgObservation(self.world)

        elif self.observation_type == "lidar":
            self._observation_encoder = LidarObservation(self.world, self.lidar_num_bins)

        else:
            raise NotImplementedError

    def _init_action_space(self) -> None:
        if self.action_type == "continuous":
            act_dim = 2
            self.action_space = spaces.Box(low=-1, high=1, shape=(act_dim,))
        elif self.action_type == "discrete":
            act_dim = 5
            self.action_space = spaces.Discrete(act_dim)
        else:
            raise ValueError

    def _get_info(self, observation):
        is_at_goal = self._is_at_goal(observation["achieved_goal"], observation["desired_goal"])
        is_critical = self._is_critical()

        self.was_critical = is_critical or self.was_critical

        is_success = self._is_success(observation["achieved_goal"], observation["desired_goal"], self.was_critical)

        cost = int(is_critical)
        info = {
            "is_success": is_success,
            "is_critical": is_critical,
            "is_at_goal": is_at_goal,
            "cost": cost,
        }
        if self.info_level == "debug":
            info["internal_state"] = self.get_internal_state()
        return info

    def seed(self, seed=None) -> None:
        self.np_random, seed = seeding.np_random(seed)
        self._observation_encoder.space.seed(seed)
        self.action_space.seed(seed)

    def reset(self, seed=None, **kwargs) -> tuple:
        if seed is not None:
            self.seed(seed=seed)

        self.task.reset_world(self.world, self.np_random)
        self.world.reset_sim()
        self.was_critical = False

        self.episodes += 1
        self.steps = 0

        observation = self._observation_encoder.observe(self.world)
        info = self._get_info(observation)

        return observation, info

    def step(self, action) -> tuple[dict, float | np.ndarray, bool, bool, dict]:
        self._execute_world_step(action)
        observation = self._observation_encoder.observe(self.world)
        info = self._get_info(observation)
        terminated = self.compute_terminated(observation["achieved_goal"], observation["desired_goal"], info)
        truncated = self.compute_truncated(observation["achieved_goal"], observation["desired_goal"], info)
        reward = self.compute_reward(observation["achieved_goal"], observation["desired_goal"], info)

        return observation, reward, terminated, truncated, info

    def _is_object_collision(self, agent_pos, objects_pos):
        agent_size = Agent.size
        object_size = Object.size
        return any([np.linalg.norm(agent_pos - obj_pos) < (agent_size + object_size) for obj_pos in objects_pos])

    def _is_hazard_collision(self, agent_pos, hazards_pos):
        agent_size = Agent.size
        hazard_size = Hazard.size
        return any([np.linalg.norm(agent_pos - haz_pos) < (agent_size + hazard_size) for haz_pos in hazards_pos])

    def _is_at_goal(self, achieved_goal, desired_goal) -> bool:
        return self._distance(achieved_goal, desired_goal).item() < self.world.goal.size

    def _is_success(self, achieved_goal, desired_goal, was_critical) -> bool:
        return self._is_at_goal(achieved_goal, desired_goal) and not was_critical

    def _is_critical(self) -> bool:
        agent_pos = self.world.agent.state.p_pos
        objects_pos = [obj.state.p_pos for obj in self.world.objects]
        hazards_pos = [haz.state.p_pos for haz in self.world.hazards]
        return self._is_object_collision(agent_pos, objects_pos) or self._is_hazard_collision(agent_pos, hazards_pos)

    def _distance(self, a, b):
        return np.linalg.norm(a - b, axis=-1, keepdims=True)

    def _squared_error(self, a, b):
        return np.sum((a - b) ** 2, axis=-1, keepdims=True)

    def compute_reward(self, achieved_goal, desired_goal, info) -> float | np.ndarray:
        if isinstance(info, np.ndarray):
            vec_reward = []
            for i in range(len(info)):
                r = self.reward_distance * self._squared_error(achieved_goal[i], desired_goal[i])
                if info[i]["is_at_goal"]:
                    r += self.reward_goal
                if info[i]["is_critical"]:
                    r += self.reward_critical
                if info[i]["is_success"]:
                    r += self.reward_success
                vec_reward.append(r)
            return np.array(vec_reward)

        else:
            r = self.reward_distance * self._squared_error(achieved_goal, desired_goal)
            if info["is_at_goal"]:
                r += self.reward_goal
            if info["is_critical"]:
                r += self.reward_critical
            if info["is_success"]:
                r += self.reward_success
            return float(r)

    def compute_terminated(self, achieved_goal, desired_goal, info) -> bool:
        """non-terminal env"""
        return info["is_at_goal"]

    def compute_truncated(self, achieved_goal, desired_goal, info) -> bool:
        """done with gymnasium TimeLimit wrapper."""
        return False

    def _execute_world_step(self, action):
        self._current_action = action
        self._set_action(self._current_action, self.world.agent)
        self.world.simulate_step()
        self.steps += 1

    def _set_action(self, action, actor, time=None):
        """
        set action for a particular actor
        """
        actor.action.u = np.zeros(self.world.dim_p)

        if actor.movable:
            # physical action
            actor.action.u = np.zeros(self.world.dim_p)
            if self.action_type:
                # Process continuous action
                actor.action.u[0] += action[0]
                actor.action.u[1] += action[1]
            else:
                # process discrete action
                if action == 1:
                    actor.action.u[0] = -1.0
                if action == 2:
                    actor.action.u[0] = +1.0
                if action == 3:
                    actor.action.u[1] = -1.0
                if action == 4:
                    actor.action.u[1] = +1.0

            actor.action.u *= actor.accel

    def get_internal_state(self):
        internal_state_dict = {}
        for e in self.world.entities:
            internal_state_dict[e.name] = {"p_pos": e.state.p_pos.copy(), "p_vel": e.state.p_vel.copy()}
        return internal_state_dict

    def set_internal_state(self, internal_state_dict):
        self.reset()
        for ent, (ent_name, state) in zip(self.world.entities, internal_state_dict.items()):
            assert ent.name == ent_name
            ent.state.p_pos = state["p_pos"].copy()
            ent.state.p_vel = state["p_vel"].copy()

    def render(self):
        return self.renderer.render(self.world, self.render_mode)

    def close(self):
        if self.renderOn:
            pygame.event.pump()
            pygame.display.quit()
            self.renderOn = False
