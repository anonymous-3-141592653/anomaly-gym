"""Child Environment for the UR mujoco environment to reach a target."""

import numpy as np
from omegaconf import OmegaConf
from typing_extensions import override

from anomaly_gym.envs.ur2l_envs.envs.mujoco_env import URMujocoEnv
from anomaly_gym.envs.ur2l_envs.interfaces.limits import ReachLimits
from anomaly_gym.envs.ur2l_envs.sim.base import make_sim


class URMujocoReachEnv(URMujocoEnv):
    """Child Environment for the UR mujoco environment to reach a target."""

    def __init__(
        self,
        action_config: dict | None = None,
        observation_config: dict | None = None,
        reward_cfg: dict | None = None,
        sample_goal_steps: int = 50,
        goal_threshhold: float = 0.02,
        **kwargs,
    ):
        """Initialize the environment."""

        default_observation_config = {"end_effector": {}, "goal": {}}
        default_action_config = {"action_type": "cartesian", "gripper_action_type": "none"}
        default_reward_config = {
            "goal_progress_multiplier": 1.0,
            "action_norm_multiplier": 0.1,
            "action_delta_multiplier": 0.0,
            "ee_orientation_multiplier": 0.0,
            "collision_penalty": 0.0,
            "joint_limit_penalty": 0.0,
            "ee_pos_limit_penalty": 0.0,
            "ee_rot_limit_penalty": 0.0,
            "sparse_goal_reached": 0.0,
        }
        observation_config = OmegaConf.merge(default_observation_config, observation_config or {})
        action_config = OmegaConf.merge(default_action_config, action_config or {})
        self.reward_cfg = OmegaConf.merge(default_reward_config, reward_cfg or {})
        limits = ReachLimits()
        super().__init__(**kwargs, limits=limits, action_config=action_config, observation_config=observation_config)

        # Some variables to keep track of the goal.
        self._goal_threshold = goal_threshhold
        self._sample_goal_steps = sample_goal_steps

    def _make_sim(self):
        return make_sim(sim_name="reach", control_type=self.control_type, ur_config="ready")

    @override
    def _post_sim_step(self, action):
        # Sample a new goal every sample_goal_steps
        self._goal_counter += 1
        if self._goal_counter % self._sample_goal_steps == 0:
            self._sample_goal()
        return super()._post_sim_step(action)

    def _sample_goal(self):
        new_goal = self.np_random.uniform(low=self.limits.goal_min, high=self.limits.goal_max)
        self._ur_interface.set_goal(new_goal[:3], new_goal[3:7])

    def _reset(self):
        """Reset the environment."""
        self._reset_mujoco()
        self._sample_goal()
        self._prev_action = np.zeros(self.action_space.shape)
        self._goal_counter = 0
        self._goal_reached = False
        self._goal_distance = 0.0

    def _step(self, action):
        self.sim.advance_simulation(n_steps=self.n_sim_steps)

    @override
    def get_reward(self, action: np.ndarray) -> float:
        # goal progress reward
        self._goal_distance = np.linalg.norm(
            self._ur_interface.end_effector_position - self._ur_interface.goal_position
        )
        self._goal_reached = self._goal_distance < self._goal_threshold
        if self._goal_reached:
            self._goal_progress_reward = 0.0
        else:
            self._goal_progress_reward = -self.reward_cfg.goal_progress_multiplier * self._goal_distance
            self._goal_progress_reward -= 1

        # action penalties
        action_norm = np.linalg.norm(action)
        action_delta = np.linalg.norm(self._prev_action - action)
        self._action_norm_reward = -self.reward_cfg.action_norm_multiplier * action_norm
        self._action_delta_reward = -self.reward_cfg.action_delta_multiplier * action_delta

        # orientation penalties
        quat_dist = 1 - np.abs(np.dot(self._ur_interface.end_effector_quaternion, self._ur_interface.goal_quaternion))
        self._ee_orientation_reward = -self.reward_cfg.ee_orientation_multiplier * quat_dist

        self._full_reward = (
            self._goal_progress_reward,
            self._action_norm_reward,
            self._action_delta_reward,
            self._ee_orientation_reward,
            self.reward_cfg.collision_penalty if self._ur_interface.body_collision else 0.0,
            self.reward_cfg.sparse_goal_reached if self._goal_reached else 0.0,
            self.reward_cfg.joint_limit_penalty if self.reached_joint_limits() else 0.0,
            self.reward_cfg.ee_pos_limit_penalty if self.reached_ee_pos_limits() else 0.0,
            self.reward_cfg.ee_rot_limit_penalty if self.reached_ee_rot_limits() else 0.0,
        )

        self._prev_action = action

        return sum(self._full_reward)

    def get_info(self) -> dict:
        return {
            "goal_reached": self._goal_reached,
            "goal_distance": self._goal_distance,
            "term_info": self.term_info,
        }
