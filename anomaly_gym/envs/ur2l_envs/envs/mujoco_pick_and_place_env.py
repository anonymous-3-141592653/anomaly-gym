"""Child Environment for the UR mujoco environment to reach a target."""

import numpy as np
from omegaconf import OmegaConf
from typing_extensions import override

from anomaly_gym.envs.ur2l_envs.envs.mujoco_env import URMujocoEnv
from anomaly_gym.envs.ur2l_envs.interfaces.limits import PickAndPlaceLimits
from anomaly_gym.envs.ur2l_envs.sim.base import make_sim


class URMujocoPickAndPlaceEnv(URMujocoEnv):
    """Child Environment for the UR mujoco environment to reach a target."""

    def __init__(
        self,
        action_config: dict | None = None,
        observation_config: dict | None = None,
        reward_cfg: dict | None = None,
        sample_goal_steps: int = 201,
        goal_threshhold: float = 0.02,
        **kwargs,
    ):
        """Initialize the environment."""
        default_observation_config = {"end_effector": {}, "goal": {}, "gripper": {}, "block": {}}
        default_action_config = {"action_type": "cartesian", "gripper_action_type": "binary"}
        default_reward_config = {
            "goal_progress_multiplier": 1.0,
            "grasp_reward_multiplier": 0.1,
            "action_norm_multiplier": 0.0,
            "action_delta_multiplier": 0.0,
            "ee_orientation_multiplier": 0.0,
            "top_contact_penalty": -1.0,
            "collision_penalty": 0.0,
            "joint_limit_penalty": 0.0,
            "ee_pos_limit_penalty": 0.0,
            "ee_rot_limit_penalty": 0.0,
            "sparse_goal_reached": 0.0,
        }
        observation_config = OmegaConf.merge(default_observation_config, observation_config or {})
        action_config = OmegaConf.merge(default_action_config, action_config or {})
        self.reward_cfg = OmegaConf.merge(default_reward_config, reward_cfg or {})
        limits = PickAndPlaceLimits()
        super().__init__(**kwargs, limits=limits, action_config=action_config, observation_config=observation_config)

        # Some variables to keep track of the goal.
        self._goal_threshold = goal_threshhold
        self._sample_goal_steps = sample_goal_steps

    def _make_sim(self):
        return make_sim(sim_name="pick_and_place", control_type=self.control_type, ur_config="ready")

    @override
    def _post_sim_step(self, action):
        # Sample a new goal if the goal is not fixed.
        self._goal_counter += 1
        if self._goal_counter % self._sample_goal_steps == 0:
            self._sample_goal()
        return super()._post_sim_step(action)

    def _sample_goal(self):
        new_goal = self.np_random.uniform(low=self.limits.goal_min, high=self.limits.goal_max)
        self._ur_interface.set_goal(new_goal[:3], new_goal[3:7])

    def _sample_block(self):
        """Sample block position ensuring it's reachable and not too close to goal."""
        new_block = self.np_random.uniform(low=self.limits.block_min, high=self.limits.block_max)
        self._ur_interface.set_block(new_block[:3], new_block[3:7])

    @override
    def _reset(self, *, seed=None, options=None):
        """Reset the environment."""
        self._reset_mujoco()
        self._sample_goal()
        self._sample_block()
        self._prev_action = np.zeros(self.action_space.shape)
        self._goal_counter = 0
        self._goal_reached = False
        self._goal_distance = 0.0
        self._block_distance = 0.0
        self._is_grasping = False
        self._top_contact = False

    def _step(self, action):
        self.sim.advance_simulation(n_steps=self.n_sim_steps)
        self._limit_box_vel()

    def _limit_box_vel(self):
        """Limit the velocity of the box to prevent it from flying away."""
        qvel = self.sim.data.joint("block_freejoint").qvel.copy()
        qvel = np.clip(qvel, -0.25, 0.25)
        self.sim.data.joint("block_freejoint").qvel = qvel

    @override
    def get_reward(self, action: np.ndarray) -> float:
        end_effector_pos = self._ur_interface.end_effector_position
        block_pos = self._ur_interface.block_position
        goal_pos = self._ur_interface.goal_position

        # # Distance calculations
        self._block_distance = np.linalg.norm(block_pos - end_effector_pos)
        self._goal_distance = np.linalg.norm(block_pos - goal_pos)
        self._goal_reached = self._goal_distance < self._goal_threshold

        # # Gripper state
        self._is_grasping = self.is_grasping()
        self._top_contact = self._ur_interface.get_touch_sensor_data()["top_touch"] > 0.5

        # action penalties
        action_norm = np.linalg.norm(action)
        action_delta = np.linalg.norm(self._prev_action - action)
        action_norm_reward = -self.reward_cfg.action_norm_multiplier * action_norm
        action_delta_reward = -self.reward_cfg.action_delta_multiplier * action_delta

        # orientation penalties
        quat_dist = 1 - np.abs(np.dot(self._ur_interface.end_effector_quaternion, self._ur_interface.goal_quaternion))
        ee_orientation_reward = -self.reward_cfg.ee_orientation_multiplier * quat_dist

        # Combine rewards
        goal_progress_reward = -(self._block_distance + self._goal_distance)

        self._full_reward = (
            self.reward_cfg.goal_progress_multiplier * goal_progress_reward,
            self.reward_cfg.grasp_reward_multiplier * self._is_grasping,
            self.reward_cfg.top_contact_penalty * self._top_contact,
            self.reward_cfg.action_norm_multiplier * action_norm_reward,
            self.reward_cfg.action_delta_multiplier * action_delta_reward,
            self.reward_cfg.ee_orientation_multiplier * ee_orientation_reward,
            self.reward_cfg.collision_penalty * self._ur_interface.body_collision,
            self.reward_cfg.sparse_goal_reached * self._goal_reached,
            self.reward_cfg.joint_limit_penalty * self.reached_joint_limits(),
            self.reward_cfg.ee_pos_limit_penalty * self.reached_ee_pos_limits(),
            self.reward_cfg.ee_rot_limit_penalty * self.reached_ee_rot_limits(),
        )

        self._prev_action = action

        return float(sum(self._full_reward))

    def get_info(self) -> dict:
        return {
            "goal_reached": self._goal_distance < self._goal_threshold,
            "goal_distance": self._goal_distance,
            "block_distance": self._block_distance,
            "block_reached": self._block_distance < self._goal_threshold,
            "grasping": self._is_grasping,
            "term_info": self.term_info,
            "top_contact": self._top_contact,
        }

    def is_grasping(self) -> bool:
        touch_sensor_data = self._ur_interface.get_touch_sensor_data()
        return touch_sensor_data["left_touch"] > 10 and touch_sensor_data["right_touch"] > 10
