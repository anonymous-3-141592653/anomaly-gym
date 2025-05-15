"""Mujoco Gym environment used for training."""

import logging
from abc import abstractmethod
from typing import Literal

import mujoco
import numpy as np
from gymnasium.core import RenderFrame
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from typing_extensions import override

from anomaly_gym.envs.ur2l_envs.action_setters import make_action_setter
from anomaly_gym.envs.ur2l_envs.envs.ur_env import UREnv
from anomaly_gym.envs.ur2l_envs.interfaces.limits import URLimits
from anomaly_gym.envs.ur2l_envs.interfaces.ur_mujoco_interface import URMujocoInterface
from anomaly_gym.envs.ur2l_envs.observation_parsers import make_observation_collection

logger = logging.getLogger(__name__)


class URMujocoEnv(UREnv):
    """Environment to facilitate training using a mujoco simulatin of the UR robot."""

    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 25}

    def __init__(
        self,
        limits: URLimits,
        action_config: dict,
        observation_config: dict,
        ctrl_freq: float = 25.0,
        terminate_on_limits: bool = False,
        render_mode: Literal["human", "rgb_array", "depth_array"] = "rgb_array",
        render_cfg: dict | None = None,
    ):
        """Initialize the environment.

        Args:
            ctrl_freq: frequency at which we actuate the system (policy frequency)
            sim_config: keyword arguments with the configuration of the simulation.
            action_setter_config: keyword arguments for the action transform.
            observation_parser_config: keyword arguments for the observation parser.
            render_mode: "human", "rgb_array", or "depth_array".
        """
        super().__init__(ctrl_freq=ctrl_freq)
        self._terminate_on_limits = terminate_on_limits
        action_type = action_config["action_type"]
        if action_type == "cartesian":
            self.control_type = "mocap"
        elif action_type == "joint":
            self.control_type = "joint"
        else:
            raise ValueError(f"Action type {action_type} is not supported. Supported types: 'joint' and 'cartesian'.")

        self.sim = self._make_sim()
        self._model, self._data = self.sim.model, self.sim.data
        self._step_since_reset = 0
        self.limits = limits
        self._ur_interface = URMujocoInterface(sim=self.sim, limits=self.limits)
        self.render_mode = render_mode

        if render_cfg is None:
            self.render_cfg = {
                "camera_name": "diagonal_front",
                "width": 128 if self.render_mode == "rgb_array" else None,
                "height": 128 if self.render_mode == "rgb_array" else None,
            }
        else:
            self.render_cfg = render_cfg

        self.sim_dt = self._model.opt.timestep
        self.sim_freq = 1.0 / self.sim_dt
        self.n_sim_steps = int(self.sim_freq / self.ctrl_freq)

        self._obs_parser = make_observation_collection(ur_interface=self._ur_interface, **observation_config)
        self._action_setter = make_action_setter(ur_interface=self._ur_interface, **action_config)

        self.observation_space = self._obs_parser.get_observation_space()
        self.action_space = self._action_setter.get_action_space()

        logger.info("Initialized URMujoco.")

    @abstractmethod
    def _make_sim(self, **kwargs) -> mujoco.MjModel: ...

    @property
    def type(self):
        """Type of the environment: mujoco..."""
        return "mujoco"

    @property
    def time(self):
        """Current simulation time."""
        return self._data.time

    def _reset_mujoco(self):
        """Reset the mujoco environment to its initial configuration."""
        mujoco.mj_resetDataKeyframe(self._model, self._data, 0)
        mujoco.mj_forward(self._model, self._data)

    @override
    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Render frame(s) of the scene."""

        if not hasattr(self, "_mujoco_renderer"):
            self._mujoco_renderer = MujocoRenderer(model=self._model, data=self._data, **self.render_cfg)

        return self._mujoco_renderer.render(render_mode=self.render_mode)

    @abstractmethod
    def _step(self, action): ...

    @override
    def step(self, action):
        """Take a step in the environment.

        Args:
            action: np.array containing the action normalized between [-1, 1]^action_dim

        Returns:
            obs: np.array observation of the current state
            reward: np.array reward of the action given the current state of the environment
            terminated: bool indication if the episode terminated
            truncated: bool indicating if the episode has been truncated
            info: dict with auxiliary information

        """

        self._pre_sim_step(action=action)
        self._step(action)
        return self._post_sim_step(action=action)

    @abstractmethod
    def _reset(self): ...

    @override
    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._reset()
        self.term_info = "not_terminated"
        obs = self._obs_parser.get_observation()
        info = self.get_info()
        return obs, info

    @override
    def close(self):
        if hasattr(self, "_mujoco_renderer"):
            self._mujoco_renderer.close()

    def _pre_sim_step(self, action):
        """Prepare the environment for the next simulation step.

        This includes setting the action and advancing the simulation. It can be used to override the step
        function in the child class more granularly.

        Args:
            action: np.array containing the action normalized between [-1, 1]^action_dim
        """
        self._step_since_reset += 1
        self._action_setter.set_action(action=action)

    def _post_sim_step(self, action) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        """Run the post advance simulation step in the environment.

        This includes getting the observation, reward, termination and info. It can be used to override the step
        function in the child class more granularly.

        Args:
            action: np.array containing the action normalized between [-1, 1]^action_dim
        Returns:
            obs: np.array observation of the current state
            reward: np.array reward of the action given the current state of the environment
            terminated: bool indication if the episode terminated
            truncated: bool indicating if the episode has been truncated
            info: dict with auxiliary information
        """
        obs = self._obs_parser.get_observation()
        reward = self.get_reward(action=action)
        terminated = self.compute_terminated()
        truncated = self.compute_truncated()
        info = self.get_info()
        return obs, reward, terminated, truncated, info

    @abstractmethod
    def get_reward(self, action: np.ndarray) -> float:
        """Compute a reward based on the current observation and action."""
        ...

    @abstractmethod
    def get_info(self) -> dict:
        """Get auxiliary information about the environment."""
        ...

    def compute_terminated(self) -> bool:
        terminated = False

        if self._terminate_on_limits:
            if self.reached_joint_limits():
                terminated = True
                self.term_info = "joint_limits"
            if self.reached_ee_pos_limits():
                terminated = True
                self.term_info = "ee_pos_limits"
            if self.reached_ee_rot_limits():
                terminated = True
                self.term_info = "ee_rot_limits"
            if self._ur_interface.body_collision:
                terminated = True
                self.term_info = "body_collision"

        return terminated

    def compute_truncated(self) -> bool:
        return False

    def reached_joint_limits(self):
        """Check if the robot reached the joint limits."""
        joint_pos = self._ur_interface.joint_position
        return np.any(joint_pos < self.limits.joint_min) or np.any(joint_pos > self.limits.joint_max)

    def reached_ee_pos_limits(self):
        """Check if the robot reached the end effector position limits."""
        ee_pos = self._ur_interface.end_effector_position
        return np.any(ee_pos < self.limits.ee_pos_min) or np.any(ee_pos > self.limits.ee_pos_max)

    def reached_ee_rot_limits(self):
        """Check if the robot reached the end effector rotation limits."""
        ee_rot = np.abs(self._ur_interface.end_effector_rpy)
        return np.any(ee_rot < self.limits.ee_rot_min) or np.any(ee_rot > self.limits.ee_rot_max)
