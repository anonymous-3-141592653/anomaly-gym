# inspired by https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/inverted_pendulum_v4.py
# and https://github.com/kchua/handful-of-trials/blob/master/dmbrl/env/cartpole.py

import os
from typing import Any, ClassVar

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box

PENDULUM_LENGTH = 0.6
DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0, "distance": 2.04}


class MujocoCartpoleSwingupEnv(MujocoEnv, utils.EzPickle):
    metadata: ClassVar = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 25}

    def __init__(
        self, render_mode: str = "rgb_array", observation_type="vector", width=64, height=64, **kwargs: Any
    ) -> None:
        utils.EzPickle.__init__(self, **kwargs)
        self.observation_type = observation_type
        if self.observation_type == "vector":
            observation_space: Box = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        elif self.observation_type == "image":
            observation_space: Box = Box(low=0, high=255, shape=(width, height, 3), dtype=np.uint8)
        else:
            raise ValueError(f"Invalid observation type: {self.observation_type}")

        dir_path: str = os.path.dirname(os.path.realpath(__file__))
        MujocoEnv.__init__(
            self,
            model_path=f"{dir_path}/../assets/cartpole.xml",
            frame_skip=2,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            render_mode=render_mode,
            width=width,
            height=height,
            **kwargs,
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.do_simulation(action, self.frame_skip)
        state = self._get_state()
        observation = self._get_obs()

        ee_disp = self._get_ee_pos(state) - np.array([0.0, PENDULUM_LENGTH])
        reward = np.exp(-np.sum(np.square(ee_disp)) / (PENDULUM_LENGTH**2))
        reward -= 0.01 * np.sum(np.square(action))

        is_success = np.linalg.norm(ee_disp) < 0.025

        return observation, reward, False, False, {"is_success": is_success}

    def reset_model(self) -> np.ndarray:
        qpos: np.ndarray = self.init_qpos + self.np_random.normal(0, 0.1, np.shape(self.init_qpos))
        qvel: np.ndarray = self.init_qvel + self.np_random.normal(0, 0.1, np.shape(self.init_qvel))
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_state(self):
        state = np.concatenate([self.data.qpos, self.data.qvel], dtype=np.float32).ravel()
        return state

    def _get_obs(self) -> np.ndarray:
        if self.observation_type == "image":
            observation: np.ndarray = self.render()
        else:
            observation: np.ndarray = self._get_state()
        return observation

    @staticmethod
    def _get_ee_pos(x: np.ndarray) -> np.ndarray:
        x0, theta = x[0], x[1]
        return np.array([x0 - PENDULUM_LENGTH * np.sin(theta), -PENDULUM_LENGTH * np.cos(theta)])
