# inspired by: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/half_cheetah_v4.py
# and https://github.com/kchua/handful-of-trials/blob/master/dmbrl/env/half_cheetah.py

import os
from typing import Any, ClassVar

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {"distance": 4.0}


class MujocoHalfCheetahEnv(MujocoEnv, utils.EzPickle):
    metadata: ClassVar = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 20}

    def __init__(self, render_mode: str = "rgb_array", width=128, height=128, **kwargs: Any) -> None:
        utils.EzPickle.__init__(self, **kwargs)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        observation_space = Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)
        MujocoEnv.__init__(
            self,
            model_path=f"{dir_path}/../assets/half_cheetah.xml",
            frame_skip=5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            render_mode=render_mode,
            width=width,
            height=height,
            **kwargs,
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.prev_qpos = self.data.qpos.flat.copy()
        action = action.reshape(
            6,
        )
        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()

        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = observation[0]
        reward = reward_run + reward_ctrl

        # if self.render_mode is not None:
        #     self.mujoco_renderer._get_viewer(self.render_mode).cam.lookat[0] = self.data.qpos.flat[0]

        is_success = observation[0] >= 10.0

        return observation, reward, False, False, {"is_success": is_success}

    def _get_obs(self) -> np.ndarray:
        observation = np.concatenate(
            [
                (self.data.qpos.flat[:1] - self.prev_qpos[:1]) / self.dt,
                self.data.qpos.flat[1:],
                self.data.qvel.flat,
            ],
            dtype=np.float32,
        )
        return observation

    def render(self):
        self.mujoco_renderer._get_viewer(self.render_mode).cam.lookat[0] = self.data.qpos.flat[0]
        return super().render()

    def reset_model(self) -> np.ndarray:
        qpos = self.init_qpos + self.np_random.normal(loc=0, scale=0.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.normal(loc=0, scale=0.1, size=self.model.nv)
        self.set_state(qpos, qvel)
        self.prev_qpos = np.copy(self.data.qpos.flat)
        observation = self._get_obs()
        return observation
