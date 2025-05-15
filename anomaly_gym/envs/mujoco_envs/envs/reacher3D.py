# inspired by: https://github.com/kchua/handful-of-trials/blob/master/dmbrl/env/reacher.py
# and https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/reacher.py

import os
from collections import OrderedDict
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": -1,
    "distance": 4.0,
}


class MujocoReacher3DEnv(MujocoEnv, utils.EzPickle):
    metadata: ClassVar = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 50}

    def __init__(self, render_mode: str = "rgb_array", width=128, height=128, **kwargs: Any) -> None:
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        MujocoEnv.__init__(
            self,
            model_path=f"{dir_path}/../assets/reacher3d.xml",
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
        observation = self._get_obs()

        ee_disp = np.square(self.get_EE_pos(observation[None]) - self.goal)
        reward = -np.sum(ee_disp)
        reward -= 0.01 * np.square(action).sum()

        is_success = np.linalg.norm(ee_disp) < 0.1

        return observation, reward, False, False, {"is_success": is_success}

    def reset_model(self) -> np.ndarray:
        qpos, qvel = np.copy(self.init_qpos), np.copy(self.init_qvel)
        qpos[-3:] += self.np_random.normal(loc=0, scale=0.1, size=[3])
        qvel[-3:] = 0
        self.goal = qpos[-3:]
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        observation = np.concatenate([self.data.qpos.flat, self.data.qvel.flat[:-3]], dtype=np.float32)
        return observation

    def get_EE_pos(self, states: np.ndarray) -> np.ndarray:
        theta1, theta2, theta3, theta4, theta5, theta6, theta7 = (
            states[:, :1],
            states[:, 1:2],
            states[:, 2:3],
            states[:, 3:4],
            states[:, 4:5],
            states[:, 5:6],
            states[:, 6:],
        )

        rot_axis = np.concatenate(
            [np.cos(theta2) * np.cos(theta1), np.cos(theta2) * np.sin(theta1), -np.sin(theta2)], axis=1
        )
        rot_perp_axis = np.concatenate([-np.sin(theta1), np.cos(theta1), np.zeros(theta1.shape)], axis=1)
        cur_end = np.concatenate(
            [
                0.1 * np.cos(theta1) + 0.4 * np.cos(theta1) * np.cos(theta2),
                0.1 * np.sin(theta1) + 0.4 * np.sin(theta1) * np.cos(theta2) - 0.188,
                -0.4 * np.sin(theta2),
            ],
            axis=1,
        )

        for length, hinge, roll in [(0.321, theta4, theta3), (0.16828, theta6, theta5)]:
            perp_all_axis = np.cross(rot_axis, rot_perp_axis)
            x = np.cos(hinge) * rot_axis
            y = np.sin(hinge) * np.sin(roll) * rot_perp_axis
            z = -np.sin(hinge) * np.cos(roll) * perp_all_axis
            new_rot_axis = x + y + z
            new_rot_perp_axis = np.cross(new_rot_axis, rot_axis)
            new_rot_perp_axis[np.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30] = rot_perp_axis[
                np.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30
            ]
            new_rot_perp_axis /= np.linalg.norm(new_rot_perp_axis, axis=1, keepdims=True)
            rot_axis, rot_perp_axis, cur_end = new_rot_axis, new_rot_perp_axis, cur_end + length * new_rot_axis

        return cur_end
