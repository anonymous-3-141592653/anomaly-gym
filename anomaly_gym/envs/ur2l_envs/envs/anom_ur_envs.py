from pathlib import Path

import mujoco
import numpy as np
import yaml

from anomaly_gym.common.metrics import EXPERT_EP_LENGTHS
from anomaly_gym.envs.ur2l_envs.envs.mujoco_pick_and_place_env import URMujocoPickAndPlaceEnv
from anomaly_gym.envs.ur2l_envs.envs.mujoco_reach_env import URMujocoReachEnv
from anomaly_gym.envs.ur2l_envs.envs.rtde_env import URRtdeReachEnv
from anomaly_gym.envs.ur2l_envs.interfaces import URLimits, URMujocoInterface


class AnomURMixin:
    np_random: np.random.RandomState
    _data: mujoco.MjData
    _model: mujoco.MjModel
    _ur_interface: URMujocoInterface
    limits: URLimits
    type: str
    base_id: str

    def __init__(
        self,
        anomaly_type=None,
        anomaly_param: float | None = None,
        anomaly_strength: str | None = None,
        anomaly_onset: str = "start",
        **kwargs,
    ):
        assert anomaly_onset in ("start", "random"), "Invalid anomaly start"
        assert anomaly_type in self.anomaly_types or anomaly_type is None, "Invalid anomaly type: " + anomaly_type
        if anomaly_param is None:
            assert anomaly_strength is not None, "anomaly_param or anomaly_strength must be provided"
            anomaly_param = self._get_anomaly_param(anomaly_type, anomaly_strength)
        else:
            assert anomaly_strength is None, "Either anomaly_param or anomaly_strength must be provided, not both"

        self.anomaly_onset = anomaly_onset
        self.anomaly_type = anomaly_type
        self.anomaly_param = anomaly_param
        self.anomaly_strength = anomaly_strength

        super().__init__(**kwargs)
        self.init_body_mass = self._model.body_mass.copy()
        self.init_dof_frictionloss = self._model.dof_frictionloss.copy()
        self.init_dof_damping = self._model.dof_damping.copy()
        self.init_robot_speed = self._action_setter.clip_delta

    @property
    def anomaly_types(
        self,
    ):
        atypes = [
            "action_offset",
            "action_factor",
            "obs_offset",
            "obs_factor",
            "moving_goal",
        ]

        if self.type == "rtde":
            atypes += ["control_latency", "control_smoothing"]
        if self.type == "mujoco":
            atypes += ["action_noise", "obs_noise", "robot_speed", "robot_friction"]

        return atypes

    def _get_anomaly_param(self, anomaly_type, anomaly_strength) -> float:
        anomaly_params_path = Path(__file__).parents[1] / "cfgs/anomaly_parameters.yaml"
        with open(anomaly_params_path) as file:
            anomaly_params = yaml.safe_load(file)
        return anomaly_params[self.base_id][anomaly_type][anomaly_strength]["param"]

    @property
    def base_env_id(self):
        return self.base_id  # type: ignore

    def _reset_parameters(self):
        self._rtde_time = 0.005
        self._rtde_lookahead_time = 0.06
        self._model.body_mass[:] = self.init_body_mass[:]

        if self.anomaly_type == "robot_force":
            body_id = self._ur_interface._model_names.body_name2id["ur3/end_effector"]
            self._data.xfrc_applied[body_id][:] *= 0

        if self.anomaly_type == "robot_friction":
            self._model.dof_damping[:] = self.init_dof_damping[:]
            self._model.dof_frictionloss[:] = self.init_dof_frictionloss[:]

        if self.anomaly_type == "robot_speed":
            self._action_setter.clip_delta = self.init_robot_speed

    def _apply_anomaly_params(self):
        if self.anomaly_type == "control_latency":
            self._rtde_time = self.anomaly_param

        if self.anomaly_type == "control_smoothing":
            self._rtde_lookahead_time = self.anomaly_param

        if self.anomaly_type == "box_mass":
            body_id = self._ur_interface._model_names.body_name2id["block"]
            anom_body_mass = self.init_body_mass.copy()
            anom_body_mass[body_id] *= self.anomaly_param
            self._model.body_mass[:] = anom_body_mass[:]

        if self.anomaly_type == "robot_force":
            body_id = self._ur_interface._model_names.body_name2id["ur3/end_effector"]
            f_vec = np.array([self.anomaly_param, self.anomaly_param, self.anomaly_param, 0, 0, 0])
            self._data.xfrc_applied[body_id][:] = f_vec[:]

        if self.anomaly_type == "robot_friction":
            self._model.dof_damping[:] = self.init_dof_damping[:] * self.anomaly_param
            self._model.dof_frictionloss[:] = self.init_dof_frictionloss[:] + self.anomaly_param

        if self.anomaly_type == "robot_speed":
            self._action_setter.clip_delta = self.anomaly_param

    def _anomaly_step(self, action):
        if self.step_ctr == self._anomaly_onset:
            self._apply_anomaly_params()

        if self.anomaly_type == "moving_goal":
            if not hasattr(self, "goal_direction"):
                self.goal_direction = 1

            goal_pos = self._ur_interface.goal_position
            if not self.limits.goal_min[0] < goal_pos[0] < self.limits.goal_max[0]:
                self.goal_direction = -self.goal_direction

            new_goal_pos = goal_pos + (self.anomaly_param * np.array([1, 0, 0]) * self.goal_direction)
            self._ur_interface.set_goal(new_goal_pos, self._ur_interface.goal_quaternion)

        if self.anomaly_type == "action_factor":
            action = action * self.anomaly_param

        if self.anomaly_type == "action_offset":
            action = action + self.anomaly_param

        if self.anomaly_type == "action_noise":
            action = self.np_random.normal(action, self.anomaly_param)  # type: ignore

        obs, reward, term, trunc, info = super().step(action)

        if self.anomaly_type == "obs_offset":
            obs = {obs_key: obs_val + self.anomaly_param for obs_key, obs_val in obs.items()}
        if self.anomaly_type == "obs_factor":
            obs = {obs_key: obs_val * self.anomaly_param for obs_key, obs_val in obs.items()}
        if self.anomaly_type == "obs_noise":
            obs = {obs_key: self.np_random.normal(obs_val, self.anomaly_param) for obs_key, obs_val in obs.items()}  # type: ignore

        obs = {k: obs[k].astype(np.float32) for k in obs}

        return obs, reward, term, trunc, info

    def step(self, action):
        if self.step_ctr < self._anomaly_onset:
            obs, reward, term, trunc, info = super().step(action)
            info["is_anomaly"] = False

        else:
            obs, reward, term, trunc, info = self._anomaly_step(action)
            info["is_anomaly"] = True

        self.step_ctr += 1
        return obs, reward, term, trunc, info

    def reset(self, **kwargs):
        self._reset_parameters()
        ret = super().reset(**kwargs)
        self.step_ctr = 0
        if self.anomaly_onset == "random":
            self._anomaly_onset = self.np_random.integers(2, EXPERT_EP_LENGTHS[self.base_id])
        else:
            self._anomaly_onset = 0
        return ret


class Anom_URMujocoReachEnv(AnomURMixin, URMujocoReachEnv):
    base_id = "URMujoco-Reach"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Anom_URMujocoPickAndPlaceEnv(AnomURMixin, URMujocoPickAndPlaceEnv):
    base_id = "URMujoco-PnP"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Anom_URRtdeReachEnv(AnomURMixin, URRtdeReachEnv):
    base_id = "URRtde-Reach"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
