from pathlib import Path

import numpy as np
import yaml
from mujoco._structs import MjData, MjModel

from anomaly_gym.common.metrics import EXPERT_EP_LENGTHS

from .cartpole_swingup import MujocoCartpoleSwingupEnv
from .half_cheetah import MujocoHalfCheetahEnv
from .reacher3D import MujocoReacher3DEnv


class AnomalyMixin:
    task_id: str
    model: MjModel
    data: MjData
    np_random: np.random.Generator
    anom_force_body_name: str
    anom_force_vec: np.ndarray
    anomaly_types = frozenset(
        (
            "robot_mass",
            "robot_force",
            "robot_friction",
            "action_offset",
            "action_factor",
            "action_noise",
            "obs_offset",
            "obs_factor",
            "obs_noise",
        )
    )

    def __init__(
        self,
        anomaly_type=None,
        anomaly_param: float | None = None,
        anomaly_strength: str | None = None,
        anomaly_onset: str = "random",
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert anomaly_onset in ("start", "random"), "Invalid anomaly start"
        assert anomaly_type in self.anomaly_types or anomaly_type is None, "Invalid anomaly type"
        if anomaly_param is None:
            assert anomaly_strength is not None, "anomaly_param or anomaly_strength must be provided"
            anomaly_param = self._get_anomaly_param(anomaly_type, anomaly_strength)
        else:
            assert anomaly_strength is None, "Either anomaly_param or anomaly_strength must be provided, not both"

        self.anomaly_type = anomaly_type
        self.anomaly_param = anomaly_param
        self.anomaly_onset = anomaly_onset
        self.anomaly_strength = anomaly_strength

        # body_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(self.model.nbody)]
        # geom_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(self.model.ngeom)]
        self.init_body_mass = self.model.body_mass.copy()
        self.init_dof_frictionloss = self.model.dof_frictionloss.copy()

    def _get_anomaly_param(self, anomaly_type, anomaly_strength) -> float:
        anomaly_params_path = Path(__file__).parents[1] / "cfgs/anomaly_parameters.yaml"
        with open(anomaly_params_path) as file:
            anomaly_params = yaml.safe_load(file)
        return anomaly_params[self.task_id][anomaly_type][anomaly_strength]["param"]

    @property
    def base_env_id(self):
        return self.spec.id.replace("Anom_", "")  # type: ignore

    def _reset_parameters(self):
        if self.anomaly_type == "robot_mass":
            self.model.body_mass[:] = self.init_body_mass[:]

        if self.anomaly_type == "robot_force":
            body_id = self.model.body(self.anom_force_body_name).id
            self.data.xfrc_applied[body_id][:] = np.zeros_like(self.anom_force_vec)

        if self.anomaly_type == "robot_friction":
            self.model.dof_frictionloss[:] = self.init_dof_frictionloss[:]

    def _apply_anomaly_params(self):
        if self.anomaly_type == "robot_mass":
            self.model.body_mass[:] = self.init_body_mass[:] * self.anomaly_param

        if self.anomaly_type == "robot_force":
            body_id = self.model.body(self.anom_force_body_name).id
            self.data.xfrc_applied[body_id][:] = self.anom_force_vec[:] * self.anomaly_param

        if self.anomaly_type == "robot_friction":
            self.model.dof_frictionloss[:] = self.init_dof_frictionloss[:] + self.anomaly_param

    def _anomaly_step(self, action):
        if self.step_ctr == self._anomaly_onset:
            self._apply_anomaly_params()

        if self.anomaly_type == "action_offset":
            action = action + self.anomaly_param
        if self.anomaly_type == "action_factor":
            action = action * self.anomaly_param
        if self.anomaly_type == "action_noise":
            action = self.np_random.normal(action, self.anomaly_param)

        obs, reward, term, trunc, info = super().step(action)

        if self.anomaly_type == "obs_offset":
            obs = obs + self.anomaly_param
        if self.anomaly_type == "obs_factor":
            obs = obs * self.anomaly_param
        if self.anomaly_type == "obs_noise":
            obs = self.np_random.normal(obs, self.anomaly_param)

        obs = obs.astype(np.float32)

        return obs, reward, term, trunc, info

    def step(self, action):
        if self.step_ctr < self._anomaly_onset:
            obs, reward, term, trunc, info = super().step(action)  # type: ignore
            info["is_anomaly"] = False

        else:
            obs, reward, term, trunc, info = self._anomaly_step(action)
            info["is_anomaly"] = True

        self.step_ctr += 1
        return obs, reward, term, trunc, info

    def reset(self, **kwargs):
        self.step_ctr = 0
        if self.anomaly_onset == "random":
            self._anomaly_onset = self.np_random.integers(2, EXPERT_EP_LENGTHS[self.base_env_id])
        else:
            self._anomaly_onset = 0
        self._reset_parameters()
        return super().reset(**kwargs)  # type: ignore


class Anom_MujocoCartpoleSwingupEnv(AnomalyMixin, MujocoCartpoleSwingupEnv):
    task_id: str = "CartpoleSwingup"
    anom_force_body_name = "pole"
    anom_force_vec = np.array([-1, 0, 0, 0, 0, 0])


class Anom_MujocoHalfCheetahEnv(AnomalyMixin, MujocoHalfCheetahEnv):
    task_id: str = "HalfCheetah"
    anom_force_body_name = "ffoot"
    anom_force_vec = np.array([-1, 0, 0, 0, 0, 0])


class Anom_MujocoReacher3DEnv(AnomalyMixin, MujocoReacher3DEnv):
    task_id: str = "Reacher3D"
    anom_force_body_name = "r_forearm_link"
    anom_force_vec = np.array([0, 0, -1, 0, 0, 0])
