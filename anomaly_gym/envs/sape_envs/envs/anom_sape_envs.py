from pathlib import Path

import numpy as np
import yaml
from gymnasium.envs.registration import EnvSpec

from anomaly_gym.common.metrics import EXPERT_EP_LENGTHS
from anomaly_gym.envs.sape_envs.core.object_callbacks import MoveRandom
from anomaly_gym.envs.sape_envs.envs.sape_env import SapeEnv


class Anom_SapeEnv(SapeEnv):
    spec: EnvSpec
    anomaly_types = frozenset(
        (
            "force_agent",
            "moving_objects",
            "mass_agent",
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

        self.anomaly_onset = anomaly_onset
        self.anomaly_type = anomaly_type
        self.anomaly_param = anomaly_param
        self.anomaly_strength = anomaly_strength
        # self._damping = self.world.damping
        self.init_mass = self.world.agent.initial_mass

    def _get_anomaly_param(self, anomaly_type, anomaly_strength) -> float:
        anomaly_params_path = Path(__file__).parents[1] / "cfgs/anomaly_parameters.yaml"
        with open(anomaly_params_path) as file:
            anomaly_params = yaml.safe_load(file)
        return anomaly_params[self.task_id][anomaly_type][anomaly_strength]["param"]

    @property
    def base_env_id(self):
        return self.spec.id.replace("Anom_", "")  # type: ignore

    def _apply_anomaly_params(self):
        if self.anomaly_type == "force_agent":
            p_force = np.zeros((len(self.world.entities), self.world.dim_p))
            p_force[0] = self.anomaly_param  # 1.0
            self.world._external_force_vector = p_force

        elif self.anomaly_type == "force_objects":
            p_force = np.zeros((len(self.world.entities), self.world.dim_p))
            p_force[1:-1] = self.anomaly_param  # 0.5
            self.world._external_force_vector = p_force

        elif self.anomaly_type == "moving_objects":
            for obj in self.world.objects:
                obj.action_callback = MoveRandom(scale=self.anomaly_param)  # 1.0

        elif self.anomaly_type == "mass_agent":
            self.world.agent.initial_mass *= self.anomaly_param

    def _reset_parameters(self):
        self.world.agent.initial_mass = self.init_mass

    def _anomaly_step(self, action):
        if self.step_ctr == self._anomaly_onset:
            self._apply_anomaly_params()

        if self.anomaly_type == "action_factor":
            action = self.anomaly_param * action

        if self.anomaly_type == "action_offset":
            action = self.anomaly_param + action

        if self.anomaly_type == "action_noise":
            action = self.np_random.normal(action, self.anomaly_param)

        obs, reward, term, trunc, info = super().step(action)

        if self.anomaly_type == "obs_offset":
            obs = {obs_key: obs_val + self.anomaly_param for obs_key, obs_val in obs.items()}
        if self.anomaly_type == "obs_factor":
            obs = {obs_key: obs_val * self.anomaly_param for obs_key, obs_val in obs.items()}
        if self.anomaly_type == "obs_noise":
            obs = {obs_key: self.np_random.normal(obs_val, self.anomaly_param) for obs_key, obs_val in obs.items()}

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
        self.step_ctr = 0
        self._reset_parameters()
        if self.anomaly_onset == "random":
            self._anomaly_onset = self.np_random.integers(2, EXPERT_EP_LENGTHS[self.base_env_id])
        else:
            self._anomaly_onset = 0
        return super().reset(**kwargs)
