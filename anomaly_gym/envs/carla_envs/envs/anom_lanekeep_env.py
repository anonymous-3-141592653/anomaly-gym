from pathlib import Path

import numpy as np
import yaml

from anomaly_gym.common.metrics import EXPERT_EP_LENGTHS
from anomaly_gym.envs.carla_envs.core.planners import GhostPlanner
from anomaly_gym.envs.carla_envs.envs.lanekeep_env import CarlaLaneKeepEnv


class Anom_CarlaLaneKeepEnv(CarlaLaneKeepEnv):
    anomaly_types = frozenset(
        (
            "brake_fail",
            "steer_fail",
            "slippery_road",
            "ghost_driver",
            "heavy_rain",
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
        super().__init__(**kwargs)

    @property
    def base_env_id(self) -> str:
        return "Carla-LaneKeep"

    def _get_anomaly_param(self, anomaly_type, anomaly_strength) -> float:
        anomaly_params_path = Path(__file__).parents[1] / "cfgs/anomaly_parameters.yaml"
        with open(anomaly_params_path) as file:
            anomaly_params = yaml.safe_load(file)
        return anomaly_params[self.base_env_id][anomaly_type][anomaly_strength]["param"]

    def _spawn_ghost_driver(self):
        while True:
            blueprint = self.np_random.choice(self.exo_bps)
            ego_loc = self.ego_vehicle.get_location()
            ego_wp = self.map.get_waypoint(ego_loc)

            target_wp = ego_wp
            try:
                for i in range(3):
                    target_wp = target_wp.next(50.0)[0]
                transform = target_wp.transform
                transform.rotation.yaw += 180
                self._ghost_vehicle = self.world.spawn_actor(blueprint=blueprint, transform=transform)
                break

            except Exception as e:
                self._reset_vehicles()
                self._reset_sensors()
                self._set_spectator()

        self.world.tick()
        self._ghost_vehicle.set_autopilot(enabled=False, tm_port=self.tm.get_port())
        self._ghost_planner = GhostPlanner(
            vehicle=self._ghost_vehicle, map_inst=self.map, opt_dict={"target_speed": self.target_speed}
        )

    def reset(self, seed=None, options=None):
        self.step_ctr = 0
        if self.anomaly_onset == "random":
            self._anomaly_onset = self.np_random.integers(2, EXPERT_EP_LENGTHS[self.base_env_id])
        else:
            self._anomaly_onset = 0
        obs, info = super().reset(seed, options)

        self._set_weather_conditions("sunny")
        self._set_road_conditions("normal")

        return obs, info

    def _anomaly_step(self, action):
        if self.step_ctr == self._anomaly_onset:
            if self.anomaly_type == "slippery_road":
                self._set_road_conditions("slippery", self.anomaly_param)
            if self.anomaly_type == "heavy_rain":
                self._set_weather_conditions("rainy")
            if self.anomaly_type == "ghost_driver":
                self._spawn_ghost_driver()

        if self.anomaly_type == "brake_fail":
            action[0] = self.anomaly_param * action[0] if action[0] < 0 else action[0]

        if self.anomaly_type == "steer_fail":
            action[1] = self.anomaly_param * action[1]

        if self.anomaly_type == "ghost_driver":
            control = self._ghost_planner.run_step()
            self._ghost_vehicle.apply_control(control)

        if self.anomaly_type == "action_offset":
            action = action + self.anomaly_param
        if self.anomaly_type == "action_factor":
            action = action * self.anomaly_param
        if self.anomaly_type == "action_noise":
            action = self.np_random.normal(action, self.anomaly_param)

        obs, rew, term, trunc, info = super().step(action)

        if self.anomaly_type == "obs_offset":
            obs = obs + self.anomaly_param
        if self.anomaly_type == "obs_factor":
            obs = obs * self.anomaly_param
        if self.anomaly_type == "obs_noise":
            obs = self.np_random.normal(obs, self.anomaly_param)

        obs = obs.astype(np.float32)

        return obs, rew, term, trunc, info

    def step(self, action):
        if self.step_ctr < self._anomaly_onset:
            obs, rew, term, trunc, info = super().step(action)
            info["is_anomaly"] = False

        else:
            obs, rew, term, trunc, info = self._anomaly_step(action)
            info["is_anomaly"] = True

        self.step_ctr += 1
        return obs, rew, term, trunc, info
