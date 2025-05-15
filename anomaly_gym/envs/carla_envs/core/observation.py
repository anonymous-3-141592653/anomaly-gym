import numpy as np
from gymnasium import spaces

from anomaly_gym.common.goal_obs import GoalObs
from anomaly_gym.envs.carla_envs.core.utils import distance_along_road, veh_speed


class CarlaImageObs(GoalObs):
    def __init__(self, env, cam_size) -> None:
        self.env = env
        self.obs_shape = (cam_size, cam_size, 3)
        self.goal_shape = (3,)

    def _get_observation(self, env, sensor_data) -> np.ndarray:
        return sensor_data.get("camera", np.zeros(self.obs_shape))


class CarlaLaneKeepObs:
    def __init__(self, env, **kwargs) -> None:
        self.env = env
        self.obs_shape = (9,)
        self.space = spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_shape)

    def observe(self, ego_state, *args, **kwargs) -> np.ndarray:
        return self._get_observation(ego_state)

    def _get_observation(self, ego_state) -> np.ndarray:
        distance_ahead, vehicle_ahead = distance_along_road(
            self.env.ego_vehicle,
            self.env.map,
            self.env.vehicle_list,
            max_distance=self.env.MAX_DISTANCE,
        )
        if vehicle_ahead is not None:
            # delta_v = np.linalg.norm(vec2arr(vehicle_ahead.get_velocity() - self.env.ego_vehicle.get_velocity()))
            delta_vel_ahead = veh_speed(vehicle_ahead) - ego_state["speed"]

        else:
            distance_ahead = self.env.MAX_DISTANCE
            delta_vel_ahead = 0

        obs = {
            "ego_speed": ego_state["speed"] / self.env.MAX_SPEED,
            "target_speed": ego_state["target_speed"] / self.env.MAX_SPEED,
            "ego_accell": ego_state["accel"],
            "ego_heading": ego_state["heading"],
            "dist_to_lane_center": ego_state["dist_to_lane_center"],
            "dist_ahead": distance_ahead,
            "delta_vel_ahead": delta_vel_ahead,
            "act_accel": ego_state["act_accel"],
            "act_steer": ego_state["act_steer"],
        }

        return np.stack([v for v in obs.values()], dtype=np.float32)
