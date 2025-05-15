from typing import Any

import carla
import gymnasium
import numpy as np


class ContinuousAction:

    def __init__(self, **kwargs) -> None:
        self.reset()

    def reset(self):
        self.accel = 0
        self.steer = 0

    def set_action(self, action, vehicle):

        self.accel = np.clip(self.accel + (0.1 * action[0]), -1, 1)
        self.steer = np.clip(self.steer + (0.1 * action[1]), -1, 1)
        throttle = np.clip(self.accel, 0, 1)
        brake = -np.clip(self.accel, -1, 0)

        vehicle_control = carla.VehicleControl(
            throttle=float(throttle),  # [0,1]
            steer=float(self.steer),  # [-1,1]
            brake=float(brake),  # [0,1]
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
        )
        vehicle.apply_control(vehicle_control)

    @property
    def space(self):
        if not hasattr(self, "_action_space"):
            high = np.ones(2, dtype=np.float32)
            self._action_space = gymnasium.spaces.Box(-high, high)
        return self._action_space
