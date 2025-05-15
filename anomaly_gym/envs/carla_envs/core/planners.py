import os
import sys
from abc import abstractmethod

import numpy as np

from .agents.navigation.behavior_agent import BehaviorAgent
from .agents.navigation.local_planner import LocalPlanner, RoadOption


class CarlaPlannerAgent:
    def __init__(self, env):
        self.env = env

    def predict(self, obs, **kwargs):
        control = self.planner.run_step()
        action = np.array([control.throttle - control.brake, control.steer], dtype=np.float32)
        return action, None

    @property
    def planner(self):
        if not hasattr(self, "_planner"):
            self._planner = self._reset_planner()

        if self._planner._vehicle is not self.env.ego_vehicle:
            self._planner = self._reset_planner()

        return self._planner

    @abstractmethod
    def _reset_planner(self):
        pass


class CarlaLocalPlannerAgent(CarlaPlannerAgent):
    def __init__(self, env, target_speed):
        self.target_speed = target_speed
        super().__init__(env)

    def _reset_planner(self):
        return LocalPlanner(
            vehicle=self.env.ego_vehicle, map_inst=self.env.map, opt_dict={"target_speed": self.target_speed}
        )


class CarlaBehaviorAgent(CarlaPlannerAgent):
    def __init__(self, env, behavior="normal") -> None:
        assert behavior in {"cautious", "normal", "aggressive"}
        self.behavior = behavior
        super().__init__(env)

    def _reset_planner(self):
        return BehaviorAgent(vehicle=self.env.ego_vehicle, map_inst=self.env.map, behavior=self.behavior)


class GhostPlanner(LocalPlanner):
    def _compute_next_waypoints(self, k=1):
        available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
        k = min(available_entries, k)

        for _ in range(k):
            last_waypoint = self._waypoints_queue[-1][0]
            next_waypoints = list(last_waypoint.previous(self._sampling_radius))

            if len(next_waypoints) == 0:
                break
            else:
                # only one option available ==> lanefollowing
                next_waypoint = next_waypoints[0]
                road_option = RoadOption.LANEFOLLOW

        self._waypoints_queue.append((next_waypoint, road_option))


class CarlaGhostPlannerAgent(CarlaPlannerAgent):
    def __init__(self, env, target_speed):
        self.target_speed = target_speed
        super().__init__(env)

    def _reset_planner(self):
        return GhostPlanner(
            vehicle=self.env.ego_vehicle, map_inst=self.env.map, opt_dict={"target_speed": self.target_speed}
        )
