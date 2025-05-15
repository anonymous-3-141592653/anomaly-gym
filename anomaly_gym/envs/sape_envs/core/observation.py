from abc import abstractmethod

import numpy as np
from gymnasium import spaces

from .render import PygameRenderer
from .world import World


class GoalObservationEncoder:
    def __init__(self, world: World, **kwargs) -> None:
        self.world = world

    @property
    def space(self) -> spaces.Space:
        """Return the observation space."""

        if not hasattr(self, "observation_space"):
            obs = self.observe(self.world)
            self.observation_space = spaces.Dict(
                dict(
                    desired_goal=spaces.Box(-np.inf, np.inf, shape=obs["desired_goal"].shape, dtype=np.float32),
                    achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype=np.float32),
                    observation=spaces.Box(-np.inf, np.inf, shape=obs["observation"].shape, dtype=np.float32),
                )
            )
        return self.observation_space

    def observe(self, world: World) -> dict[str, np.ndarray]:
        """Return an observation of the environment state."""

        obs = self._get_observation(world).astype(np.float32)

        return {
            "observation": obs.copy(),
            "achieved_goal": world.goal.state.p_pos - world.agent.state.p_pos,
            "desired_goal": np.zeros(world.dim_p, dtype=np.float32),
        }

    @abstractmethod
    def _get_observation(self, world: World):
        raise NotImplementedError()


class VectorObservation(GoalObservationEncoder):
    def __init__(self, world: World, **kwargs) -> None:
        super().__init__(world, **kwargs)

    def _get_observation(self, world: World) -> np.ndarray:
        rel_obj_pos = [obj.state.p_pos - world.agent.state.p_pos for obj in world.objects]
        rel_haz_pos = [haz.state.p_pos - world.agent.state.p_pos for haz in world.hazards]
        obs = np.concatenate(
            [
                world.agent.state.p_pos,
                world.agent.state.p_vel,
                world.goal.state.p_pos - world.agent.state.p_pos,
                *rel_obj_pos,
                *rel_haz_pos,
            ]
        )
        return obs


class LidarObservation(GoalObservationEncoder):
    def __init__(self, world: World, lidar_num_bins, **kwargs) -> None:
        super().__init__(world, **kwargs)
        self.lidar_num_bins = lidar_num_bins
        self.lidar_exp_gain = 1.0
        self.lidar_max_dist = 1.0
        self.lidar_bin_size = 360 / self.lidar_num_bins
        self.lidar_alias = True

    def _distance(self, a, b) -> np.floating:
        return np.linalg.norm(a - b, axis=-1, keepdims=True)

    def _angle(self, a, b) -> float:
        d = b - a
        rad = np.arctan2(d[1], d[0])
        deg = np.degrees(rad) % 360
        return deg

    def _get_observation(self, world: World) -> np.ndarray:
        object_positions = [obj.state.p_pos for obj in world.objects]
        hazard_positions = [obj.state.p_pos for obj in world.hazards]

        obj_lidar_obs = self._pseudo_lidar_observation(world, object_positions)
        haz_lidar_obs = self._pseudo_lidar_observation(world, hazard_positions)
        obs = np.concatenate(
            [
                world.agent.state.p_pos,
                world.agent.state.p_vel,
                world.goal.state.p_pos - world.agent.state.p_pos,
                obj_lidar_obs,
                haz_lidar_obs,
            ]
        )
        return obs

    def _pseudo_lidar_observation(self, world: World, other_positions: list[np.ndarray]) -> np.ndarray:
        """
        based on: https://github.com/openai/safety-gym/tree/master/safety_gym/envs
        Return a robot-centric lidar observation of a list of positions.

        Lidar is a set of bins around the robot (divided evenly in a circle).
        The detection directions are exclusive and exhaustive for a full 360 view.
        Each bin reads 0 if there are no objects in that direction.
        If there are multiple objects, the distance to the closest one is used.
        Otherwise the bin reads the fraction of the distance towards the robot.

        E.g. if the object is 90% of lidar_max_dist away, the bin will read 0.1,
        and if the object is 10% of lidar_max_dist away, the bin will read 0.9.
        (The reading can be thought of as "closeness" or inverse distance)

        This encoding has some desirable properties:
            - bins read 0 when empty
            - bins smoothly increase as objects get close
            - maximum reading is 1.0 (where the object overlaps the robot)
            - close objects occlude far objects
            - constant size observation with variable numbers of objects
        """

        obs = np.zeros(self.lidar_num_bins)

        agent_pos = world.agent.state.p_pos

        for pos in other_positions:
            dist = self._distance(agent_pos, pos)
            angle = self._angle(agent_pos, pos)
            bin_id = int(angle / self.lidar_bin_size)
            if self.lidar_max_dist is None:
                sensor = np.exp(-self.lidar_exp_gain * dist)
            else:
                sensor = max(0, self.lidar_max_dist - dist) / self.lidar_max_dist
            obs[bin_id] = max(obs[bin_id], sensor)

            # Aliasing
            if self.lidar_alias:
                bin_angle = self.lidar_bin_size * bin_id
                alias = (angle - bin_angle) / self.lidar_bin_size
                assert 0 <= alias <= 1, f"bad alias {alias}, dist {dist}, angle {angle}, bin {bin}"
                bin_plus = (bin_id + 1) % self.lidar_num_bins
                bin_minus = (bin_id - 1) % self.lidar_num_bins
                obs[bin_plus] = max(obs[bin_plus], alias * sensor)
                obs[bin_minus] = max(obs[bin_minus], (1 - alias) * sensor)

        return obs


class ImgObservation(GoalObservationEncoder):
    def __init__(self, world: World, **kwargs) -> None:
        super().__init__(world, **kwargs)
        self._obs_renderer = PygameRenderer(width=64, height=64)

    def _get_observation(self, world):
        obs = self._obs_renderer.render(world)
        return obs
