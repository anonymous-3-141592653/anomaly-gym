import os
import random
import socket
import subprocess
import time

import carla
import dotenv
import gymnasium
import numpy as np

from anomaly_gym.envs.carla_envs.core.action import ContinuousAction
from anomaly_gym.envs.carla_envs.core.observation import CarlaLaneKeepObs
from anomaly_gym.envs.carla_envs.core.sensors import CarlaSensorInterface
from anomaly_gym.envs.carla_envs.core.utils import angle_between, vec2arr

dotenv.load_dotenv(override=True)



def get_free_port():
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def get_rpc_port():
    rpc_port = os.environ.get("CARLA_RPC_PORT")
    if rpc_port is None:
        rpc_port = get_free_port()
        print("[CARLA ENV] Env variable CARLA_RPC_PORT not set, using open port:", rpc_port)
    else:
        rpc_port = int(rpc_port)

    print("[CARLA ENV] Attempt to connect to: ", rpc_port)
    return rpc_port


def start_carla_server(port):
    carla_cmd = os.environ.get("CARLA_SERVER_CMD")
    if carla_cmd is None:
        print("[CARLA ENV] Env variable CARLA_SERVER_CMD not set, using default command: ")
        carla_cmd = os.environ["HOME"] + f"/carla/CarlaUE4.sh"
        print(carla_cmd)

    carla_cmd += f" -carla-rpc-port={port} -RenderOffScreen -nosound"

    proc = subprocess.Popen(carla_cmd, shell=True)
    time.sleep(10)
    return proc


def stop_carla_server(proc):
    if proc is not None:
        proc.terminate()
        proc.wait()
        print("[CARLA ENV] Carla server stopped.")
    else:
        print("[CARLA ENV] No Carla server process to stop.")


class CarlaLaneKeepEnv(gymnasium.Env):
    MAX_DISTANCE = 100
    MAX_SPEED = 100 / 3.6
    metadata = {"render_modes": [None, "human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        n_exo_vehicles=100,
        fps=20,
        map_name="Town04",
        weather_conditions="sunny",
        road_conditions="normal",
        render_mode=None,
        start_server_process=True,
    ):
        self.render_mode = render_mode
        self.delta_seconds = float(1 / fps)
        self.cam_size = 256
        self.n_exo_vehicles = n_exo_vehicles
        self.target_speed = 50 / 3.6  # km/h
        self.road_conditions = road_conditions

        # init client
        self.rpc_port = get_rpc_port()
        if start_server_process:
            self.server_proc = start_carla_server(self.rpc_port)

        self.client = carla.Client("localhost", self.rpc_port)
        self.client.set_timeout(60.0)

        # init world
        self.world = self.client.get_world()
        if map_name not in self.world.get_map().name:
            self.client.load_world(map_name)
        self.map = self.world.get_map()
        no_rendering_mode = True if render_mode is None else False
        self.world.apply_settings(
            carla.WorldSettings(
                synchronous_mode=True, fixed_delta_seconds=self.delta_seconds, no_rendering_mode=no_rendering_mode
            )
        )
        self.client.reload_world(False)
        self._set_weather_conditions(weather_conditions)

        # init vehicles
        self.bp_lib = self.world.get_blueprint_library()
        self.ego_vehicle_id = "vehicle.ford.crown"
        car_bps = [v for v in self.bp_lib.filter("vehicle") if v.get_attribute("base_type").as_str() == "car"]
        self.exo_bps = [v for v in car_bps if v.id != self.ego_vehicle_id]
        self.ego_bps = self.bp_lib.filter(self.ego_vehicle_id)
        self.spawn_points = self._get_highway_spawn_points()
        self.tm = self.client.get_trafficmanager(self.rpc_port + 50)
        self.tm.set_synchronous_mode(True)
        self.tm.set_hybrid_physics_mode(True)
        self.tm.set_hybrid_physics_radius(100.0)

        # init sensors
        sensors = ["collision", "lane_invasion"]
        if render_mode == "rgb_array":
            sensors.append("camera_first_person")

        if render_mode == "human":
            sensors.append("camera_third_person")
            # ["camera_third_person", "lidar", "imu", "gnss"]

        self.sensor_interface = CarlaSensorInterface(
            world=self.world,
            cam_size=self.cam_size,
            sensors=sensors,
        )

        # init spaces
        self.observation_encoder = CarlaLaneKeepObs(self)
        self.action_encoder = ContinuousAction()
        self.observation_space = self.observation_encoder.space
        self.action_space = self.action_encoder.space

    def _set_weather_conditions(self, conditions="sunny"):
        """note: weather does not control vehilce physics"""
        if conditions == "sunny":
            weather = carla.WeatherParameters(cloudiness=0.0, precipitation=00.0, sun_altitude_angle=90.0)

        elif conditions == "cloudy":
            weather = carla.WeatherParameters(cloudiness=40.0, precipitation=0.0, sun_altitude_angle=70.0)

        elif conditions == "rainy":
            weather = carla.WeatherParameters(
                cloudiness=40.0, precipitation=100.0, sun_altitude_angle=70.0, wetness=100, precipitation_deposits=100
            )

        else:
            raise NotImplementedError

        self.world.set_weather(weather)

    def _set_road_conditions(self, conditions="normal", param=0.1):
        if conditions == "normal":
            pass
        elif conditions == "slippery":
            physics_control = self.ego_vehicle.get_physics_control()
            wheels = physics_control.wheels
            for w in wheels:
                w.tire_friction *= param
            physics_control.wheels = wheels
            self.ego_vehicle.apply_physics_control(physics_control)
        else:
            raise NotImplementedError

    def _get_highway_spawn_points(self):
        waypoints = self.map.generate_waypoints(30.0)
        highway_roads = set([w.road_id for w in waypoints if w.lane_id == 3 and not w.is_junction])
        transforms = []
        for w in waypoints:
            if w.road_id in highway_roads and abs(w.lane_id) < 4:
                t = w.transform
                t.location.z += 0.1
                transforms.append(t)
        return transforms

    def _sample_spawn_point(self):
        while True:
            transform = self.np_random.choice(self.spawn_points)
            if transform not in self._current_spawn_points:
                self._current_spawn_points.add(transform)
                return transform

    def _add_exo_vehicles(self, n=1):
        self.exo_vehicles = []
        batch = []
        for i in range(n):
            blueprint = self.np_random.choice(self.exo_bps)
            blueprint.set_attribute("role_name", "autopilot")
            batch.append(
                carla.command.SpawnActor(blueprint, self._sample_spawn_point()).then(
                    carla.command.SetAutopilot(carla.command.FutureActor, True, self.tm.get_port())
                )
            )
        for response in self.client.apply_batch_sync(batch, True):
            self.exo_vehicles.append(response.actor_id)

        # print(f"spawned {len(self.exo_vehicles)} exo vehicles")

    def _add_ego_vehicle(self):
        blueprint = self.np_random.choice(self.ego_bps)
        blueprint.set_attribute("role_name", "ego")  # or hero
        transform = self._sample_spawn_point()
        self.ego_vehicle = self.world.spawn_actor(blueprint=blueprint, transform=transform)
        self.ego_vehicle.set_autopilot(enabled=False, tm_port=self.tm.get_port())

    def _set_spectator(self, transform=None):
        # do only if self.rendering_mode = human
        spectator = self.world.get_spectator()
        if transform is None:
            transform = carla.Transform(
                self.ego_vehicle.get_transform().transform(carla.Location(x=-4, z=2.5)),
                self.ego_vehicle.get_transform().rotation,
            )
        spectator.set_transform(transform)

    def _destroy_vehicles(self):
        actor_list = self.world.get_actors()
        for vehicle in actor_list.filter("*vehicle*"):
            vehicle.destroy()

    def _reset_vehicles(self):
        self._destroy_vehicles()
        self._current_spawn_points = set()

        self._add_ego_vehicle()
        self._add_exo_vehicles(n=self.n_exo_vehicles)

    def _reset_sensors(self):
        self.sensor_interface.reset(self.world, self.ego_vehicle)

    def _get_target(self, vehicle_loc, vehicle_wp, target_lane_id):
        road_wp = self.map.get_waypoint(vehicle_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        target_road_id = road_wp.road_id
        target_wp = self.map.get_waypoint_xodr(target_road_id, target_lane_id, vehicle_wp.s)
        if target_wp is None:
            target_wp = self.map.get_waypoint_xodr(target_road_id, target_lane_id, vehicle_wp.s + 0.02)
        if target_wp is None:
            target_wp = self.map.get_waypoint_xodr(target_road_id, target_lane_id, vehicle_wp.s - 0.02)
        if target_wp is None:
            raise ValueError("state error")
        else:
            target_loc = target_wp.transform.location
            target_loc_xy = np.array([target_loc.x, target_loc.y])
            return target_wp, target_loc_xy

    def _get_highway_vector(self, target_wp):
        # get highway vector
        target_loc = target_wp.transform.location
        target_loc_xy = np.array([target_loc.x, target_loc.y])
        next_waypoint = target_wp.next(0.1)[0]
        next_loc = next_waypoint.transform.location
        next_loc_xy = np.array([next_loc.x, next_loc.y])
        highway_vector = next_loc_xy - target_loc_xy
        return highway_vector

    def _get_ego_state(self):
        ego_loc = self.ego_vehicle.get_location()
        ego_loc_xy = vec2arr(ego_loc)[:2]
        ego_vel_xy = vec2arr(self.ego_vehicle.get_velocity())[:2]
        ego_speed = np.linalg.norm(ego_vel_xy)  # in m/s
        ego_accel_xy = vec2arr(self.ego_vehicle.get_acceleration())[:2]
        ego_accel = np.linalg.norm(ego_accel_xy)
        ego_wp = self.map.get_waypoint(ego_loc)
        ego_vec = self.ego_vehicle.get_transform().get_forward_vector()
        ego_vec_xy = vec2arr(ego_vec)[:2]

        target_wp, target_loc_xy = self._get_target(ego_loc, ego_wp, self.target_lane_id)
        highway_vector = self._get_highway_vector(target_wp)

        dist_to_lane_center = np.linalg.norm(ego_loc_xy - target_loc_xy)
        heading_angle = angle_between(ego_vec_xy, highway_vector)
        if np.isnan(heading_angle):
            heading_angle = 0.0

        # cross-product > 0 :right, < 0 :left
        angle_sign = np.sign(np.cross(highway_vector, ego_vec_xy))
        dist_sign = np.sin(np.cross(highway_vector, ego_loc_xy - target_loc_xy))

        heading_angle = heading_angle * angle_sign
        dist_to_lane_center = dist_to_lane_center * dist_sign

        state_dict = {
            "location": ego_loc_xy,
            "speed": ego_speed,
            "target_speed": self.target_speed,
            "accel": ego_accel,
            "heading": heading_angle,
            "dist_to_lane_center": dist_to_lane_center,
            "target_lane_id": self.target_lane_id,
            "lane_id": ego_wp.lane_id,
            "road_id": ego_wp.road_id,
            "collision": self.sensor_interface.data["collision"],
            "lane_invasion": self.sensor_interface.data["lane_invasion"],
            "act_accel": self.action_encoder.accel,
            "act_steer": self.action_encoder.steer,
        }
        return state_dict

    def _set_action(self, action):
        self.action_encoder.set_action(action, self.ego_vehicle)

    def _carla_step(self):
        self.world.tick()
        self._set_spectator()

    @property
    def vehicle_list(self):
        return list(self.world.get_actors().filter("*vehicle*"))

    def compute_reward(self, action, ego_state):
        term, term_cause = self.compute_terminated(ego_state)
        if term:
            return -100
        else:
            velocity_reward = -np.abs((ego_state["speed"] - self.target_speed) / self.MAX_SPEED)
            heading_reward = -0.1 * np.abs(ego_state["heading"])
            distance_reward = -0.1 * np.abs(ego_state["dist_to_lane_center"])
            action_reward = -0.01 * np.linalg.norm(action)
            return velocity_reward + heading_reward + distance_reward + action_reward

    def compute_terminated(self, ego_state):
        if ego_state["collision"]:
            return True, "collision"
        elif abs(ego_state["heading"]) > np.pi / 4:
            return True, "heading>45deg"
        elif abs(ego_state["dist_to_lane_center"]) > 1.5:
            return True, "dist_to_lane_center>1.5m"
        else:
            return False, "nc"

    def compute_truncated(self):
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._reset_vehicles()
        self._reset_sensors()
        self._set_spectator()
        self._set_road_conditions(self.road_conditions)
        self.world.tick()

        # this needs to happen after world tick
        ego_loc = self.ego_vehicle.get_location()
        ego_wp = self.map.get_waypoint(ego_loc, project_to_road=True)
        self.target_lane_id = ego_wp.lane_id

        self.action_encoder.reset()
        self.episode_start = time.time()

        sensor_data = self.sensor_interface.step()
        ego_state = self._get_ego_state()

        observation = self.observation_encoder.observe(ego_state, sensor_data)
        term, term_cause = self.compute_terminated(ego_state)
        self._info = {**ego_state, "term_cause": term_cause, "is_success": False, "is_critical": False}
        return observation, self._info

    def step(self, action):
        try:
            self._set_action(action)
            self._carla_step()

            sensor_data = self.sensor_interface.step()
            ego_state = self._get_ego_state()

            observation = self.observation_encoder.observe(ego_state, sensor_data)
            term, term_cause = self.compute_terminated(ego_state)
            trunc = self.compute_truncated()
            reward = self.compute_reward(action, ego_state)
            self._info = {**ego_state, "term_cause": term_cause, "is_success": not term, "is_critical": term}
        except Exception as ex:
            reward = 0
            term = True
            trunc = False
            observation = self.observation_space.sample()
            # reuse previous info since this needs to have the same entries with the same shapes as the initial info
            self._info.update({"term_cause": ex.__str__()})

        return observation, reward, term, trunc, self._info

    def render(self):
        rgb_array = self.sensor_interface.data["camera"]
        return rgb_array

    def close(self):
        self.sensor_interface._destroy_sensors(self.world)
        if hasattr(self, "server_proc"):
            stop_carla_server(self.server_proc)
        time.sleep(1)
        return super().close()
