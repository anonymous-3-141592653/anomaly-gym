import queue

import carla
import numpy as np


def generic_sensor_callback(sensor_data, sensor_queue, sensor_name):
    "https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/sensor_synchronization.py"
    sensor_queue.put((sensor_name, sensor_data.frame, np.array(sensor_data.raw_data)))


def camera_sensor_callback(sensor_data, sensor_queue, sensor_name, cam_size):
    "https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/sensor_synchronization.py"
    rgb_img = transform_img(np.array(sensor_data.raw_data), cam_size, cam_size)
    sensor_queue.put((sensor_name, sensor_data.frame, rgb_img))


def gnss_sensor_callback(sensor_data, sensor_queue, sensor_name):
    gnss_dict = {"lat": sensor_data.latitude, "long": sensor_data.longitude}
    sensor_queue.put((sensor_name, sensor_data.frame, gnss_dict))


def imu_sensor_callback(sensor_data, sensor_queue, sensor_name):
    imu_dict = {
        "gyro_xyz": np.array([sensor_data.gyroscope.x, sensor_data.gyroscope.y, sensor_data.gyroscope.z]),
        "accel_xyz": np.array([sensor_data.accelerometer.x, sensor_data.accelerometer.y, sensor_data.accelerometer.z]),
        "compass": sensor_data.compass,
    }
    sensor_queue.put((sensor_name, sensor_data.frame, imu_dict))


def event_sensor_callback(sensor_data, event_buffer, sensor_name):
    "https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/sensor_synchronization.py"
    event_buffer[sensor_name] = sensor_data.frame


def transform_img(bgra_img, image_h, image_w):
    bgra = bgra_img.reshape(image_h, image_w, 4)  # BGRA format
    bgr = bgra[:, :, :3]  # BGR format (h x w x 3)
    rgb_img = np.flip(bgr, axis=2)  # RGB format (h x w x 3)
    return rgb_img


class CarlaSensorInterface:
    def __init__(self, world, sensors=None, cam_size=None) -> None:
        self.bp_lib = world.get_blueprint_library()
        self.cam_size = cam_size
        self.sensors = sensors if sensors is not None else []
        self._event_sensors = []  # sensors that only send data at specific events
        self._stream_sensors = []  # sensors that continuously send data in each timestep

    def reset(self, world, ego_vehicle):
        self._destroy_sensors(world)
        self._init_sensors(world, ego_vehicle)

    def _destroy_sensors(self, world):
        while len(self._stream_sensors) > 0:
            s = self._stream_sensors.pop()
            if s.is_listening:
                s.stop()
            s.destroy()

        while len(self._event_sensors) > 0:
            s = self._event_sensors.pop()
            if s.is_listening:
                s.stop()
            s.destroy()

        self.stream_queue = queue.Queue()
        self.event_buffer = {}
        world.tick()

    def _init_sensors(self, world, ego_vehicle):
        if "camera_first_person" in self.sensors:
            cam_bp = self.bp_lib.find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", str(self.cam_size))
            cam_bp.set_attribute("image_size_y", str(self.cam_size))
            cam_tranform = carla.Transform(carla.Location(x=0.5, z=1.5))
            camera = world.spawn_actor(cam_bp, cam_tranform, attach_to=ego_vehicle)
            camera.listen(lambda data: camera_sensor_callback(data, self.stream_queue, "camera", self.cam_size))
            self._stream_sensors.append(camera)

        if "camera_third_person" in self.sensors:
            cam_bp = self.bp_lib.find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", str(self.cam_size))
            cam_bp.set_attribute("image_size_y", str(self.cam_size))
            cam_tranform = carla.Transform(carla.Location(x=-5.5, z=3))
            camera = world.spawn_actor(cam_bp, cam_tranform, attach_to=ego_vehicle)
            camera.listen(lambda data: camera_sensor_callback(data, self.stream_queue, "camera", self.cam_size))
            self._stream_sensors.append(camera)

        if "lidar" in self.sensors:
            lidar_bp = self.bp_lib.find("sensor.lidar.ray_cast_semantic")
            lidar_bp.set_attribute("channels", "32")
            lidar_bp.set_attribute("range", "100")
            lidar_bp.set_attribute("rotation_frequency", "20")
            lidar_bp.set_attribute("lower_fov", "-30")
            lidar_bp.set_attribute("upper_fov", "30")
            lidar_bp.set_attribute("points_per_second", "1000")
            lidar_bp.set_attribute("horizontal_fov", "360")
            lidar_transform = carla.Transform(carla.Location(x=0.0, z=5.0))
            lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego_vehicle)
            lidar.listen(lambda data: generic_sensor_callback(data, self.stream_queue, "lidar"))
            self._stream_sensors.append(lidar)

        if "radar" in self.sensors:
            radar_bp = self.bp_lib.find("sensor.other.radar")
            radar = world.spawn_actor(radar_bp, carla.Transform(), attach_to=ego_vehicle)
            radar.listen(lambda data: generic_sensor_callback(data, self.stream_queue, "radar"))
            self._stream_sensors.append(radar)

        if "gnss" in self.sensors:
            gnss_bp = self.bp_lib.find("sensor.other.gnss")
            gnss_sensor = world.spawn_actor(gnss_bp, carla.Transform(), attach_to=ego_vehicle)
            gnss_sensor.listen(lambda data: gnss_sensor_callback(data, self.stream_queue, "gnss"))
            self._stream_sensors.append(gnss_sensor)

        if "imu" in self.sensors:
            imu_bp = self.bp_lib.find("sensor.other.imu")
            imu_sensor = world.spawn_actor(imu_bp, carla.Transform(), attach_to=ego_vehicle)
            imu_sensor.listen(lambda data: imu_sensor_callback(data, self.stream_queue, "imu"))
            self._stream_sensors.append(imu_sensor)

        if "collision" in self.sensors:
            collision_bp = self.bp_lib.find("sensor.other.collision")
            collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=ego_vehicle)
            collision_sensor.listen(lambda data: event_sensor_callback(data, self.event_buffer, "collision"))
            self._event_sensors.append(collision_sensor)

        if "lane_invasion" in self.sensors:
            lane_inv_bp = self.bp_lib.find("sensor.other.lane_invasion")
            lane_inv_sensor = world.spawn_actor(lane_inv_bp, carla.Transform(), attach_to=ego_vehicle)
            lane_inv_sensor.listen(lambda data: event_sensor_callback(data, self.event_buffer, "lane_invasion"))
            self._event_sensors.append(lane_inv_sensor)

    def step(self):
        try:
            self.data = {}
            frames = set()
            for _ in range(len(self._stream_sensors)):
                s_name, s_frame, s_data = self.stream_queue.get(block=True, timeout=1.0)
                self.data[s_name] = s_data
                frames.add(s_frame)

            assert len(self.data) == len(self._stream_sensors), f"sensors: {self._stream_sensors}, data: {self.data}"
            # assert len(frames) == 1, "sensor asynchrony detected"

            for sensor in self._event_sensors:
                s_name = sensor.type_id.replace("sensor.other.", "")
                self.data[s_name] = s_name in self.event_buffer

            for e in self.event_buffer.values():
                frames.add(e)

            # assert len(frames) == 1, "sensor asynchrony detected"

            self.event_buffer = {}
            return self.data
        except:
            raise ValueError("sensor error")
