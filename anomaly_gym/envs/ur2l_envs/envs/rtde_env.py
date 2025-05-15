import logging
import os
import time

import mujoco
import numpy as np
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from anomaly_gym.envs.ur2l_envs.envs.mujoco_reach_env import URMujocoReachEnv
from anomaly_gym.envs.ur2l_envs.interfaces.limits import URLimits
from anomaly_gym.envs.ur2l_envs.interfaces.realsense_interface import RealSenseInterface

logger = logging.getLogger("ur2l_gym.rtde_env")


class RtdeMixin:
    limits: URLimits
    control_type: str
    _data: mujoco.MjData
    _model: mujoco.MjModel

    def _init_rtde(self, rtde_control_freq: float = 25.0, rtde_robot_ip: str | None = None):
        rtde_robot_ip = rtde_robot_ip or os.environ.get("RTDE_ROBOT_IP")
        if rtde_robot_ip is None:
            rtde_robot_ip = "127.0.0.1"
        logger.info(f"RTDE attempt to connect to: {rtde_robot_ip}")
        self._rtde_c = RTDEControlInterface(rtde_robot_ip, frequency=rtde_control_freq)
        self._rtde_r = RTDEReceiveInterface(rtde_robot_ip)

        if self.control_type == "joint":
            self._rtde_velocity = 1.0
            self._rtde_acceleration = 1.5
            self._rtde_time = 0.001
            self._rtde_lookahead_time = 0.03
            self._rtde_gain = 1000

        elif self.control_type == "mocap":
            self._rtde_velocity = 1.5
            self._rtde_acceleration = 2.5
            self._rtde_time = 0.005
            self._rtde_lookahead_time = 0.06
            self._rtde_gain = 100
            self._action_factor = 0.009
        else:
            raise ValueError(f"Invalid control type: {self.control_type}")

        logger.info("Initialized URMujoco.")

    def _step_rtde(self, action):
        if self.control_type == "joint":
            joint_q = self._data.qpos[:6]
            self._rtde_c.servoJ(
                joint_q,
                self._rtde_velocity,
                self._rtde_acceleration,
                self._rtde_time,
                self._rtde_lookahead_time,
                self._rtde_gain,
            )

        elif self.control_type == "mocap":
            actual_tcp = np.array(self._rtde_r.getActualTCPPose())
            target_tcp = actual_tcp
            action[0:2] *= -1  # invert x and y axis
            target_tcp[:3] += action[:3] * self._action_factor
            target_copy = target_tcp.copy()
            target_tcp[:3] = np.clip(target_tcp[:3], self.limits.tcp_pos_min, self.limits.tcp_pos_max)

            if not np.allclose(target_tcp[:3], target_copy[:3]):
                logger.warning(f"Target TCP {target_copy[:3]} is out of bounds. Clipping to {target_tcp[:3]}.")

            self._rtde_c.servoL(
                target_tcp,
                self._rtde_velocity,
                self._rtde_acceleration,
                self._rtde_time,
                self._rtde_lookahead_time,
                self._rtde_gain,
            )
            pass
        else:
            raise ValueError(f"Invalid control type: {self.control_type}")

    @property
    def type(self):
        """Type of the environment: rtde..."""
        return "rtde"

    def _reset_rtde(self):
        self._rtde_c.servoStop()
        self._rtde_c.moveJ(self._data.qpos[:6], 1.5, 1.5)
        time.sleep(1.0)

    def _sync_rtde_to_sim(self):
        actual_q = self._rtde_r.getActualQ()
        actual_qd = self._rtde_r.getActualQd()
        self._data.qpos[:6] = actual_q
        self._data.qvel[:6] = actual_qd
        mujoco.mj_forward(self._model, self._data)

    def _close_rtde(self):
        self._rtde_c.disconnect()
        self._rtde_r.disconnect()


class URRtdeReachEnv(RtdeMixin, URMujocoReachEnv):
    metadata = {"render_modes": ["human", "rgb_array", "realsense"], "render_fps": 25}

    def __init__(
        self, rtde_control_freq: float = 25.0, rtde_robot_ip: str | None = None, render_mode="realsense", **kwargs
    ):
        super().__init__(render_mode=render_mode, **kwargs)
        self._init_rtde(rtde_control_freq=rtde_control_freq, rtde_robot_ip=rtde_robot_ip)
        if self.render_mode == "realsense":
            self.rs_interface = RealSenseInterface()

    def _step(self, action):
        super()._step(action)
        self._step_rtde(action)
        self._rtde_c.waitPeriod(self._p)
        self._sync_rtde_to_sim()
        self._p = self._rtde_c.initPeriod()

    def _reset(self):
        super()._reset()
        self._reset_rtde()
        self._sync_rtde_to_sim()
        self._p = self._rtde_c.initPeriod()

    def close(self):
        self._close_rtde()
        return super().close()

    def render(self):
        if self.render_mode == "realsense":
            color_img, depth_img = self.rs_interface.get_frame()
            return color_img.copy()
        else:
            return super().render()
