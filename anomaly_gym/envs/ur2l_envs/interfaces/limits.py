import numpy as np


class URLimits:
    joint_min: list[float]
    joint_max: list[float]
    ee_pos_min: list[float]
    ee_pos_max: list[float]
    ee_rot_min: list[float]  # roll, pitch, yaw
    ee_rot_max: list[float]  # roll, pitch, yaw
    tcp_pos_min: list[float]
    tcp_pos_max: list[float]
    q_ctrl_min: float
    q_ctrl_max: float
    target_min: list[float]
    target_max: list[float]
    max_gripper_width: float
    min_gripper_width: float
    goal_min: list[float]
    goal_max: list[float]


class ReachLimits(URLimits):
    def __init__(self):
        # hard limits
        self.joint_min = [-np.pi, -np.pi, 0.25, -np.pi, -np.pi, -np.pi]
        self.joint_max = [np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]
        self.ee_pos_min = [-0.3, 0.2, 0.05]
        self.ee_pos_max = [0.2, 0.4, 0.25]
        self.ee_rot_min = [np.pi - 0.4, -0.4, -0.4]
        self.ee_rot_max = [np.pi + 0.4, 0.4, 0.4]

        self.tcp_pos_min = [-0.2, -0.4, 0.175]  # * [-1, -1, 1] + [0, 0, 0.125]
        self.tcp_pos_max = [0.3, -0.2, 0.375]

        # soft limits
        self.q_ctrl_min = -0.1
        self.q_ctrl_max = 0.1
        self.target_min = [-0.3, 0.2, 0.05]
        self.target_max = [0.2, 0.4, 0.25]

        # soft gripper limits
        self.max_gripper_width = 0.084
        self.min_gripper_width = 0.01

        # goal limits
        self.goal_min = [-0.25, 0.25, 0.1, 0.7, -0.7, 0.0, 0.0]
        self.goal_max = [0.15, 0.35, 0.2, 0.7, -0.7, 0.0, 0.0]


class PickAndPlaceLimits(ReachLimits):
    def __init__(self):
        super().__init__()
        # block limits
        self.block_min = [-0.15, 0.2, 0.055, 0.0, 0.0, 0.0, 0.0]
        self.block_max = [0.15, 0.4, 0.055, 0.0, 0.0, 0.0, 0.0]
