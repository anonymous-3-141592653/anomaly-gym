"""Abstract class that defines interactions with an UR robot."""

from abc import ABC, abstractmethod

import numpy as np

from anomaly_gym.envs.ur2l_envs.interfaces.limits import URLimits


class URInterface(ABC):
    """Abstract class that defines interactions with an UR robot."""

    limits: URLimits

    def __init__(self, limits):
        """Initialize the URInterface.

        Args:
            limits (URLimits): Limits for the robot.
        """
        self.limits = limits

    @property
    @abstractmethod
    def n_robot_joints(self) -> int:
        """Number of joints of the robot."""
        ...

    @property
    @abstractmethod
    def joint_position(self) -> np.ndarray:
        """Get the current joint position of the robot.

        Returns:
            np.array: Current joint position of the robot (6,).
        """
        ...

    @property
    @abstractmethod
    def joint_velocity(self) -> np.ndarray:
        """Get the current joint velocity of the robot.

        Returns:
            np.array: Current joint velocity of the robot (6,).
        """
        ...

    @property
    @abstractmethod
    def gripper_distance(self) -> float:
        """Get the opening of the gripper."""
        ...

    @property
    @abstractmethod
    def target_position(self) -> np.ndarray:
        """Get the target position of the robot."""
        ...

    @property
    @abstractmethod
    def target_quaternion(self) -> np.ndarray:
        """Get the target quaternion of the robot."""
        ...

    @property
    @abstractmethod
    def goal_position(self) -> np.ndarray:
        """Get the goal position of the robot."""
        ...

    @property
    @abstractmethod
    def goal_quaternion(self) -> np.ndarray:
        """Get the goal quaternion of the robot."""
        ...

    @property
    @abstractmethod
    def block_position(self) -> np.ndarray:
        """Get the block position for pick and place."""
        ...

    @property
    @abstractmethod
    def block_quaternion(self) -> np.ndarray:
        """Get the block quaternion for pick and place."""
        ...

    @property
    @abstractmethod
    def end_effector_position(self) -> np.ndarray:
        """Get the position of the end-effector.

        Returns:
            np.array: Position of the end-effector (3,).
        """
        ...

    @property
    @abstractmethod
    def end_effector_velocity(self) -> np.ndarray:
        """Get the velocity of the end-effector.

        Returns:
            np.array: Velocity of the end-effector (3,).
        """
        ...

    @property
    @abstractmethod
    def end_effector_quaternion(self) -> np.ndarray:
        """Get the quaternion of the end-effector.

        Returns:
            np.array: Quaternion of the end-effector (4,).
        """
        ...

    @property
    @abstractmethod
    def end_effector_rpy(self) -> np.ndarray:
        """Get the roll, pitch, yaw of the end-effector.

        Returns:
            np.array: Roll, pitch, yaw of the end-effector (3,).
        """
        ...

    @property
    @abstractmethod
    def body_collision(self) -> bool:
        """Get the collision of the robot's body."""
        ...

    # === Setters ===
    @abstractmethod
    def set_gripper_distance(self, distance: float):
        """Set the target distance of the gripper."""
        ...

    @abstractmethod
    def set_target(self, target_pos: np.ndarray, target_quat: np.ndarray):
        """Set the target position and quaternion of the robot."""
        ...

    @abstractmethod
    def set_goal(self, position: np.ndarray, quaternion: np.ndarray):
        """Set the goal position and quaternion of the robot."""
        ...

    @abstractmethod
    def set_block(self, position: np.ndarray, quaternion: np.ndarray):
        """Set the block position and quaternion for pick and place."""
        ...

    @abstractmethod
    def set_q_ctrl(self, q_ctrl: np.ndarray):
        """Set the joint control of the robot.

        Args:
            q_ctrl (np.array): Joint control of the robot (6,).
        """
        ...
