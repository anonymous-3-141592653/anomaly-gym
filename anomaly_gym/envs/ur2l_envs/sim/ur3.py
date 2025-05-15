"""Class to model a UR3 robot in mujoco with different actuators."""

from pathlib import Path
from typing import Literal

import mujoco
import numpy as np


class UR3Model:
    """Class to model a UR3 robot in mujoco."""

    def __init__(
        self,
        control_type: Literal["joint", "mocap", "none"] = "joint",
    ):
        """Initialize the UR3 specification."""
        description_package = Path(__file__).parent
        if control_type == "mocap":
            self.model_path = f"{description_package}/mjcf/ur3.xml"
        elif control_type == "joint":
            self.model_path = f"{description_package}/mjcf/ur3_joint.xml"
        elif control_type == "none":
            self.model_path = f"{description_package}/mjcf/ur3.xml"
        else:
            raise ValueError(
                f"Control type {control_type} is not supported. "
                "Supported control types are 'joint', 'mocap' and 'none'."
            )

        self.spec = mujoco.MjSpec.from_file(self.model_path)

        # Add sensors
        for joint in self.spec.joints:
            self.spec.add_sensor(
                type=mujoco.mjtSensor.mjSENS_JOINTPOS,
                refname=joint.name,
                name=f"{joint.name}:pos",
            )
            self.spec.add_sensor(
                type=mujoco.mjtSensor.mjSENS_JOINTVEL,
                refname=joint.name,
                name=f"{joint.name}:vel",
            )

    def compile(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        """Compile the model."""
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        return self.model, self.data

    def get_joint_pos(
        self,
        config_name: Literal["extended", "ready", "ready_2"] = "ready_2",
    ) -> list[float]:
        """Get joint positions for different configuration types."""
        if config_name == "ready":
            return [1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0]
        elif config_name == "ready_2":
            return [0.0048, -2.32, -2.10, -1.86, -1.59, -0.028]
        elif config_name == "ready_3":
            return [-1.5708, -2.32, -2.10, -1.86, -1.59, -0.028]
        elif config_name == "extended":
            return [0, -1.571, 0, -1.571, 0, 0]
        else:
            raise ValueError(f"Unknown config name: {config_name}")

    def get_end_effector_pose(
        self,
        config_name: Literal["extended", "ready"] = "ready",
    ) -> tuple[list[float], list[float]]:
        """Get end effector pose for a certain configuration.

        Remark: This is w.r.t. the base frame.
        """

        if config_name == "ready":
            return [-0.11234822, +0.29730842, 0.16113758], [0, 1, 0, 0]
        elif config_name == "extended":
            return [-1.27804935e-04, 2.94250000e-01 + 0.05, 6.93999983e-01], [0.707, -0.707, 0, 0]
        else:
            raise ValueError(f"Unknown config name: {config_name}")
