"""Class to model the gripper in mujoco."""

from pathlib import Path

import mujoco


class GripperModel:
    """Class to model the gripper in mujoco."""

    def __init__(
        self,
    ):
        """Initialize the gripper specification."""
        description_package = Path(__file__).parent
        self.model_path = f"{description_package}/mjcf/gripper.xml"

        self.spec = mujoco.MjSpec.from_file(self.model_path)

    def compile(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        """Compile the model."""
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        return self.model, self.data
