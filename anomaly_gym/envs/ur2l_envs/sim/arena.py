"""Create a simple arena."""

from pathlib import Path
from typing import Literal

import mujoco


class Arena:
    """Create a simple arena."""

    def __init__(
        self,
        task: Literal["simple", "goal", "pick"] = "simple",
    ):
        """Initialize the UR3 specification."""
        description_package = Path(__file__).parent
        self.model_path = f"{description_package}/mjcf/arena.xml"

        self.spec = mujoco.MjSpec.from_file(self.model_path)

    def compile(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        """Compile the model."""
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        return self.model, self.data
