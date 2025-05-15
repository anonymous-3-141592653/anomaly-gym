"""Action transforms for the gripper that scale an gripper action of a policy from -1 to 1 in a width for the opening of a gripper."""

from abc import ABC, abstractmethod

import numpy as np
from typing_extensions import override


class GripperAction(ABC):
    """Interface for gripper action transforms."""

    action_size: int

    @abstractmethod
    def __init__(self): ...

    @abstractmethod
    def __call__(self, gripper_action: float, current_gripper_distance: float) -> float:
        """Return the desired target width of the gripper."""
        ...


class GripperNoAction(GripperAction):
    """No action transform for the gripper."""

    def __init__(self):
        """Initialize the no action transform."""
        self.action_size = 1

    @override
    def __call__(self, gripper_action: float, current_gripper_distance: float) -> float:
        """Return the desired width of the gripper."""
        return 0


class GripperRelativeAction(GripperAction):
    """Transform the value of the gripper relative."""

    def __init__(self, gripper_control_delta: float = 0.05):
        """Initialize the relative target action transform."""
        self.gripper_control_delta = gripper_control_delta
        self.action_size = 1

    def __call__(self, gripper_action: float, current_gripper_distance: float) -> float:
        """Return the desired width of the gripper."""
        gripper_delta = self.gripper_control_delta * gripper_action
        return current_gripper_distance + gripper_delta


class GripperBinaryAction(GripperAction):
    """Transform the value of the gripper to either open or close the gripper."""

    def __init__(self, min_activation_value: float = 0.2, min_deactivation_value: float = -0.2):
        """Initialize the relative target action transform."""
        assert min_activation_value > min_deactivation_value
        assert -1 <= min_activation_value <= 1
        assert -1 <= min_deactivation_value <= 1
        self.min_activation_value = min_activation_value
        self.min_deactivation_value = min_deactivation_value
        self.is_open = False
        self.action_size = 1

    @override
    def __call__(self, gripper_action: float, current_gripper_distance: float) -> float:
        """Return the desired width of the gripper."""
        # Add hysteresis to prevent chattering
        if not self.is_open:
            self.is_open = gripper_action > self.min_activation_value
        else:
            self.is_open = gripper_action > self.min_deactivation_value
        return int(self.is_open)


def get_gripper_action(gripper_action_type: str, gripper_action_cfg: dict | None = None) -> GripperAction:
    """Get the gripper action transform based on the type."""
    if gripper_action_type == "relative":
        return GripperRelativeAction(**gripper_action_cfg or {})
    elif gripper_action_type == "binary":
        return GripperBinaryAction(**gripper_action_cfg or {})
    elif gripper_action_type == "none":
        return GripperNoAction()
    else:
        raise ValueError(f"Unknown gripper action type: {gripper_action_type}")
