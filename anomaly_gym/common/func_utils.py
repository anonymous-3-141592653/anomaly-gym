import numpy as np


# === Some utils ===
def linear_map(
    x: float | np.ndarray,
    a_low: float,
    a_high: float,
    b_low: float,
    b_high: float,
):
    """Map x linearly from the interval [a_low, a_high] to [b_low, b_high]."""
    return (b_high - b_low) / (a_high - a_low) * (x - a_low) + b_low


def min_max_norm(x: float | np.ndarray, min: float, max: float, new_min: float = -1.0, new_max: float = 1.0):
    """Normalize the value x based on the min and max."""
    return linear_map(x, min, max, new_min, new_max)
