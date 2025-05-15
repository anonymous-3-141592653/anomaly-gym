"""Init module for ur2l_gym.sim."""

from anomaly_gym.envs.ur2l_envs.sim.base import MujocoSim, make_sim

__all__ = [
    "MujocoSim",
    "make_sim",
]
