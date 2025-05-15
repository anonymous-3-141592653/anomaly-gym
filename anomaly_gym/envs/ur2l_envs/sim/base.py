"""Function to create a simulation in mujoco."""

from copy import deepcopy
from dataclasses import dataclass

import mujoco

from anomaly_gym.envs.ur2l_envs.sim.pick_and_place import make_pick_and_place_sim
from anomaly_gym.envs.ur2l_envs.sim.reach import make_reach_sim


@dataclass
class MujocoSim:
    """Mujoco simulation object with model and data."""

    model: mujoco.MjModel
    data: mujoco.MjData
    robot_name: str
    gripper_name: str

    def copy_state(self, other: "MujocoSim"):
        """Copy the whole simulation state from an other simulation to this one.

        Note: we assume that the simulation data is generated from the same model.
        https://mujoco.readthedocs.io/en/latest/programming/simulation.html#state-and-control
        """
        self.data.time = other.data.time
        mujoco.mju_copy(self.data.qpos, other.data.qpos)  # self.model.nq
        mujoco.mju_copy(self.data.qvel, other.data.qvel)  # self.model.nv
        mujoco.mju_copy(self.data.act, other.data.act)  # self.model.na

        # copy mocap body pose and userdata
        self.data.mocap_pos = other.data.mocap_pos.copy()  # self.model.nmocap
        self.data.mocap_quat = other.data.mocap_quat.copy()  # self.model.nmocap
        mujoco.mju_copy(self.data.userdata, other.data.userdata)  # self.model.nuserdata

        # copy warm - start acceleration
        mujoco.mju_copy(self.data.qacc_warmstart, other.data.qacc_warmstart)  # self.model.nv

    @classmethod
    def from_sim_object(cls, other_sim: "MujocoSim"):
        """Create a simulation from another simulation object."""
        model = deepcopy(other_sim.model)
        data = deepcopy(other_sim.data)
        cls = MujocoSim(model=model, data=data, robot_name=other_sim.robot_name, gripper_name=other_sim.robot_name)
        cls.copy_state(other=other_sim)
        return cls

    @property
    def sim_dt(self):
        """Get the simulation timestep."""
        return self.model.opt.timestep

    @property
    def sim_freq(self):
        """Return the simulation frequency."""
        return 1.0 / self.sim_dt

    def advance_simulation(self, n_steps: int = 1):
        """Advance the simulation for a given number of steps."""
        mujoco.mj_step(self.model, self.data, n_steps)


def make_sim(sim_name: str, **kwargs) -> MujocoSim:
    """Make a simulation based on a configuration dictionary."""
    if sim_name == "reach":
        model, data = make_reach_sim(**kwargs)
        return MujocoSim(model=model, data=data, robot_name="ur3", gripper_name="2f85")
    elif sim_name == "pick_and_place":
        model, data = make_pick_and_place_sim(**kwargs)
        return MujocoSim(model=model, data=data, robot_name="ur3", gripper_name="2f85")
    else:
        raise NotImplementedError
