"""Function to create a reach simulation in mujoco."""

import mujoco

from anomaly_gym.envs.ur2l_envs.sim.arena import Arena
from anomaly_gym.envs.ur2l_envs.sim.common import (
    add_goal,
    add_target,
    attach_gripper,
    attach_robot_arm,
    set_initial_conditions,
)
from anomaly_gym.envs.ur2l_envs.sim.gripper import GripperModel
from anomaly_gym.envs.ur2l_envs.sim.ur3 import UR3Model

# TODO: Add parameters to configure the task. For now this is good enough.
# TODO: Modularize the code, right now the extra functions are only there for readability


def make_reach_sim(ur_config: str = "ready", save_xml_file: str = None, control_type="mocap"):
    """Make a reach simulation in mujoco."""
    arena = Arena()
    ur3 = UR3Model(control_type=control_type)
    gripper = GripperModel()

    attach_gripper(ur3, gripper)
    attach_robot_arm(arena, ur3)

    add_goal(arena)

    ur3_joints_qpos = ur3.get_joint_pos(ur_config)
    if control_type == "joint":
        set_initial_conditions(arena, ur_qpos=ur3_joints_qpos, ur_ctrl=ur3_joints_qpos)
    else:
        add_target(arena, ur3, starting_config=ur_config)
        set_initial_conditions(arena, ur_qpos=ur3_joints_qpos)

    model, data = arena.compile()
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    if save_xml_file:
        with open(save_xml_file, "w") as f:
            f.write(arena.spec.to_xml())

    return model, data
