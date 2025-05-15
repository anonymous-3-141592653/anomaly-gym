"""Create the pick and place environment simulation."""

import mujoco
import numpy as np

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


def make_pick_and_place_sim(
    ur_config: str = "ready",
    goal_pos: list[float] = None,
    block_pos: list[float] = None,
    block_quat: list[float] = None,
    block_size: list[float] = None,
    save_xml_file: str = None,
    control_type="mocap",
):
    """Make a reach simulation in mujoco."""

    arena = Arena()
    ur3 = UR3Model(control_type="mocap")
    gripper = GripperModel()

    ur3_pos, _ = ur3.get_end_effector_pose(ur_config)

    if block_quat is None:
        block_quat = np.array([1, 0, 0, 0])
    if block_pos is None:
        block_pos = np.array([0, 0.35, 0.045])

    attach_gripper(ur3, gripper)
    attach_robot_arm(arena, ur3)

    add_goal(arena, goal_pos=goal_pos)
    add_block(arena, block_pos, block_quat, block_size)

    ur3_joints_qpos = ur3.get_joint_pos(ur_config)
    if control_type == "joint":
        set_initial_conditions(arena, ur_qpos=ur3_joints_qpos, ur_ctrl=ur3_joints_qpos)
    else:
        add_target(arena, ur3, starting_config=ur_config)
        set_initial_conditions(arena, ur_qpos=ur3_joints_qpos, extra_qpos=block_pos.tolist() + block_quat.tolist())

    model, data = arena.compile()
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    if save_xml_file:
        with open(save_xml_file, "w") as f:
            f.write(arena.spec.to_xml())

    return model, data


def add_block(arena: Arena, block_pos: list[float], block_quat: list[float], block_size: list[float] = None):
    """Add a block to the simulation."""
    if block_size is None:
        block_size = [0.03, 0.03, 0.03]
    W, _, H = block_size
    block_body = arena.spec.worldbody.add_body(name="block", pos=block_pos, quat=block_quat)
    block_body.add_site(name="block")
    block_body.add_geom(
        name="block",
        pos=[0, 0, 0],
        rgba=[0, 0.5, 0, 1.0],
        mass=0.001,
        condim=4,
        solref=[0.0001, 1],
        # solimp=[1, 1, 0.001, 0.5, 2],
        margin=0.0005,
        friction=[0.99, 0.99, 0.99],
        size=block_size,
        type=mujoco.mjtGeom.mjGEOM_BOX,
    )
    block_body.add_freejoint(name="block_freejoint")
    block_body.add_site(
        name="top_sensor_site",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0, 0, H],
        size=[W - 0.002, W - 0.002, 0.001],
        rgba=[1, 0, 0, 1],
    )
    block_body.add_site(
        name="left_sensor_site",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[W, 0, 0],
        size=[0.001, W, H],
        rgba=[0, 0, 1, 1],
    )
    block_body.add_site(
        name="right_sensor_site",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[-W, 0, 0],
        size=[0.001, W, W],
        rgba=[0, 0, 1, 1],
    )

    arena.spec.add_sensor(
        name="left_touch",
        objname="left_sensor_site",
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        type=mujoco.mjtSensor.mjSENS_TOUCH,
    )
    arena.spec.add_sensor(
        name="right_touch",
        objname="right_sensor_site",
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        type=mujoco.mjtSensor.mjSENS_TOUCH,
    )
    arena.spec.add_sensor(
        name="top_touch",
        objname="top_sensor_site",
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        type=mujoco.mjtSensor.mjSENS_TOUCH,
    )
