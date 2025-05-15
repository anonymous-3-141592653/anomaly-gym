"""Some common util functions to create the simulation."""

import mujoco
import numpy as np

from anomaly_gym.envs.ur2l_envs.sim.arena import Arena
from anomaly_gym.envs.ur2l_envs.sim.gripper import GripperModel
from anomaly_gym.envs.ur2l_envs.sim.ur3 import UR3Model


def attach_gripper(ur3: UR3Model, gripper: GripperModel):
    """Attach the gripper to the UR3."""
    flange = ur3.spec.find_frame("flange")
    flange.attach_body(gripper.spec.worldbody, "2f85/", "")


def attach_robot_arm(arena: Arena, ur3: UR3Model):
    """Attach the robot arm to the arena."""
    robot_base = arena.spec.find_frame("robot_base")
    robot_base.attach_body(ur3.spec.worldbody, "ur3/", "")


def get_robot_base_pose(arena: Arena):
    """Get the pose of the base of the robot."""
    robot_base = arena.spec.find_body("table_stand")
    return robot_base.pos, robot_base.quat


def add_target(
    arena: Arena,
    ur3: UR3Model,
    starting_config: str = "ready",
):
    """Add the target to the arena and weld it to the end effector.

    This allows the robot to be controlled using the mocap object.

    Args:
        arena: The arena to add the target to.
        ur3: The UR3 robot to add the target to.
        starting_config: The starting config of the end effector.
    """

    ee_pos, ee_quat = ur3.get_end_effector_pose(starting_config)

    target_body = arena.spec.worldbody.add_body(name="target", mocap=True, pos=ee_pos, quat=ee_quat)
    target_body.add_site(name="target")
    sizes = [
        [0.005, 0.005, 0.005],  # small cube for a visible center.
        [1, 0.005, 0.005],  # long thin box along the x-axis (crosshair).
        [0.005, 1, 0.005],  #  long thin box along the y-axis (crosshair).
        [0.005, 0.005, 1],  # long thin box along the z-axis (crosshair).
    ]
    for size in sizes:
        target_body.add_geom(
            conaffinity=0,
            contype=0,
            pos=[0, 0, 0],
            rgba=[0, 0.5, 0, 0.2],
            size=size,
            type=mujoco.mjtGeom.mjGEOM_BOX,
        )

    arena.spec.add_equality(
        type=mujoco.mjtEq.mjEQ_WELD,
        name1="target",
        name2="ur3/end_effector",
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        solimp=[0.9, 0.95, 0.001, 0.5, 2],
        solref=[0.04, 1.0],
    )


def set_initial_conditions(
    arena: Arena,
    ur_qpos: list[float],
    gripper_qpos: list[float] = 8 * [0.0],
    ur_ctrl: list[float] = None,
    gripper_ctrl: list[float] = None,
    extra_qpos: list[float] = None,
    extra_ctrl: list[float] = None,
):
    """Set the initial conditions for the simulation."""
    if ur_ctrl is None:
        ur_ctrl = []
    if gripper_ctrl is None:
        gripper_ctrl = [0.0]
    if extra_qpos is None:
        extra_qpos = []
    if extra_ctrl is None:
        extra_ctrl = []
    qpos = ur_qpos + gripper_qpos + extra_qpos
    ctrl = ur_ctrl + gripper_ctrl + extra_ctrl
    arena.spec.add_key(name="default", qpos=qpos, ctrl=ctrl)


def add_goal(arena: Arena, goal_pos: list[float] = None):
    """Add a goal body to the arena which is a mocap body as well.

    This is needed for the reach task.
    """
    if goal_pos is None:
        goal_pos = [-0.11234822, +0.29730842, 0.16113758]
    goal = arena.spec.worldbody.add_body(name="goal", mocap=True, pos=goal_pos)

    goal.add_site(name="goal")
    sizes = [
        [0.005, 0.005, 0.005],
        [1, 0.005, 0.005],
        [0.005, 1, 0.005],
        [0.005, 0.005, 1],
    ]
    for size in sizes:
        goal.add_geom(
            conaffinity=0,
            contype=0,
            pos=[0, 0, 0],
            rgba=[0, 0, 0.5, 0.2],
            size=size,
            type=mujoco.mjtGeom.mjGEOM_BOX,
        )


def add_table(arena: Arena, table_pos: list[float] = None):
    """Add a table to the arena."""
    if table_pos is None:
        table_pos = [0, 0.1, 0]
    table = arena.spec.worldbody.add_body(name="table", pos=table_pos)
    table.add_geom(
        name="table",
        pos=table_pos,
        rgba=[0.5, 0.5, 0.5, 1.0],
        mass=0.001,
        condim=4,
        friction=[0.99, 0.99, 0.99],
        size=[0.1, 0.1, 0.1],
        type=mujoco.mjtGeom.mjGEOM_BOX,
    )
