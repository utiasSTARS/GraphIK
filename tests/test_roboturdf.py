"""Error-path tests for RobotURDF."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

import graphik
from graphik.utils.roboturdf import RobotURDF, load_truncated_ur10, load_ur10


@pytest.fixture(scope="module")
def ur10_urdf():
    return RobotURDF(graphik.__path__[0] + "/robots/urdfs/ur10_mod.urdf")


def test_get_parents_without_base_joint_raises_value_error(ur10_urdf):
    joints = list(ur10_urdf.T_zero.keys())
    base_joint = ur10_urdf.find_first_joint()
    joints_without_base = [j for j in joints if j is not base_joint]
    with pytest.raises(ValueError):
        ur10_urdf.get_parents(joints_without_base)


def test_actuated_joint_index_of_fixed_joint_raises_value_error(ur10_urdf):
    fixed_joint = next(
        j for j in ur10_urdf._joints if j not in ur10_urdf.urdf.actuated_joints
    )
    with pytest.raises(ValueError):
        ur10_urdf.actuated_joint_index(fixed_joint)


def _bounds_as_arrays(robot):
    joint_ids = [f"p{idx}" for idx in range(1, robot.n + 1)]
    lb = np.array([robot.lb[joint_id] for joint_id in joint_ids])
    ub = np.array([robot.ub[joint_id] for joint_id in joint_ids])
    return lb, ub


def test_urdf_loader_preserves_explicit_joint_limits():
    lb = np.array([-0.25, -0.35, -0.45, -0.55, -0.65, -0.75])
    ub = np.array([0.25, 0.35, 0.45, 0.55, 0.65, 0.75])

    robot, _ = load_ur10(limits=(lb, ub))

    robot_lb, robot_ub = _bounds_as_arrays(robot)
    assert_allclose(robot_lb, lb)
    assert_allclose(robot_ub, ub)


def test_truncated_ur10_loader_preserves_explicit_joint_limits():
    lb = np.array([-0.25, -0.35, -0.45])
    ub = np.array([0.25, 0.35, 0.45])

    robot, _ = load_truncated_ur10(3, limits=(lb, ub))

    robot_lb, robot_ub = _bounds_as_arrays(robot)
    assert_allclose(robot_lb, lb)
    assert_allclose(robot_ub, ub)
