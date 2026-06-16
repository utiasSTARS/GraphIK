"""Shared builders for solver tests."""
import numpy as np

from graphik.graphs import ProblemGraph
from graphik.robots import Robot
from graphik.utils.utils import list_to_variable_dict


def planar_chain(n):
    """n-joint planar chain with unit links and +/-pi joint limits."""
    params = {
        "link_lengths": list_to_variable_dict(np.ones(n)),
        "theta": list_to_variable_dict(np.zeros(n)),
        "joint_limits_upper": list_to_variable_dict(np.pi * np.ones(n)),
        "joint_limits_lower": list_to_variable_dict(-np.pi * np.ones(n)),
        "num_joints": n,
        "dim": 2,
    }
    robot = Robot(params)
    return robot, ProblemGraph(robot)
