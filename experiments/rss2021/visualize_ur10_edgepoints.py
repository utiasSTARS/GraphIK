import graphik
import numpy as np

from graphik.utils.roboturdf import RobotURDF, plot_balls_from_points
from matplotlib import pyplot as plt
from numpy import pi
from numpy.linalg import norm

from graphik.graphs.graph_base import RobotGraph, RobotRevoluteGraph

# from graphik.robots.revolute import Revolute3dChain
from graphik.robots.robot_base import RobotRevolute
from graphik.solvers.riemannian_solver import RiemannianSolver

from numpy.testing import assert_array_less
from graphik.utils.dgp import (
    adjacency_matrix_from_graph,
    pos_from_graph,
    graph_from_pos,
)
from graphik.utils.utils import (
    best_fit_transform,
    list_to_variable_dict,
    dZ,
    safe_arccos,
)

from graphik.utils.dgp import dist_to_gram, distance_matrix_from_graph

if __name__ == "__main__":
    np.random.seed(21)

    n = 6
    angular_limits = np.minimum(np.random.rand(n) * (pi / 2) + pi / 2, pi)
    ub = angular_limits
    lb = -angular_limits

    fname = graphik.__path__[0] + "/robots/urdfs/ur10_mod.urdf"
    urdf_robot = RobotURDF(fname)
    robot = urdf_robot.make_Revolute3d(ub, lb)  # make the Revolute class from a URDF
    q_zero = list_to_variable_dict(n * [0])
    urdf_robot.visualize(
        q=q_zero, with_frames=False, with_balls=True, with_edges=True, with_robot=True, transparency=0.5
    )
