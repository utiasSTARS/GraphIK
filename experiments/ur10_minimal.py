import networkx as nx
import numpy as np
import graphik
from graphik.robots.robot_base import RobotRevolute
from graphik.graphs.graph_base import RobotGraph, RobotRevoluteGraph
from graphik.solvers.riemannian_solver import RiemannianSolver
from graphik.utils.dgp import (
    adjacency_matrix_from_graph,
    bound_smoothing,
    graph_from_pos,
    pos_from_graph,
)
from graphik.utils.geometry import trans_axis
from graphik.utils.roboturdf import RobotURDF
from graphik.utils.utils import best_fit_transform, list_to_variable_dict, safe_arccos
from numpy import pi
from numpy.linalg import norm

POS = "pos"
DIST = "weight"

if __name__ == "__main__":

    # fname = graphik.__path__[0] + "/robots/urdfs/ur10_mod.urdf"
    # # fname = graphik.__path__[0] + "/robots/urdfs/lwa4p.urdf"
    # # fname = graphik.__path__[0] + "/robots/urdfs/lwa4d.urdf"
    # # fname = graphik.__path__[0] + "/robots/urdfs/panda_arm.urdf"
    # # fname = graphik.__path__[0] + "/robots/urdfs/kuka_iiwr.urdf"
    # # fname = graphik.__path__[0] + "/robots/urdfs/kuka_lwr.urdf"
    # # fname = graphik.__path__[0] + "/robots/urdfs/jaco2arm6DOF_no_hand.urdf"
    # #
    # ub = pi * np.ones(6)
    # lb = -ub
    # urdf_robot = RobotURDF(fname)
    # robot = urdf_robot.make_Revolute3d(ub, lb)  # make the Revolute class from a URDF
    # graph = RobotRevoluteGraph(robot)

    # q_zero = list_to_variable_dict(robot.n * [0])
    # G = graph.realization(q_zero)
    # P = dict(G.nodes(data=POS))

    S = nx.Graph(nx.empty_graph())
    S.add_node("p0", **{POS: np.array([0, 0, 0])})
    S.add_node("x", **{POS: np.array([1, 0, 0])})
    S.add_node("y", **{POS: np.array([0, 1, 0])})
    S.add_node("q0", **{POS: np.array([0, 0, 1])})
    S.add_node("q1", **{POS: np.array([1, 0, 0])})
    S.add_node("p2", **{POS: np.array([0.0, 0.049041, 0.612])})
    S.add_node("q2", **{POS: np.array([0.0, 1.049041, 0.612])})
    # S.add_node("p3", **{POS: array([0.0, 0.049041, 1.1843])})
    # S.add_node("q3", **{POS: array([0.0, 1.049041, 1.1843])})
    S.add_node("p3", **{POS: np.array([0.0, 0.163941, 1.1843])})  # we equate p3 and p4
    S.add_node("q3", **{POS: np.array([0.0, 1.163941, 1.1843])})
    S.add_node("q4", **{POS: np.array([1.0, 0.163941, 1.1843])})
    S.add_node("p5", **{POS: np.array([0.1157, 0.256141, 1.1843])})  # actually p6

    S.add_edges_from(
        [
            ("p0", "x"),
            ("p0", "y"),
            ("p0", "q0"),
            ("x", "y"),
            ("x", "q0"),
            ("y", "q0"),
            ("q0", "q1"),
            ("p0", "q1"),
            ("p0", "p2"),
            ("p0", "q2"),
            ("q1", "p2"),
            ("q1", "q2"),
            ("p2", "q2"),
            ("p2", "p3"),
            ("p2", "q3"),
            ("q2", "p3"),
            ("q2", "q3"),
            ("p3", "q3"),
            ("p3", "q4"),
            ("q3", "q4"),
            ("p3", "p5"),
            ("q4", "p5"),
        ]
    )

    for u, v in S.edges():
        S[u][v][DIST] = np.linalg.norm(S.nodes[u][POS] - S.nodes[v][POS])

    print(S.edges(data=True))
