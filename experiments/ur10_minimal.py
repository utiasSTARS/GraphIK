import networkx as nx
import numpy as np
import graphik
from graphik.robots.robot_base import RobotRevolute
from graphik.graphs.graph_base import RobotGraph, RobotRevoluteGraph
from graphik.solvers.riemannian_solver import RiemannianSolver
from graphik.utils.dgp import (
    adjacency_matrix_from_graph,
    bound_smoothing,
    graph_complete_edges,
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


class MinimalUR10Problem:
    def __init__(self):
        self.generate_graph()

    def generate_graph(self):

        S = nx.DiGraph(nx.empty_graph())
        S.add_node("p0", **{POS: np.array([0, 0, 0])})
        S.add_node("x", **{POS: np.array([1, 0, 0])})
        S.add_node("y", **{POS: np.array([0, 1, 0])})
        S.add_node("q0", **{POS: np.array([0, 0, 1])})
        S.add_node("q1", **{POS: np.array([1, 0, 0])})
        S.add_node("p2", **{POS: np.array([0.0, 0.049041, 0.612])})
        S.add_node("q2", **{POS: np.array([0.0, 1.049041, 0.612])})
        # S.add_node("p3", **{POS: array([0.0, 0.049041, 1.1843])})
        # S.add_node("q3", **{POS: array([0.0, 1.049041, 1.1843])})
        S.add_node("p3", **{POS: np.array([0.0, 0.163941, 1.1843])})  # p3 == p4
        S.add_node("q3", **{POS: np.array([0.0, 1.163941, 1.1843])})
        # S.add_node("q4", **{POS: np.array([1.0, 0.163941, 1.1843])}) # q4 is where p4 used to be
        S.add_node("q4", **{POS: np.array([0.1157, 0.163941, 1.1843])})
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

        for u in S:
            del S.nodes[u][POS]

        self.directed = S

        S.add_node("p0", **{POS: np.array([0, 0, 0])})
        S.add_node("x", **{POS: np.array([1, 0, 0])})
        S.add_node("y", **{POS: np.array([0, 1, 0])})
        S.add_node("q0", **{POS: np.array([0, 0, 1])})

    def complete_from_pos(self, goals: dict):

        G = self.directed.copy()
        for key, val in goals.items():
            if key in G.nodes:
                G.nodes[key][POS] = goals[key]

        return graph_complete_edges(G)


if __name__ == "__main__":

    fname = graphik.__path__[0] + "/robots/urdfs/ur10_mod.urdf"
    ub = pi * np.ones(6)
    lb = -ub
    urdf_robot = RobotURDF(fname)
    robot = urdf_robot.make_Revolute3d(ub, lb)  # make the Revolute class from a URDF
    graph = RobotRevoluteGraph(robot)

    q = list_to_variable_dict(robot.n * [0])
    T_goal_1, T_goal_2 = robot.get_pose(q, f"p6"), robot.get_pose(q, f"p4")

    # p5 is p6 and q4 is p4
    goals = {
        f"p5": T_goal_1.trans,
        f"q4": T_goal_2.trans,
    }

    ur10_min = MinimalUR10Problem()
    H = ur10_min.complete_from_pos(goals)
    # print(H.edges(data=True))
    print(adjacency_matrix_from_graph(H))
