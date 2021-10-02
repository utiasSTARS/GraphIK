from typing import Dict, List, Any
import numpy as np
import numpy.linalg as la
from graphik.robots import RobotSpherical
from graphik.graphs.graph_base import RobotGraph
from graphik.utils.constants import *
import networkx as nx
from numpy import cos, pi, sqrt


class RobotSphericalGraph(RobotGraph):
    def __init__(
        self,
        robot: RobotSpherical,
    ):
        self.dim = robot.dim
        self.robot = robot
        self.structure = robot.structure
        self.base = self.base_subgraph()
        self.directed = nx.compose(self.base, self.structure)
        # self.directed = nx.freeze(self.root_angle_limits(self.directed))
        self.directed = self.root_angle_limits(self.directed)
        super(RobotSphericalGraph, self).__init__()

    def root_angle_limits(self, G: nx.DiGraph) -> nx.DiGraph:
        ax = "z"

        S = self.robot.structure
        l1 = la.norm(G.nodes[ax][POS])
        for node in S.successors(ROOT):
            if DIST in S[ROOT][node]:
                l2 = S[ROOT][node][DIST]
                lb = self.robot.lb[node]
                ub = self.robot.ub[node]
                lim = max(abs(ub), abs(lb))

                # Assumes bounds are less than pi in magnitude
                G.add_edge(ax, node)
                G[ax][node][UPPER] = l1 + l2
                G[ax][node][LOWER] = sqrt(
                    l1 ** 2 + l2 ** 2 - 2 * l1 * l2 * cos(pi - lim)
                )
                G[ax][node][BOUNDED] = ["below"]

        return G

    @staticmethod
    def base_subgraph() -> nx.DiGraph:
        base = nx.DiGraph(
            [
                ("p0", "x"),
                ("p0", "y"),
                ("p0", "z"),
                ("x", "y"),
                ("y", "z"),
                ("z", "x"),
            ]
        )
        base.add_nodes_from(
            [
                ("p0", {POS: np.array([0, 0, 0])}),
                ("x", {POS: np.array([1, 0, 0]), TYPE: "base"}),
                ("y", {POS: np.array([0, 1, 0]), TYPE: "base"}),
                # ("z", {POS: np.array([0, 0, 1])}),
                ("z", {POS: np.array([0, 0, -1]), TYPE: "base"}),
            ]
        )
        for u, v in base.edges():
            base[u][v][DIST] = la.norm(base.nodes[u][POS] - base.nodes[v][POS])
            base[u][v][LOWER] = base[u][v][DIST]
            base[u][v][UPPER] = base[u][v][DIST]
        return base

    def realization(self, joint_angles: Dict[str, List[float]]) -> nx.DiGraph:
        """
        Given a dictionary of joint variables generate a representative graph.
        This graph will be fully conncected.
        """
        P = {}
        for name in self.robot.structure:  # TODO make single for loop
            P[name] = self.robot.get_pose(joint_angles, name).trans

        for name in self.base:
            P[name] = self.base.nodes[name][POS]

        return self.complete_from_pos(P)

    def complete_from_pose_goal(self, pose_goals):
        pos = {}
        for ee in self.robot.end_effectors:
            T_goal = pose_goals[ee[0]]  # first ID is the last point in chain
            d = self.directed[ee[1]][ee[0]][DIST]
            z = T_goal.rot.as_matrix()[0:3, -1]
            pos[ee[0]] = T_goal.trans
            pos[ee[1]] = T_goal.trans - z * d

        return self.complete_from_pos(pos)
