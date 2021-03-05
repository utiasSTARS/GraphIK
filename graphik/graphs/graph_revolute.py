from typing import Dict, List, Any
import numpy as np
import numpy.linalg as la
from graphik.robots import RobotRevolute
from graphik.graphs.graph_base import RobotGraph
from graphik.utils.constants import *
from graphik.utils.geometry import rot_axis, trans_axis
from liegroups.numpy import SE3
import networkx as nx
from numpy import cos, pi, sqrt


class RobotRevoluteGraph(RobotGraph):
    def __init__(
        self,
        robot: RobotRevolute,
    ):

        self.dim = 3
        self.robot = robot
        self.base = self.base_subgraph()
        self.directed = nx.compose(self.base, self.robot.structure)
        # self.directed = nx.freeze(self.root_angle_limits(self.directed))
        self.directed = self.root_angle_limits(self.directed)
        super(RobotRevoluteGraph, self).__init__()

    def root_angle_limits(self, G: nx.DiGraph) -> nx.DiGraph:
        axis_length = self.robot.axis_length
        T = self.robot.T_zero
        T1 = T["p0"]
        base_names = ["x", "y"]
        names = ["p1", "q1"]
        T_axis = trans_axis(axis_length, "z")

        for base_node in base_names:
            for node in names:
                T0 = SE3.from_matrix(np.identity(4))
                T0.trans = G.nodes[base_node][POS]
                if node[0] == "p":
                    d_max, d_min, limit = self.robot.max_min_distance(T0, T1, T["p1"])
                else:
                    d_max, d_min, limit = self.robot.max_min_distance(
                        T0, T1, T["p1"].dot(T_axis)
                    )

                if limit:
                    if node[0] == "p":
                        T_rel = T1.inv().dot(T["p1"])
                    else:
                        T_rel = T1.inv().dot(T["p1"].dot(T_axis))

                    d_limit = la.norm(
                        T1.dot(rot_axis(self.robot.ub["p1"], "z")).dot(T_rel).trans
                        - T0.trans
                    )

                    if limit == "above":
                        d_max = d_limit
                    else:
                        d_min = d_limit
                    self.robot.limited_joints += ["p1"]  # joint at p0 is limited

                G.add_edge(base_node, node)
                if d_max == d_min:
                    G[base_node][node][DIST] = d_max
                G[base_node][node][BOUNDED] = [limit]
                G[base_node][node][UPPER] = d_max
                G[base_node][node][LOWER] = d_min

        return G

    def base_subgraph(self) -> nx.DiGraph:
        axis_length = self.robot.axis_length
        base = nx.DiGraph(
            [
                ("p0", "x"),
                ("p0", "y"),
                ("p0", "q0"),
                ("x", "y"),
                ("y", "q0"),
                ("q0", "x"),
            ]
        )
        base.add_nodes_from(
            [
                ("p0", {POS: np.array([0, 0, 0])}),
                ("x", {POS: np.array([axis_length, 0, 0]), TYPE: "base"}),
                ("y", {POS: np.array([0, -axis_length, 0]), TYPE: "base"}),
                ("q0", {POS: np.array([0, 0, axis_length])}),
            ]
        )
        for u, v in base.edges():
            base[u][v][DIST] = la.norm(base.nodes[u][POS] - base.nodes[v][POS])
            base[u][v][LOWER] = base[u][v][DIST]
            base[u][v][UPPER] = base[u][v][DIST]
            base[u][v][BOUNDED] = []
        return base

    def realization(self, joint_angles: Dict[str, float]) -> nx.DiGraph:
        """
        Given a dictionary of joint variables generate a representative graph.
        This graph will be fully connected.
        """

        axis_length = self.robot.axis_length
        T_all = self.robot.get_all_poses(joint_angles)

        P = {}
        for node, T in T_all.items():
            P[node] = T.trans
            P["q" + node[1:]] = T.dot(trans_axis(axis_length, "z")).trans
        return self.complete_from_pos(P)

    def distance_bounds_from_sampling(self):
        robot = self.robot
        G = self.directed
        ids = self.node_ids
        q_rand = robot.random_configuration()
        D_min = self.distance_matrix_from_joints(q_rand)
        D_max = self.distance_matrix_from_joints(q_rand)

        for _ in range(2000):
            q_rand = robot.random_configuration()
            D_rand = self.distance_matrix_from_joints(q_rand)
            D_max[D_rand > D_max] = D_rand[D_rand > D_max]
            D_min[D_rand < D_min] = D_rand[D_rand < D_min]

        for idx in range(len(D_max)):
            for jdx in range(len(D_max)):
                e1 = ids[idx]
                e2 = ids[jdx]
                G.add_edge(e1, e2)
                G[e1][e2][LOWER] = np.sqrt(D_min[idx, jdx])
                G[e1][e2][UPPER] = np.sqrt(D_max[idx, jdx])
                if abs(D_max[idx, jdx] - D_min[idx, jdx]) < 1e-5:
                    G[e1][e2][DIST] = abs(D_max[idx, jdx] - D_min[idx, jdx])


if __name__ == "__main__":
    import graphik
    from graphik.utils.roboturdf import RobotURDF

    n = 6
    ub = (pi) * np.ones(n)
    lb = -ub
    modified_dh = False

    fname = graphik.__path__[0] + "/robots/urdfs/ur10_mod.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/lwa4p.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/lwa4d.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/panda_arm.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/kuka_iiwr.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/kuka_lwr.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/jaco2arm6DOF_no_hand.urdf"

    urdf_robot = RobotURDF(fname)
    robot = urdf_robot.make_Revolute3d(ub, lb)  # make the Revolute class from a URDF
    graph = RobotRevoluteGraph(robot)
    import timeit

    print(
        max(
            timeit.repeat(
                "graph.realization(robot.random_configuration())",
                globals=globals(),
                number=1,
                repeat=1000,
            )
        )
    )
