from abc import ABC, abstractmethod

import networkx as nx
import numpy as np
import numpy.linalg as la
from typing import Dict, List, Any
from numpy.typing import ArrayLike
from graphik.robots.robot_base import Robot, RobotPlanar, RobotRevolute
from graphik.utils.constants import *
from graphik.utils.dgp import (
    graph_complete_edges,
    distance_matrix_from_graph,
    adjacency_matrix_from_graph,
)
from graphik.utils.geometry import rot_axis, trans_axis
from liegroups.numpy import SE3
from numpy import cos, pi, sqrt


class RobotGraph(ABC):
    """
    Abstract base class for graph structures equipped with optimization and EDM completion features.
    Note that the robots analyzed by this graph have a joint structure described by a tree, yet the constraints between
    the variables (e.g., graph coordinate points) induced by these joints have a graph structure (hence the name).
    """

    def __init__(self):
        pass

    @property
    def robot(self) -> Robot:
        """
        Robot object that this graph represents.
        :returns: Robot object
        """
        return self._robot

    @robot.setter
    def robot(self, robot: Robot):
        self._robot = robot

    @property
    def directed(self) -> nx.DiGraph:
        """
        Directed graph representing the connectivity of robot nodes to nodes
        defined by the base and environment.
        :returns: Graph object representing the above
        """
        return self._directed

    @directed.setter
    def directed(self, G: nx.DiGraph):
        self._directed = G

    @property
    def base(self) -> nx.DiGraph:
        return self._base

    @base.setter
    def base(self, G: nx.DiGraph):
        self._base = G

    @property
    def dim(self) -> int:
        """
        :returns: Expected lowest embedding dimension of the graph
        """
        return self._dim

    @dim.setter
    def dim(self, dim: int):
        self._dim = dim

    @property
    def n_nodes(self) -> int:
        """
        :returns: Number of nodes in the graph
        """
        return self.directed.number_of_nodes()

    @property
    def node_ids(self) -> List[str]:
        """
        :returns: List of nodes in this graph.
        """
        return list(self.directed.nodes())

    @abstractmethod
    def realization(self, joint_angles: Dict[str, Any]) -> nx.DiGraph:
        """
        Given a set of joint angles, return a graph realization in R^dim.
        :param x: Decision variables (revolute joints, prismatic joints)
        :returns: Graph with node locations stored in the [POS]
        atribute and edge weights corresponding to distances between the nodes.
        """
        raise NotImplementedError

    def distance_matrix(self) -> ArrayLike:
        """
        Returns a partial distance matrix of known distances in the problem graph.
        :returns: Distance matrix
        """
        return distance_matrix_from_graph(self.directed)

    def distance_matrix_from_joints(self, joint_angles: ArrayLike) -> ArrayLike:
        """
        Given a set of joint angles, return a matrix whose element
        [idx,jdx] corresponds to the squared distance between nodes idx and jdx.
        :param x: Decision variables (revolute joints, prismatic joints)
        :returns: Matrix of squared distances
        """
        return distance_matrix_from_graph(self.realization(joint_angles))

    def adjacency_matrix(self) -> ArrayLike:
        """
        Returns the adjacency matrix representing the edges that are known,
        given the kinematic and base structure, as well as the end-effector targets.
        :returns: Adjacency matrix
        """
        return adjacency_matrix_from_graph(self.directed)

    def complete_from_pos(self, P: dict, dist: bool = True) -> nx.DiGraph:
        """
        Given a dictionary of node name and position key-value pairs,
        generate a copy of the problem graph and fill the POS attributes of
        nodes corresponding to keys with assigned values.
        If dist is True, populate all edges between nodes with assinged POS attributes,
        and return the new graph.
        Note that this function maintains the node ordering of self.directed!
        :param P: a dictionary of node name position pairs
        :returns: graph with connected nodes with POS attribute
        """
        G = self.directed.copy()  # copy of the original object

        for name, pos in P.items():
            if name in self.node_ids:
                G.nodes[name][POS] = pos

        if dist:
            G = graph_complete_edges(G)

        return G

    def add_anchor_node(self, name: str, data: Dict[str, Any]):
        if POS not in data:
            raise KeyError("Node needs to gave a position to be added.")

        self.directed.add_nodes_from([(name, data)])
        for nname, ndata in self.directed.nodes(data=True):
            if POS in ndata and nname != name:
                self.directed.add_edge(nname, name)
                self.directed[nname][name][DIST] = la.norm(ndata[POS] - data[POS])
                self.directed[nname][name][LOWER] = la.norm(ndata[POS] - data[POS])
                self.directed[nname][name][UPPER] = la.norm(ndata[POS] - data[POS])
                self.directed[nname][name][BOUNDED] = []

    def add_spherical_obstacle(self, name: str, position: ArrayLike, radius: float):
        # Add a fixed node representing the obstacle to the graph
        self.add_anchor_node(name, {POS: position, TYPE: OBSTACLE})

        # Set lower (and upper) distance limits to robot nodes
        for node, node_type in self.directed.nodes(data=TYPE):
            if node_type == ROBOT and node[0] == "p":
                self.directed.add_edge(node, name)
                self.directed[node][name][BOUNDED] = [BELOW]
                self.directed[node][name][LOWER] = radius
                self.directed[node][name][UPPER] = 100

    def check_distance_limits(
        self, G: nx.DiGraph, tol=1e-10
    ) -> List[Dict[str, List[Any]]]:
        """Given a graph of the same """
        typ = nx.get_node_attributes(self.directed, name=TYPE)
        # broken_limits = {"edge": [], "value": [], "type": [], "side": []}
        broken_limits = []
        for u, v, data in self.directed.edges(data=True):
            if BELOW in data[BOUNDED] or ABOVE in data[BOUNDED]:
                if G[u][v][DIST] < data[LOWER] - tol:
                    broken_limit = {}
                    if (typ[u] == "robot" and typ[v] == "obstacle") or (
                        typ[u] == "obstacle" and typ[v] == "robot"
                    ):
                        broken_limit["edge"] = (u, v)
                        broken_limit["value"] = G[u][v][DIST] - data[LOWER]
                        broken_limit["type"] = "obstacle"
                        broken_limit["side"] = LOWER
                        broken_limits += [broken_limit]
                    if typ[u] == "robot" and typ[v] == "robot":
                        broken_limit["edge"] = (u, v)
                        broken_limit["value"] = G[u][v][DIST] - data[LOWER]
                        broken_limit["type"] = "joint"
                        broken_limit["side"] = LOWER
                        broken_limits += [broken_limit]
                if G[u][v][DIST] > data[UPPER] + tol:
                    broken_limit = {}
                    if (typ[u] == "robot" and typ[v] == "obstacle") or (
                        typ[u] == "obstacle" and typ[v] == "robot"
                    ):
                        broken_limit["edge"] = (u, v)
                        broken_limit["value"] = G[u][v][DIST] - data[UPPER]
                        broken_limit["type"] = "obstacle"
                        broken_limit["side"] = UPPER
                        broken_limits += [broken_limit]
                    if typ[u] == "robot" and typ[v] == "robot":
                        broken_limit["edge"] = (u, v)
                        broken_limit["value"] = G[u][v][DIST] - data[UPPER]
                        broken_limit["type"] = "joint"
                        broken_limit["side"] = UPPER
                        broken_limits += [broken_limit]

        return broken_limits

    def distance_bound_matrices(self) -> ArrayLike:
        """
        Generates a matrices of distance bounds induced by joint variables.
        """
        L = np.zeros([self.n_nodes, self.n_nodes])  # fake distance matrix
        U = np.zeros([self.n_nodes, self.n_nodes])  # fake distance matrix
        G = self.directed
        for e1, e2, data in G.edges(data=True):
            if BOUNDED in data:
                # print(e1, e2, data)
                udx = self.node_ids.index(e1)
                vdx = self.node_ids.index(e2)
                if "below" in data[BOUNDED]:
                    L[udx, vdx] = data[LOWER] ** 2
                    L[vdx, udx] = L[udx, vdx]
                if "above" in data[BOUNDED]:
                    U[udx, vdx] = data[UPPER] ** 2
                    U[vdx, udx] = U[udx, vdx]
        return L, U


class RobotPlanarGraph(RobotGraph):
    def __init__(self, robot: RobotPlanar):
        self.dim = robot.dim
        self.robot = robot
        self.structure = robot.structure
        self.base = self.base_subgraph()
        self.directed = nx.compose(self.base, self.structure)
        # self.directed = nx.freeze(self.root_angle_limits(self.directed))
        self.directed = self.root_angle_limits(self.directed)
        super(RobotPlanarGraph, self).__init__()

    @staticmethod
    def base_subgraph() -> nx.DiGraph:
        base = nx.DiGraph([("p0", "x"), ("p0", "y"), ("x", "y")])

        # Invert x axis because of the way joint limits are set up, makes no difference
        base.add_nodes_from(
            [
                ("p0", {POS: np.array([0, 0])}),
                ("x", {POS: np.array([-1, 0]), TYPE: "base"}),
                ("y", {POS: np.array([0, 1]), TYPE: "base"}),
            ]
        )

        for u, v in base.edges():
            base[u][v][DIST] = la.norm(base.nodes[u][POS] - base.nodes[v][POS])
            base[u][v][LOWER] = base[u][v][DIST]
            base[u][v][UPPER] = base[u][v][DIST]

        return base

    def root_angle_limits(self, G: nx.DiGraph) -> nx.DiGraph:
        ax = "x"

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
                G[ax][node][BOUNDED] = "below"
        return G

    def realization(self, joint_angles: Dict[str, float]) -> nx.DiGraph:
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

    def complete_from_pose_goal(self, pose_goals: dict):
        pos = {}
        for ee in self.robot.end_effectors:
            T_goal = pose_goals[ee[0]]  # first ID is the last point in chain
            d = self.directed[ee[1]][ee[0]][DIST]
            z = T_goal.rot.as_matrix()[0:3, -1]
            pos[ee[0]] = T_goal.trans
            pos[ee[1]] = T_goal.trans - z * d

        return self.complete_from_pos(pos)


class RobotSphericalGraph(RobotGraph):
    def __init__(
        self,
        robot: Robot,
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
