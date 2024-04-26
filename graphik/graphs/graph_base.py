from abc import abstractmethod
from liegroups.numpy.se2 import SE2Matrix
from liegroups.numpy.se3 import SE3Matrix

import networkx as nx
import numpy as np
import numpy.linalg as la
from typing import Dict, List, Any, Union
from numpy.typing import ArrayLike
from graphik.robots.robot_base import Robot, SEMatrix
from graphik.utils.constants import *
from graphik.utils import (
    distance_matrix_from_graph,
    adjacency_matrix_from_graph,
    graph_complete_edges,
)


class ProblemGraph(nx.DiGraph):
    """
    Abstract base class for graph structures equipped with optimization and EDM completion features.
    Note that the robots analyzed by this graph have a joint structure described by a tree, yet the constraints between
    the variables (e.g., graph coordinate points) induced by these joints have a graph structure (hence the name).
    """

    def __init__(self, robot, params):
        super(ProblemGraph, self).__init__(
            robot=robot, dim=robot.dim, axis_length=params.get("axis_length", 1)
        )

    @property
    def base_nodes(self) -> List[str]:
        """
        :returns: List of nodes in this graph describing the base coordinate system.
        """
        try:
            return self._base_nodes
        except AttributeError:
            self._base_nodes = [
                node for node, data in self.nodes(data=TYPE) if BASE in data
            ]
            return self._base_nodes

    @property
    def structure_nodes(self) -> List[str]:
        """
        :returns: List of nodes in this graph describing the robot's structure
        """
        try:
            return self._structure_nodes
        except AttributeError:
            self._structure_nodes = [
                node for node, data in self.nodes(data=TYPE) if ROBOT in data
            ]
            return self._structure_nodes

    @property
    def end_effector_nodes(self) -> List[str]:
        """
        :returns: List of nodes in this graph describing the end effectors
        """
        try:
            return self._end_effector_nodes
        except AttributeError:
            self._end_effector_nodes = [
                node for node, data in self.nodes(data=TYPE) if END_EFFECTOR in data
            ]
            return self._end_effector_nodes

    @property
    def base(self) -> nx.DiGraph:
        """
        :returns: base coordinate system subgraph
        """
        return self.to_directed(as_view=True).subgraph(self.base_nodes)

    @property
    def structure(self) -> nx.DiGraph:
        """
        :returns: robot structure subgraph
        """
        return self.to_directed(as_view=True).subgraph(self.structure_nodes)

    @property
    def robot(self) -> Robot:
        """
        Robot object that this graph represents.
        """
        return self.graph["robot"]

    @property
    def dim(self) -> int:
        """
        Dimensionality of problem
        """
        return self.graph["dim"]

    @property
    def axis_length(self) -> float:
        """
        Length of axis for revolute robots
        """
        return self.graph["axis_length"]

    @property
    def node_ids(self) -> List[str]:
        """
        :returns: List of nodes in this graph.
        """
        return list(self.nodes())

    def realization(self, joint_angles: Dict[str, float]) -> nx.DiGraph:
        """
        Given a set of joint angles, return a graph realization in R^dim.
        :param x: Decision variables (revolute joints, prismatic joints)
        :returns: Graph with node locations stored in the [POS]
        atribute and edge weights corresponding to distances between the nodes.
        """
        T_all = self.robot.get_all_poses(joint_angles)
        return self.from_pos(self._pose_goal(T_all))

    def distance_matrix(self) -> ArrayLike:
        """
        Returns a partial distance matrix of known distances in the problem graph.
        :returns: Distance matrix
        """
        return distance_matrix_from_graph(self.to_undirected(as_view=True))

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
        return adjacency_matrix_from_graph(self.to_undirected(as_view=True))

    def from_pos(self, P: Dict, dist: bool = True, overwrite: bool = False) -> nx.DiGraph:
        """
        Given a dictionary of node name and position key-value pairs,
        generate a copy of the problem graph and fill the POS attributes of
        nodes corresponding to keys with assigned values.
        If dist is True, populate all edges between nodes with assinged POS attributes,
        and return the new graph.
        :param P: a dictionary of node name position pairs
        :returns: graph with connected nodes with POS attribute
        """
        G = self.to_directed()  # copy of the original object

        for name, pos in P.items():
            if name in G:
                G.nodes[name][POS] = pos

        if dist:
            G = graph_complete_edges(G, overwrite=overwrite)

        return G

    @abstractmethod
    def _pose_goal(self, T_goal: Dict[str, SEMatrix]) -> Dict[str, ArrayLike]:
        raise NotImplementedError

    def from_pose(self, T_goal: Union[SEMatrix, Dict[str, SEMatrix]]) -> nx.DiGraph:
        """
        Given a dictionary of node name and pose key-value pairs,
        generate a copy of the problem graph and fill the POS attributes of nodes
        such that it represents the induced IK problem.
        """
        if isinstance(T_goal, (SE2Matrix, SE3Matrix)):
            T_goal = {self.robot.end_effectors[0]: T_goal}

        return self.from_pos(self._pose_goal(T_goal))

    def add_anchor_node(self, name: str, data: Dict[str, Any]):
        """
        Adds a node with a known position to the problem graph, connecting
        it to all other nodes with known positions.
        :param name: name of the anchor node
        :param data: attributes of the anchor node
        """
        if POS not in data:
            raise KeyError("Node needs to gave a position to be added.")

        self.add_nodes_from([(name, data)])
        for nname, ndata in self.nodes(data=True):
            if POS in ndata and nname != name:
                self.add_edge(nname, name)
                self[nname][name][DIST] = la.norm(ndata[POS] - data[POS])
                self[nname][name][LOWER] = la.norm(ndata[POS] - data[POS])
                self[nname][name][UPPER] = la.norm(ndata[POS] - data[POS])
                self[nname][name][BOUNDED] = []

    def add_spherical_obstacle(self, name: str, position: ArrayLike, radius: float):
        # Add a fixed node representing the obstacle to the graph
        self.add_anchor_node(name, {POS: position, TYPE: OBSTACLE})

        # Set lower (and upper) distance limits to robot nodes
        for node, node_type in self.nodes(data=TYPE):
            if node_type == ROBOT and node[0] == MAIN_PREFIX:
                self.add_edge(node, name)
                self[node][name][BOUNDED] = [BELOW]
                self[node][name][LOWER] = radius
                self[node][name][UPPER] = 100

    def clear_obstacles(self):
        # Clears all obstacles from the graph
        node_types = nx.get_node_attributes(self, TYPE)
        obstacles = [node for node, typ in node_types.items() if typ == OBSTACLE]
        self.remove_nodes_from(obstacles)

    def check_distance_limits(
        self, G: nx.DiGraph, tol=1e-10
    ) -> List[Dict[str, List[Any]]]:
        """Given a graph of the same """
        typ = nx.get_node_attributes(self, name=TYPE)
        broken_limits = []
        for u, v, data in self.edges(data=True):
            if BELOW in data[BOUNDED] or ABOVE in data[BOUNDED]:
                if G[u][v][DIST] < data[LOWER] - tol:
                    broken_limit = {}
                    if (typ[u] == ROBOT and typ[v] == OBSTACLE) or (
                        typ[u] == OBSTACLE and typ[v] == ROBOT
                    ):
                        broken_limit["edge"] = (u, v)
                        broken_limit["value"] = G[u][v][DIST] - data[LOWER]
                        broken_limit["type"] = OBSTACLE
                        broken_limit["side"] = LOWER
                        broken_limits += [broken_limit]
                    if typ[u] == ROBOT and typ[v] == ROBOT:
                        broken_limit["edge"] = (u, v)
                        broken_limit["value"] = G[u][v][DIST] - data[LOWER]
                        broken_limit["type"] = "joint"
                        broken_limit["side"] = LOWER
                        broken_limits += [broken_limit]
                if G[u][v][DIST] > data[UPPER] + tol:
                    broken_limit = {}
                    if (typ[u] == ROBOT and typ[v] == OBSTACLE) or (
                        typ[u] == OBSTACLE and typ[v] == ROBOT
                    ):
                        broken_limit["edge"] = (u, v)
                        broken_limit["value"] = G[u][v][DIST] - data[UPPER]
                        broken_limit["type"] = OBSTACLE
                        broken_limit["side"] = UPPER
                        broken_limits += [broken_limit]
                    if typ[u] == ROBOT and typ[v] == ROBOT:
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
        n_nodes = self.number_of_nodes()
        L = np.zeros([n_nodes, n_nodes])  # fake distance matrix
        U = np.zeros([n_nodes, n_nodes])  # fake distance matrix
        for e1, e2, data in self.edges(data=True):
            if BOUNDED in data:
                udx = self.node_ids.index(e1)
                vdx = self.node_ids.index(e2)
                if BELOW in data[BOUNDED]:
                    L[udx, vdx] = data[LOWER] ** 2
                    L[vdx, udx] = L[udx, vdx]
                if ABOVE in data[BOUNDED]:
                    U[udx, vdx] = data[UPPER] ** 2
                    U[vdx, udx] = U[udx, vdx]
        return L, U
