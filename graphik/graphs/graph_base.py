import networkx as nx
import numpy as np
import numpy.linalg as la

from abc import ABC
from liegroups.numpy import SE3
from numpy.linalg import norm
from graphik.robots.robot_base import Robot, RobotRevolute
from graphik.utils.utils import (
    list_to_variable_dict,
    rotZ,
    trans_axis,
)
from numpy import cos, pi, sin, sqrt, arctan2

LOWER = "lower_limit"
UPPER = "upper_limit"
BOUNDED = "bounded"
DIST = "weight"
POS = "pos"
ROOT = "p0"
UNDEFINED = None


# TODO add global constants for names of attributes
class Graph(ABC):
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
        :returns: Object of the abstract class Robot
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
    def node_ids(self) -> list:
        """
        :returns: List of nodes in this graph.
        """
        return list(self.directed.nodes())

    def realization(self, x: np.array) -> nx.DiGraph:
        """
        Given a set of decision variables x, return a graph realization in R^dim.
        :param x: Decision variables (revolute joints, prismatic joints)
        :returns: Graph with node locations stored in the [POS]
        atribute and edge weights corresponding to distances between the nodes.
        """
        # NOTE do we need to work with directed graphs?
        raise NotImplementedError

    def distance_matrix(self, x: np.array) -> np.ndarray:
        """
        Given a set of decision variables x, return a matrix whose element
        [idx,jdx] corresponds to the squared distance between nodes idx and jdx.
        :param x: Decision variables (revolute joints, prismatic joints)
        :returns: Matrix of squared distances
        """
        D = nx.to_numpy_array(self.realization(x)) ** 2
        return D + D.T

    def distance_matrix_from_graph(self, G: nx.DiGraph) -> np.ndarray:
        selected_edges = [(u, v) for u, v, d in G.edges(data=True) if DIST in d]
        return (
            nx.to_numpy_array(
                nx.to_undirected(G.edge_subgraph(selected_edges)), weight=DIST
            )
            ** 2
        )

    def adjacency_matrix(self, G: nx.DiGraph = None) -> np.ndarray:
        """
        Returns the adjacency matrix representing the edges that are known,
        given the kinematic and base structure, as well as the end-effector targets.
        :returns: Adjacency matrix
        """
        if G is None:
            G = self.directed

        selected_edges = [(u, v) for u, v, d in G.edges(data=True) if DIST in d]
        return nx.to_numpy_array(
            nx.to_undirected(G.edge_subgraph(selected_edges)), weight=""
        )

    def pos_from_graph(self, G: nx.DiGraph) -> np.ndarray:
        """
        Returns an n x m matrix of node positions from a given graph,
        where n is the number of nodes and m is the point dimension.
        :param G: graph where all nodes have a populated POS field
        :returns: n x m matrix of node positions
        """

        X = np.zeros([len(G), self.dim])  # matrix of vertex positions
        for idx, name in enumerate(G):
            X[idx, :] = G.nodes[name][POS]
        return X

    def graph_from_pos(self, P: dict) -> nx.DiGraph:
        """
        NOTE: should be external method really
        Generates an nx.DiGraph object of the subclass type given
        an n x m matrix where n is the number of nodes and m is the dimension.
        Connects all graph nodes.
        :param P: n x m matrix of node positions
        :returns: graph where all nodes have a populated POS field + edges
        """

        G = nx.empty_graph(self.node_ids, create_using=nx.DiGraph)
        for idx, name in enumerate(self.node_ids):
            G.nodes[name][POS] = P[idx, :]
        return self.complete_edges(G)

    def complete_from_pos(self, P: dict, from_empty=False) -> nx.DiGraph:
        """
        Given a dictionary of node name and position key-value pairs,
        generate a copy of the problem graph and fill the POS attributes of
        nodes corresponding to keys with assigned values.
        Then, populate all edges between nodes with assinged POS attributes.
        :param P: a dictionary of node name position pairs
        :returns: graph with connected nodes with POS attribute
        """
        if not from_empty:
            G = self.directed.copy()
        else:
            G = nx.empty_graph(self.node_ids, create_using=nx.DiGraph)

        for name, pos in P.items():
            if name in G.nodes():
                G.nodes[name][POS] = pos

        return self.complete_edges(G)

    @staticmethod
    def complete_edges(G: nx.DiGraph) -> nx.DiGraph:
        """
        Given a graph with all defined node positions, calculate all unknown edges.
        :param G: Graph with some unknown edges
        :returns: Graph with all known edges
        """

        for idx, u in enumerate(G.nodes()):
            for jdx, v in enumerate(G.nodes()):
                # if both nodes have known positions
                if (POS in G.nodes[u]) and (POS in G.nodes[v]) and jdx > idx:
                    # if a distance edge exists already in the other direction
                    if G.has_edge(v, u):
                        if DIST in G[v][u]:
                            continue
                    d = norm(G.nodes[u][POS] - G.nodes[v][POS])
                    G.add_edges_from(
                        [
                            (u, v, {DIST: d}),
                            (u, v, {LOWER: d}),
                            (u, v, {UPPER: d}),
                        ]
                    )

        return G

    def distance_bounds(self, G: nx.DiGraph) -> tuple:
        """
        Given a graph with some edges containing upper and lower bounds on distance,
        calculates approximation on lower and upper bounds on all distance matrix elements.
        Distances known exactly correspond to equal lower and upper limits.

        "Distance Geometry Theory, Algorithms and Chemical Applications", Havel, 2002.
        """

        # Generate bipartite graph from two copies of G
        H = nx.DiGraph()

        for u, v, d in G.edges(data=True):
            H.add_edge(u, f"{u}s", weight=0)
            H.add_edge(v, f"{v}s", weight=0)
            H.add_edge(u, f"{v}s", weight=-G[u][v][LOWER])
            H.add_edge(v, f"{u}s", weight=-G[u][v][LOWER])
            H.add_edge(u, v, weight=G[u][v][UPPER])
            H.add_edge(v, u, weight=G[u][v][UPPER])
            H.add_edge(f"{u}s", f"{v}s", weight=G[u][v][UPPER])
            H.add_edge(f"{v}s", f"{u}s", weight=G[u][v][UPPER])

        # Find all shortest paths in bipirtatie graph
        bounds = dict(nx.all_pairs_bellman_ford_path_length(H, weight=DIST))

        N = len(G)
        lower_bounds = np.zeros([N, N])
        upper_bounds = np.zeros([N, N])

        ids = self.node_ids
        for u in G:
            for v in G:
                if bounds[u][v + "s"] < 0:
                    lower_bounds[ids.index(u), ids.index(v)] = -bounds[u][v + "s"]
                else:
                    lower_bounds[ids.index(u), ids.index(v)] = 0
                    # lower_bounds[ids.index(u), ids.index(v)] = bounds[u][f"{v}s"]
                upper_bounds[ids.index(u), ids.index(v)] = bounds[u][v]

        return lower_bounds, upper_bounds

    def distance_bound_matrix(self):
        """
        Generates a matrix of distance bounds induced by joint variables.
        """
        X = np.zeros([self.n_nodes, self.n_nodes])  # fake distance matrix
        G = self.directed
        for val in self.robot.limit_edges:
            udx = self.node_ids.index(val[0])
            vdx = self.node_ids.index(val[1])
            X[udx, vdx] = G[val[0]][val[1]][LOWER] ** 2
            X[vdx, udx] = X[udx, vdx]
        return X

    def distance_bound_matrices(self):
        """
        Generates a matrices of distance bounds induced by joint variables.
        """
        L = np.zeros([self.n_nodes, self.n_nodes])  # fake distance matrix
        U = np.zeros([self.n_nodes, self.n_nodes])  # fake distance matrix
        G = self.directed
        for val in self.robot.limit_edges:
            udx = self.node_ids.index(val[0])
            vdx = self.node_ids.index(val[1])
            if "below" in G[val[0]][val[1]][BOUNDED]:
                L[udx, vdx] = G[val[0]][val[1]][LOWER] ** 2
                L[vdx, udx] = L[udx, vdx]
            if "above" in G[val[0]][val[1]][BOUNDED]:
                U[udx, vdx] = G[val[0]][val[1]][UPPER] ** 2
                U[vdx, udx] = U[udx, vdx]
        return L, U

    def parent_node_id(self, node_id):
        """
        Returns the id of the parent node of node_id. Returns None for the root node id (0).
        :param node_id:
        :returns:
        """
        assert node_id in self.directed.nodes(), "Node {:} does not exist!".format(
            node_id
        )
        return tuple(self.robot.structure.predecessors(node_id))[0]


class SphericalRobotGraph(Graph):
    def __init__(
        self,
        robot: Robot,
    ):
        self.dim = robot.dim
        self.robot = robot
        self.structure = robot.structure
        self.base = self.base_subgraph()
        self.directed = nx.compose(self.base, self.structure)
        self.directed = nx.freeze(self.root_angle_limits(self.directed))
        super(SphericalRobotGraph, self).__init__()

    def root_angle_limits(self, G: nx.DiGraph) -> nx.DiGraph:
        if self.dim == 2:
            ax = "x"
        else:
            ax = "z"

        S = self.robot.structure
        l1 = norm(G.nodes[ax][POS])
        for node in S.successors(ROOT):
            if DIST in S[ROOT][node]:
                l2 = S[ROOT][node][DIST]
                lb = self.robot.lb[node]
                ub = self.robot.ub[node]
                lim = max(abs(ub), abs(lb))

                if self.dim == 2:

                    # Assumes bounds are less than pi in magnitude
                    G.add_edge(ax, node)
                    G[ax][node][UPPER] = l1 + l2
                    G[ax][node][LOWER] = sqrt(
                        l1 ** 2 + l2 ** 2 - 2 * l1 * l2 * cos(pi - lim)
                    )
                    G[ax][node][BOUNDED] = "below"
                    self.robot.limit_edges.append([ax, node])

                if self.dim == 3:
                    # Assumes bounds are less than pi in magnitude
                    G.add_edge(ax, node)
                    G[ax][node][UPPER] = l1 + l2
                    G[ax][node][LOWER] = sqrt(
                        l1 ** 2 + l2 ** 2 - 2 * l1 * l2 * cos(pi - lim)
                    )
                    G[ax][node][BOUNDED] = "below"
                    self.robot.limit_edges.append([ax, node])

        return G

    @staticmethod
    def base_subgraph_2d() -> nx.DiGraph:
        base = nx.DiGraph([("p0", "x"), ("p0", "y"), ("x", "y")])

        # Invert x axis because of the way joint limits are set up, makes no difference
        base.add_nodes_from(
            [
                ("p0", {POS: np.array([0, 0])}),
                # ("x", {POS: np.array([1, 0])}),
                ("x", {POS: np.array([-1, 0])}),
                ("y", {POS: np.array([0, 1])}),
            ]
        )

        for u, v in base.edges():
            base[u][v][DIST] = la.norm(base.nodes[u][POS] - base.nodes[v][POS])
            base[u][v][LOWER] = base[u][v][DIST]
            base[u][v][UPPER] = base[u][v][DIST]

        return base

    @staticmethod
    def base_subgraph_3d() -> nx.DiGraph:
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
                ("x", {POS: np.array([1, 0, 0])}),
                ("y", {POS: np.array([0, 1, 0])}),
                # ("z", {POS: np.array([0, 0, 1])}),
                ("z", {POS: np.array([0, 0, -1])}),
            ]
        )
        for u, v in base.edges():
            base[u][v][DIST] = la.norm(base.nodes[u][POS] - base.nodes[v][POS])
            base[u][v][LOWER] = base[u][v][DIST]
            base[u][v][UPPER] = base[u][v][DIST]
        return base

    def base_subgraph(self) -> nx.DiGraph:
        if self.dim == 2:
            return self.base_subgraph_2d()
        else:
            return self.base_subgraph_3d()

    def realization(self, x: dict) -> nx.DiGraph:
        """
        Given a dictionary of joint variables generate a representative graph.
        This graph will be fully conncected.
        """
        P = {}
        for name in self.robot.structure:  # TODO make single for loop
            P[name] = self.robot.get_pose(x, name).trans

        for name in self.base:
            P[name] = self.base.nodes[name][POS]

        return self.complete_from_pos(P, from_empty=True)

    def complete_from_pose_goal(self, pose_goals):
        pos = {}
        for ee in self.robot.end_effectors:
            T_goal = pose_goals[ee[0]]  # first ID is the last point in chain
            d = self.directed[ee[1]][ee[0]][DIST]
            z = T_goal.rot.as_matrix()[0:3, -1]
            pos[ee[0]] = T_goal.trans
            pos[ee[1]] = T_goal.trans - z * d

        return self.complete_from_pos(pos)


class Revolute3dRobotGraph(Graph):
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
        super(Revolute3dRobotGraph, self).__init__()

    def root_angle_limits(self, G: nx.DiGraph) -> nx.DiGraph:
        axis_length = self.robot.axis_length
        T = self.robot.T_zero
        T1 = T["p0"]
        base_names = ["x", "y"]
        names = ["p1", "q1"]
        transZ = trans_axis(axis_length, "z")

        for base_node in base_names:
            for node in names:
                T0 = SE3.from_matrix(np.identity(4))
                T0.trans = G.nodes[base_node][POS]
                if node[0] == "p":
                    d_max, d_min, limit = self.robot.max_min_distance(T0, T1, T["p1"])
                else:
                    d_max, d_min, limit = self.robot.max_min_distance(
                        T0, T1, T["p1"].dot(transZ)
                    )

                if limit:
                    if node[0] == "p":
                        T_rel = T1.inv().dot(T["p1"])
                    else:
                        T_rel = T1.inv().dot(T["p1"].dot(transZ))

                    d_limit = norm(
                        T1.dot(rotZ(self.robot.ub["p1"])).dot(T_rel).trans - T0.trans
                    )

                    if limit == "above":
                        d_max = d_limit
                    else:
                        d_min = d_limit
                    self.robot.limited_joints += ["p1"]  # joint at p0 is limited
                    self.robot.limit_edges += [[base_node, node]]  # TODO remove/fix

                G.add_edge(base_node, node)

                if d_max == d_min:
                    G[base_node][node][DIST] = d_max

                G[base_node][node][UPPER] = d_max
                G[base_node][node][LOWER] = d_min
                G[base_node][node][BOUNDED] = limit
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
                ("x", {POS: np.array([axis_length, 0, 0])}),
                ("y", {POS: np.array([0, -axis_length, 0])}),
                ("q0", {POS: np.array([0, 0, axis_length])}),
            ]
        )
        for u, v in base.edges():
            base[u][v][DIST] = norm(base.nodes[u][POS] - base.nodes[v][POS])
            base[u][v][LOWER] = base[u][v][DIST]
            base[u][v][UPPER] = base[u][v][DIST]
        return base

    def realization(self, x: np.ndarray) -> np.ndarray:
        """
        Given a dictionary of joint variables generate a representative graph.
        This graph will be fully conncected.
        """
        [dim, n_nodes, base, struct, ids, axis_length] = [
            self.dim,
            self.n_nodes,
            self.base,
            self.robot.structure,
            self.node_ids,
            self.robot.axis_length,
        ]
        # s = flatten([[0], self.robot.s])  # b/c we're starting from root
        X = np.zeros([n_nodes, dim])
        X[: dim + 1, :] = self.pos_from_graph(base)
        for idx in range(len(x)):  # get node locations
            if type(x) is not dict:
                T = self.robot.get_pose(list_to_variable_dict(x), "p" + str(idx + 1))
            else:
                T = self.robot.get_pose(x, "p" + str(idx + 1))
            X[ids.index(f"p{idx+1}"), :] = T.trans
            X[ids.index(f"q{idx+1}"), :] = T.dot(trans_axis(axis_length, "z")).trans
        return self.graph_from_pos(X)

    def distance_bounds_from_sampling(self):
        robot = self.robot
        G = self.directed
        ids = self.node_ids
        q_rand = robot.random_configuration()
        D_min = self.distance_matrix(q_rand)
        D_max = self.distance_matrix(q_rand)

        for _ in range(2000):
            q_rand = robot.random_configuration()
            D_rand = self.distance_matrix(q_rand)
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
