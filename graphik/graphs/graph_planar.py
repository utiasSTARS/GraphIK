from typing import Dict, List, Union, Any
import numpy as np
import numpy.linalg as la
from graphik.robots import RobotPlanar
from graphik.graphs.graph_base import RobotGraph
from graphik.utils import *
from liegroups.numpy import SE2, SO2
import networkx as nx
from numpy import cos, pi, sqrt


class RobotPlanarGraph(RobotGraph):
    def __init__(self, robot: RobotPlanar, params: Dict = {}):

        self.dim = 2
        self.robot = robot
        self.axis_length = params.get("axis_length", 1)  # distance between p and q

        self.base = self.base_subgraph()
        self.structure = self.structure_graph()
        self.set_limits()

        self.directed = nx.compose(self.base, self.structure)
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
            base[u][v][BOUNDED] = []

        return base

    def structure_graph(self):
        robot = self.robot
        end_effectors = self.robot.end_effectors
        kinematic_map = self.robot.kinematic_map

        structure = nx.empty_graph(create_using=nx.DiGraph)

        for ee in end_effectors:
            k_map = kinematic_map[ROOT][ee]
            for idx in range(len(k_map)):
                cur = k_map[idx]
                cur_pos = robot.nodes[cur]["T0"].trans

                # Add nodes for joint and edge between them
                structure.add_nodes_from([(cur, {POS: cur_pos})])

                # If there exists a preceeding joint, connect it to new
                if idx != 0:
                    pred = k_map[idx - 1]
                    dist = la.norm(
                        structure.nodes[cur][POS] - structure.nodes[pred][POS]
                    )
                    structure.add_edge(
                        pred,
                        cur,
                        **{DIST: dist, LOWER: dist, UPPER: dist, BOUNDED: []},
                    )

        # Delete positions used for weights
        for u in structure.nodes:
            del structure.nodes[u][POS]

        # Set node type to robot
        nx.set_node_attributes(structure, ROBOT, TYPE)

        return structure

    def root_angle_limits(self, G: nx.DiGraph) -> nx.DiGraph:
        ax = "x"

        S = self.structure
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

    def set_limits(self):
        """
        Sets known bounds on the distances between joints.
        This is induced by link length and joint limits.
        """
        S = self.structure
        for u in S:
            # direct successors are fully known
            for v in (suc for suc in S.successors(u) if suc):
                S[u][v]["upper_limit"] = S[u][v][DIST]
                S[u][v]["lower_limit"] = S[u][v][DIST]
            for v in (des for des in level2_descendants(S, u) if des):
                ids = self.robot.kinematic_map[u][v]  # TODO generate this at init
                l1 = self.robot.l[ids[1]]
                l2 = self.robot.l[ids[2]]
                lb = self.robot.lb[ids[2]]  # symmetric limit
                ub = self.robot.ub[ids[2]]  # symmetric limit
                lim = max(abs(ub), abs(lb))
                S.add_edge(u, v)
                S[u][v]["upper_limit"] = l1 + l2
                S[u][v]["lower_limit"] = sqrt(
                    l1 ** 2 + l2 ** 2 - 2 * l1 * l2 * cos(pi - lim)
                )
                S[u][v][BOUNDED] = "below"

    def realization(self, joint_angles: Dict[str, float]) -> nx.DiGraph:
        """
        Given a dictionary of joint variables generate a representative graph.
        This graph will be fully conncected.
        """
        P = {}
        for name in self.structure:  # TODO make single for loop
            P[name] = self.get_pose(joint_angles, name).trans

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

    def joint_variables(self, G: nx.Graph) -> Dict[str, float]:
        """
        Finds the set of decision variables corresponding to the
        graph realization G.

        :param G: networkx.DiGraph with known vertex positions
        :returns: array of joint variables t
        :rtype: np.ndarray
        """
        R = {"p0": SO2.identity()}
        joint_variables = {}

        for u, v, dat in self.structure.edges(data=DIST):
            if dat:
                diff_uv = G.nodes[v][POS] - G.nodes[u][POS]
                len_uv = np.linalg.norm(diff_uv)
                sol = np.linalg.solve(len_uv * R[u].as_matrix(), diff_uv)
                theta_idx = np.math.atan2(sol[1], sol[0])
                joint_variables[v] = wraptopi(theta_idx)
                Rz = SO2.from_angle(theta_idx)
                R[v] = R[u].dot(Rz)

        return joint_variables

    def get_pose(
        self, joint_angles: Dict[str, float], query_node: Union[List[str], str]
    ) -> Union[Dict[str, SE2], SE2]:
        return self.robot.pose(joint_angles, query_node)
