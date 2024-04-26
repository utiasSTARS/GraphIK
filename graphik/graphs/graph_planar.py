from typing import Dict, List, Union, Any, Optional
import numpy as np
import numpy.linalg as la
from graphik.robots import RobotPlanar
from graphik.graphs.graph_base import ProblemGraph
from graphik.utils import *
from liegroups.numpy import SE2, SO2
import networkx as nx
from numpy import cos, pi
from math import sqrt, atan2


class ProblemGraphPlanar(ProblemGraph):
    def __init__(self, robot: RobotPlanar, params: Dict = {}):
        super(ProblemGraphPlanar, self).__init__(robot, params)

        # initialize the base and structure subgraphs
        base = self.init_base_subgraph()
        structure = self.init_structure_subgraph()
        composition = nx.compose(base, structure)

        # set known node locations and distances for the problem graph
        self.add_nodes_from(composition.nodes(data=True))
        self.add_edges_from(composition.edges(data=True))

        # set distance limits corresponding to joint limits
        self.set_limits()
        self.root_angle_limits()

    def init_base_subgraph(self) -> nx.DiGraph:
        base = nx.DiGraph([("p0", "x"), ("p0", "y"), ("x", "y")])

        # Invert x axis because of the way joint limits are set up, makes no difference
        base.add_nodes_from(
            [
                ("p0", {POS: np.array([0, 0]), TYPE: [BASE, ROBOT]}),
                ("x", {POS: np.array([-1, 0]), TYPE: [BASE]}),
                ("y", {POS: np.array([0, 1]), TYPE: [BASE]}),
            ]
        )

        for u, v in base.edges():
            base[u][v][DIST] = la.norm(base.nodes[u][POS] - base.nodes[v][POS])
            base[u][v][LOWER] = base[u][v][DIST]
            base[u][v][UPPER] = base[u][v][DIST]
            base[u][v][BOUNDED] = []

        return base

    def init_structure_subgraph(self) -> nx.DiGraph:
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
                structure.add_nodes_from([(cur, {POS: cur_pos, TYPE: [ROBOT]})])
                if cur == ROOT:
                    structure.nodes[cur][TYPE] += [BASE]

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

                    if cur in self.robot.end_effectors:
                        structure.nodes[cur][TYPE] += [END_EFFECTOR]
                        structure.nodes[pred][TYPE] += [END_EFFECTOR]

        # Delete positions used for weights
        for u in structure.nodes:
            del structure.nodes[u][POS]

        return structure

    def root_angle_limits(self):
        ax = "x"

        S = self.structure
        l1 = la.norm(self.nodes[ax][POS])
        for node in S.successors(ROOT):
            if DIST in S[ROOT][node]:
                l2 = S[ROOT][node][DIST]
                lb = self.robot.lb[node]
                ub = self.robot.ub[node]
                lim = max(abs(ub), abs(lb))

                # Assumes bounds are less than pi in magnitude
                self.add_edge(ax, node)
                self[ax][node][UPPER] = l1 + l2
                self[ax][node][LOWER] = sqrt(
                    l1 ** 2 + l2 ** 2 - 2 * l1 * l2 * cos(pi - lim)
                )
                self[ax][node][BOUNDED] = BELOW

    def set_limits(self):
        """
        Sets known bounds on the distances between joints.
        This is induced by link length and joint limits.
        TODO unify with root_angle_limits
        """
        S = self.structure
        for u in S:
            # direct successors are fully known
            for v in (suc for suc in S.successors(u) if suc):
                self[u][v][UPPER] = S[u][v][DIST]
                self[u][v][LOWER] = S[u][v][DIST]
            for v in (des for des in level2_descendants(S, u) if des):
                ids = self.robot.kinematic_map[u][v]  # TODO generate this at init
                l1 = self.robot.l[ids[1]]
                l2 = self.robot.l[ids[2]]
                lb = self.robot.lb[ids[2]]  # symmetric limit
                ub = self.robot.ub[ids[2]]  # symmetric limit
                lim = max(abs(ub), abs(lb))
                self.add_edge(u, v)
                self[u][v][UPPER] = l1 + l2
                self[u][v][LOWER] = sqrt(
                    l1 ** 2 + l2 ** 2 - 2 * l1 * l2 * cos(pi - lim)
                )
                self[u][v][BOUNDED] = BELOW

    def _pose_goal(self, T_goal: Dict[str, SE2]) -> Dict[str, ArrayLike]:
        pos = {}
        for u, T_goal_u in T_goal.items():
            for v in self.structure.predecessors(u):
                if DIST in self[v][u]:
                    d = self[v][u][DIST]
                    z = T_goal_u.rot.as_matrix()[0:2, 0]
                    pos[u] = T_goal_u.trans
                    pos[v] = T_goal_u.trans - z * d
        return pos

    def joint_variables(self, G: nx.Graph, T_final: Optional[Dict[str, SE2]] = None) -> Dict[str, float]:
        """
        Finds the set of decision variables corresponding to the
        graph realization G.

        :param G: networkx.DiGraph with known vertex positions
        :returns: array of joint variables t
        :rtype: np.ndarray
        """
        joint_variables = {}

        # resolve rotation and translation
        R_, t_ = best_fit_transform(np.vstack((G.nodes[ROOT][POS],
                                               G.nodes["x"][POS],
                                               G.nodes["y"][POS])) ,
                                    np.vstack(([0,0], [-1,0], [0,1])))

        R = {ROOT: SO2.identity()}

        for u, v, dat in self.structure.edges(data=DIST):
            if dat:
                diff_uv = R_.dot(G.nodes[v][POS] - G.nodes[u][POS])
                len_uv = np.linalg.norm(diff_uv)
                sol = R[u].as_matrix().T.dot(diff_uv/len_uv)
                theta_idx = atan2(sol[1], sol[0])
                joint_variables[v] = wraptopi(theta_idx)
                Rz = SO2.from_angle(theta_idx)
                R[v] = R[u].dot(Rz)

        return joint_variables

    def get_pose(
        self, joint_angles: Dict[str, float], query_node: Union[List[str], str]
    ) -> Union[Dict[str, SE2], SE2]:
        return self.robot.pose(joint_angles, query_node)
