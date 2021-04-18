from typing import Dict, List, Union, Any
from numpy.typing import ArrayLike

import networkx as nx
import numpy as np
from graphik.robots.robot_base import Robot
from graphik.utils.constants import *
from graphik.utils.kinematics import fk_2d
from graphik.utils.utils import level2_descendants, wraptopi
from liegroups.numpy import SE2, SO2
from numpy import cos, pi, sqrt, inf


class RobotPlanar(Robot):
    def __init__(self, params):
        self.dim = 2
        self.T_base = params.get("T_base", SE2.identity())
        self.a = params["a"]
        self.th = params["theta"]
        self.n = len(self.a)
        self.joint_ids = [f"p{idx}" for idx in range(self.n)]
        self.lb = params.get(
            "joint_limits_lower", dict(zip(self.joint_ids, self.n * [-pi]))
        )
        self.ub = params.get(
            "joint_limits_upper", dict(zip(self.joint_ids, self.n * [pi]))
        )

        self.parents = params.get(
            "parents", {f"p{idx}": [f"p{idx+1}"] for idx in range(self.n)}
        )
        self.kinematic_map = nx.shortest_path(nx.from_dict_of_lists(self.parents))

        self.generate_structure_graph()
        self.set_limits()

        super(RobotPlanar, self).__init__()

    def chain_graph(self) -> nx.DiGraph:
        """
        Directed graph representing the robots chain structure
        """
        edg_lst = [
            (f"p{idx}", f"p{idx+1}", self.a[f"p{idx+1}"]) for idx in range(self.n)
        ]
        chain_graph = nx.DiGraph()
        chain_graph.add_weighted_edges_from(edg_lst)
        nx.set_node_attributes(chain_graph, "robot", TYPE)
        nx.set_edge_attributes(chain_graph, [], BOUNDED)
        return chain_graph

    def tree_graph(self, parents: dict) -> nx.DiGraph:
        """
        Needed for forward kinematics (computing the shortest path).
        :return: Directed graph representing the robot's tree structure.
        """
        tree_graph = nx.DiGraph(parents)
        for parent, child in tree_graph.edges():
            tree_graph.edges[parent, child][DIST] = self.a[child]
        nx.set_node_attributes(tree_graph, "robot", TYPE)
        nx.set_edge_attributes(tree_graph, [], BOUNDED)
        return tree_graph

    def generate_structure_graph(self):
        self.structure = self.tree_graph(self.parents)

    @property
    def end_effectors(self) -> List[Any]:
        """
        Returns the names of end effector nodes and the nodes
        preceeding them (required for orientation goals) as
        a list of lists.
        """
        if not hasattr(self, "_end_effectors"):
            S = self.structure
            self._end_effectors = [
                [x, y]
                for x in S
                if S.out_degree(x) == 0
                for y in S.predecessors(x)
                if DIST in S[y][x]
                if S[y][x][DIST] < inf
            ]

        return self._end_effectors

    def get_pose(self, node_inputs: Dict[str, float], query_node: str) -> SE2:
        """
        Returns an SE2 element corresponding to the location
        of the query_node in the configuration determined by
        node_inputs.
        """
        if query_node == "p0":
            return SE2.identity()

        path_nodes = self.kinematic_map["p0"][query_node][1:]
        q = np.asarray([node_inputs[node] for node in path_nodes])
        a = np.asarray([self.a[node] for node in path_nodes])
        th = np.asarray([self.th[node] for node in path_nodes])
        return fk_2d(a, th, q)

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
                ids = self.kinematic_map[u][v]  # TODO generate this at init
                l1 = self.a[ids[1]]
                l2 = self.a[ids[2]]
                lb = self.lb[ids[2]]  # symmetric limit
                ub = self.ub[ids[2]]  # symmetric limit
                lim = max(abs(ub), abs(lb))
                S.add_edge(u, v)
                S[u][v]["upper_limit"] = l1 + l2
                S[u][v]["lower_limit"] = sqrt(
                    l1 ** 2 + l2 ** 2 - 2 * l1 * l2 * cos(pi - lim)
                )
                S[u][v][BOUNDED] = "below"

    def random_configuration(self) -> Dict[str, float]:
        q = {}
        for key in self.structure:
            if key != "p0":
                q[key] = self.lb[key] + (self.ub[key] - self.lb[key]) * np.random.rand()
        return q

    def jacobian(
        self, joint_angles: Dict[str, float], nodes: Union[List[str], str] = None
    ) -> Dict[str, ArrayLike]:
        kinematic_map = self.kinematic_map["p0"]  # get map to all nodes from root

        # find end-effector nodes #FIXME so much code
        if nodes is None:
            nodes = []
            for ee in self.end_effectors:  # get p nodes in end-effectors
                nodes += [ee]

        Ts = self.get_all_poses(joint_angles)  # get all frame poses
        J = {}
        for node in nodes:  # iterate through end-effector nodes
            path = kinematic_map[node][1:]  # find nodes for joints to end-effector
            T_0_n = Ts[node].as_matrix()  # ee frame
            p_ee = T_0_n[0:2, -1]
            J_i = []
            for joint in path:  # algorithm fills Jac per column
                T_0_i = Ts[list(self.parents.predecessors(joint))[0]].as_matrix()
                p_i = T_0_i[:2, -1]
                e = p_i - p_ee
                J_i += [np.hstack([e[1], -e[0], 1])]
            J[node] = np.zeros([3, self.n])
            J[node][:, : len(J_i)] = np.column_stack(J_i)
        return J

    def jacobian_cost(self, joint_angles: dict, ee_goals) -> np.ndarray:
        """
        Calculate the planar robot's Jacobian with respect to the Euclidean squared cost function.
        """
        kinematic_map = self.kinematic_map["p0"]  # get map to all nodes from root
        end_effector_nodes = ee_goals.keys()
        J = np.zeros(self.n)
        for (
            ee
        ) in end_effector_nodes:  # iterate through end-effector nodes, assumes sorted
            ee_path = kinematic_map[ee][
                1:
            ]  # [:-1]  # no last node, only phys. joint locations
            t_ee = self.get_pose(joint_angles, ee).trans
            dg_ee_x = t_ee[0] - ee_goals[ee].trans[0]
            dg_ee_y = t_ee[1] - ee_goals[ee].trans[1]
            for (pdx, joint_p) in enumerate(ee_path):  # algorithm fills Jac per column
                p_idx = int(joint_p[1:]) - 1
                for jdx in range(pdx, len(ee_path)):
                    node_jdx = ee_path[jdx]
                    theta_jdx = sum([joint_angles[key] for key in ee_path[0 : jdx + 1]])
                    J[p_idx] += (
                        2.0
                        * self.a[node_jdx]
                        * (-dg_ee_x * np.sin(theta_jdx) + dg_ee_y * np.cos(theta_jdx))
                    )

        return J

    def hessian_cost(self, joint_angles: dict, ee_goals) -> np.ndarray:
        """
        Calculate the planar robot's Hessian with respect to the Euclidean squared cost function.
        """
        kinematic_map = self.kinematic_map["p0"]  # get map to all nodes from root
        end_effector_nodes = ee_goals.keys()
        H = np.zeros((self.n, self.n))
        for (
            ee
        ) in end_effector_nodes:  # iterate through end-effector nodes, assumes sorted
            ee_path = kinematic_map[ee][
                1:
            ]  # [:-1]  # no last node, only phys. joint locations
            t_ee = self.get_pose(joint_angles, ee).trans
            dg_ee_x = t_ee[0] - ee_goals[ee].trans[0]
            dg_ee_y = t_ee[1] - ee_goals[ee].trans[1]
            for (pdx, joint_p) in enumerate(ee_path):  # algorithm fills Hess per column
                p_idx = int(joint_p[1:]) - 1
                sin_p_term = 0.0
                cos_p_term = 0.0
                for jdx in range(pdx, len(ee_path)):
                    node_jdx = ee_path[jdx]
                    theta_jdx = sum([joint_angles[key] for key in ee_path[0 : jdx + 1]])
                    sin_p_term += self.a[node_jdx] * np.sin(theta_jdx)
                    cos_p_term += self.a[node_jdx] * np.cos(theta_jdx)

                for (qdx, joint_q) in enumerate(
                    ee_path[pdx:]
                ):  # TODO: check if starting from pdx works
                    qdx = qdx + pdx
                    q_idx = int(joint_q[1:]) - 1
                    sin_q_term = 0.0
                    cos_q_term = 0.0
                    for kdx in range(qdx, len(ee_path)):
                        node_kdx = ee_path[kdx]
                        theta_kdx = sum(
                            [joint_angles[key] for key in ee_path[0 : kdx + 1]]
                        )
                        sin_q_term += self.a[node_kdx] * np.sin(theta_kdx)
                        cos_q_term += self.a[node_kdx] * np.cos(theta_kdx)

                    # assert(q_idx >= p_idx)
                    H[p_idx, q_idx] += (
                        2.0 * sin_q_term * sin_p_term
                        - 2.0 * dg_ee_x * cos_q_term
                        + 2.0 * cos_p_term * cos_q_term
                        - 2.0 * dg_ee_y * sin_q_term
                    )

        return H + H.T - np.diag(np.diag(H))
