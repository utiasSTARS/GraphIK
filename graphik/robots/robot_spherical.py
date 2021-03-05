from typing import Dict, List, Any
import numpy as np
import networkx as nx
from graphik.robots.robot_base import Robot
from graphik.robots import RobotRevolute
from graphik.utils.constants import *
from liegroups.numpy import SE3, SO3
from graphik.utils.geometry import roty, trans_axis
from graphik.utils.kinematics import fk_3d_sph
from graphik.utils.utils import (
    level2_descendants,
    spherical_angle_bounds_to_revolute,
    wraptopi,
)
from graphik.utils.constants import *
from numpy import arctan2, cos, cross, pi, sin, sqrt
from numpy.linalg import norm


class RobotSpherical(Robot):
    def __init__(self, params):

        self.dim = 3
        self.T_base = params.get("T_base", SE3.identity())
        self.a = params["a"]
        self.al = params["alpha"]
        self.d = params["d"]
        self.th = params["theta"]
        self.n = len(self.a)  # number of links
        self.joint_ids = [f"p{idx}" for idx in range(self.n + 1)]
        self.lb = params.get(
            "joint_limits_upper", dict(zip(self.joint_ids, self.n * [-pi]))
        )
        self.ub = params.get(
            "joint_limits_lower", dict(zip(self.joint_ids, self.n * [pi]))
        )

        if "parents" in params:
            self.parents = params["parents"]
            self.structure = self.tree_graph()
        else:
            self.structure = self.chain_graph()
            self.parents = nx.to_dict_of_dicts(self.structure)

        self.kinematic_map = nx.shortest_path(self.structure.copy())
        self.set_limits()
        super(RobotSpherical, self).__init__()

    @property
    def spherical(self) -> bool:
        return True

    def chain_graph(self) -> nx.DiGraph:
        """
        Directed graph representing the robots chain structure
        """
        edg_lst = [
            (f"p{idx}", f"p{idx+1}", self.d[f"p{idx+1}"]) for idx in range(self.n)
        ]
        chain_graph = nx.DiGraph()
        chain_graph.add_weighted_edges_from(edg_lst)
        nx.set_node_attributes(chain_graph, "robot", TYPE)
        return chain_graph

    def tree_graph(self) -> nx.DiGraph:
        """
        Needed for forward kinematics (computing the shortest path).
        :return: Directed graph representing the robot's tree structure.
        """
        tree_graph = nx.DiGraph(self.parents)
        for parent, child in tree_graph.edges():
            tree_graph.edges[parent, child][DIST] = self.d[child]
        nx.set_node_attributes(tree_graph, "robot", TYPE)
        return tree_graph

    @property
    def end_effectors(self) -> list:
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
                if S[y][x][DIST] < np.inf
            ]

        return self._end_effectors

    def get_pose(self, joint_values: dict, query_node: str) -> SE3:
        """
        Returns an SE3 element corresponding to the location
        of the query_node in the configuration determined by
        node_inputs.
        """
        if query_node == "p0":
            return SE3.identity()
        path_nodes = self.kinematic_map["p0"][query_node][1:]
        q = np.array([joint_values[node][0] for node in path_nodes])
        alpha = np.array([joint_values[node][1] for node in path_nodes])
        a = np.array([self.a[node] for node in path_nodes])
        d = np.array([self.d[node] for node in path_nodes])

        return fk_3d_sph(a, alpha, d, q)

    def joint_variables(self, G: nx.Graph, T_final: dict = None) -> np.ndarray:
        """
        Finds the set of decision variables corresponding to the
        graph realization G.

        :param G: networkx.DiGraph with known vertex positions
        :returns: array of joint variables t
        :rtype: np.ndarray
        """
        R = {"p0": SO3.identity()}
        joint_variables = {}
        for u, v, dat in self.structure.edges(data=DIST):
            if dat:
                diff_uv = G.nodes[v][POS] - G.nodes[u][POS]
                len_uv = np.linalg.norm(diff_uv)

                sol = np.linalg.lstsq(len_uv * R[u].as_matrix(), diff_uv)
                sol = sol[0]

                theta_idx = np.math.atan2(sol[1], sol[0]) + pi / 2
                Rz = SO3.rotz(theta_idx)

                alpha_idx = abs(np.math.acos(min(sol[2], 1)))
                Rx = SO3.rotx(alpha_idx)

                joint_variables[v] = [wraptopi(theta_idx), alpha_idx]
                R[v] = R[u].dot(Rz.dot(Rx))

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
                S[u][v][UPPER] = S[u][v][DIST]
                S[u][v][LOWER] = S[u][v][DIST]
            for v in (des for des in level2_descendants(S, u) if des):
                ids = self.kinematic_map[u][v]
                l1 = self.d[ids[1]]
                l2 = self.d[ids[2]]
                lb = self.lb[ids[2]]
                ub = self.ub[ids[2]]
                lim = max(abs(ub), abs(lb))
                S.add_edge(u, v)
                S[u][v][UPPER] = l1 + l2
                S[u][v][LOWER] = sqrt(l1 ** 2 + l2 ** 2 - 2 * l1 * l2 * cos(pi - lim))
                S[u][v][BOUNDED] = "below"

    def random_configuration(self):
        """
        Returns a random set of joint values within the joint limits
        determined by lb and ub.
        """
        q = {}
        for key in self.structure:
            if key != "p0":
                q[key] = [
                    -pi + 2 * pi * np.random.rand(),
                    np.abs(
                        wraptopi(
                            self.lb[key]
                            + (self.ub[key] - self.lb[key]) * np.random.rand()
                        )
                    ),
                ]
        return q

    def jacobian(self, joint_angles: dict, query_frame: str = "") -> Any:
        """
        Calculate the linear velocity robot Jacobian for all end-effectors.
        TODO: make frame selectable
        """

        kinematic_map = self.kinematic_map["p0"]  # get map to all nodes from root
        end_effector_nodes = []
        for ee in self.end_effectors:  # get p nodes in end-effectors
            if ee[0][0] == "p":
                end_effector_nodes += [ee[0]]
            if ee[1][0] == "p":
                end_effector_nodes += [ee[1]]

        node_names = [
            name for name in self.structure if name[0] == "p"
        ]  # list of p node ids

        # Ts = self.get_full_pose_fast_lambdify(joint_angles)  # all frame poses
        Ts = self.get_all_poses(joint_angles)  # all frame poses
        Ts["p0"] = np.eye(4)

        J = np.zeros([0, len(node_names) - 1])
        for ee in end_effector_nodes:  # iterate through end-effector nodes
            ee_path = kinematic_map[ee][:-1]  # no last node, only phys. joint locations

            T_0_ee = Ts[ee]  # ee frame
            p_ee = T_0_ee[0:3, -1]  # ee position

            Jp_t = np.zeros([3, len(node_names) - 1])  # translation jac for theta
            Jp_al = np.zeros([3, len(node_names) - 1])  # translation jac alpha
            for joint in ee_path:  # algorithm fills Jac per column
                T_0_i = Ts[joint]
                z_hat_i = T_0_i[:3, 2]
                x_hat_i = T_0_i[:3, 0]
                p_i = T_0_i[:3, -1]
                j_idx = node_names.index(joint)
                Jp_t[:, j_idx] = np.cross(z_hat_i, p_ee - p_i)
                Jp_al[:, j_idx] = np.cross(x_hat_i, p_ee - p_i)

            J_ee = np.vstack([Jp_t, Jp_al])
            J = np.vstack([J, J_ee])  # stack big jac for multiple ee

        return J

    def to_revolute(self):
        if len(self.end_effectors) > 1:
            return self.to_revolute_tree()
        else:
            return self.to_revolute_chain()

    def to_revolute_tree(self):
        """
        Convert to a revolute tree representation (for local solver).
        :return:
        """
        T_zero = {"p0": SE3.identity()}
        stack = ["p0"]
        tree_structure = {"p0": []}
        ang_lims_map = {}
        old_to_new_names = {
            "p0": "p0"
        }  # Returned for user of the method (to map old joint names to new ones)
        ub, lb = spherical_angle_bounds_to_revolute(self.ub, self.lb)
        count = 1
        while len(stack) > 0:
            joint = stack.pop(0)
            new_joint = old_to_new_names[joint]
            for child in self.parents[joint]:
                stack += [child]
                new_child = "p" + str(count)
                count += 1
                # ub[new_child] = self.ub[child]
                # lb[new_child] = self.lb[child]
                ang_lims_map[child] = new_child
                tree_structure[new_joint] += [new_child]
                new_grand_child = "p" + str(count)
                count += 1
                old_to_new_names[child] = new_grand_child
                tree_structure[new_child] = [new_grand_child]
                Ry = SE3(SO3(roty(np.pi / 2)), np.zeros(3))
                T_zero[new_child] = T_zero[new_joint].dot(Ry)
                d = self.d[child]
                Ry_back = SE3(SO3(roty(-np.pi / 2)), np.zeros(3))
                T_zero[new_grand_child] = (
                    T_zero[new_child].dot(Ry_back).dot(trans_axis(d, "z"))
                )
                tree_structure[new_grand_child] = []

        # for key in old_to_new_names:
        #     if key in self.ub.keys():
        #         ub[old_to_new_names[key]] = self.ub[key]
        #         lb[old_to_new_names[key]] = self.lb[key]

        # for key in T_zero:
        #     if key not in ub.keys() and key is not 'p0':
        #         ub[key] = np.pi
        #         lb[key] = -np.pi

        params = {"T_zero": T_zero, "ub": ub, "lb": lb, "parents": tree_structure}

        # print("normal ub: {:}".format(self.ub))
        # print("ub: {:}".format(ub))
        # print("lb: {:}".format(lb))
        return RobotRevolute(params), old_to_new_names, ang_lims_map

    def to_revolute_chain(self):
        """
        Convert to a revolute chain representation (for local solver).
        :return:
        """
        T_zero = {"p0": SE3.identity()}
        ang_lims_map = {}
        old_to_new_names = {
            "p0": "p0"
        }  # Returned for user of the method (to map old joint names to new ones)
        ub, lb = spherical_angle_bounds_to_revolute(self.ub, self.lb)
        count = 1
        joint_prev = "p0"
        for (
            joint
        ) in self.d:  # Assumes the dictionary is in chain order (perhaps enforce?)
            new_node1 = "p" + str(count)
            count += 1
            # ub[new_node1] = self.ub[joint]
            # lb[new_node1] = self.lb[joint]
            ang_lims_map[joint] = new_node1

            new_node2 = "p" + str(count)
            count += 1
            old_to_new_names[joint] = new_node2

            Ry = SE3(SO3(roty(np.pi / 2)), np.zeros(3))
            T_zero[new_node1] = T_zero[joint_prev].dot(Ry)
            d = self.d[joint]
            Ry_back = SE3(SO3(roty(-np.pi / 2)), np.zeros(3))
            T_zero[new_node2] = T_zero[new_node1].dot(Ry_back).dot(trans_axis(d, "z"))

            joint_prev = new_node2

        # for key in T_zero:
        #     if key not in ub.keys() and key is not 'p0':
        #         ub[key] = np.pi
        #         lb[key] = -np.pi

        params = {"T_zero": T_zero, "ub": ub, "lb": lb}
        return RobotRevolute(params), old_to_new_names, ang_lims_map
