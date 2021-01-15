from abc import ABC, abstractmethod

import networkx as nx
import numpy as np
import sympy as sp
from graphik.utils.geometry import cross_symb, rot_axis, roty, skew, trans_axis
from graphik.utils.kinematics import fk_2d, fk_3d, fk_3d_sph, modified_fk_3d
from graphik.utils.utils import (
    flatten,
    level2_descendants,
    list_to_variable_dict,
    spherical_angle_bounds_to_revolute,
    wraptopi,
)
from graphik.utils.constants import *
from liegroups.numpy import SE2, SE3, SO2, SO3
from liegroups.numpy._base import SEMatrixBase
from numpy import arctan2, cos, cross, pi, sin, sqrt
from numpy.linalg import norm


class Robot(ABC):
    """
    Describes the kinematic parameters for a robot whose joints and links form a tree (no loops like in parallel
    mechanisms).
    """

    def __init__(self):
        self.lambdified = False

    @abstractmethod
    def get_pose(self, node_inputs: dict, query_node: str):
        """Given a list of N joint variables, calculate the Nth joint's pose.

        :param node_inputs: joint variables node names as keys mapping to values
        :param query_node: node ID of node whose pose we want
        :returns: SE2 or SE3 pose
        :rtype: lie.SE3Matrix
        """
        raise NotImplementedError

    def get_all_poses(self, joint_angles: dict) -> dict:
        T = {ROOT: self.T_base}
        for ee in self.end_effectors:
            for node in self.kinematic_map[ROOT][ee[0]][1:]:
                T[node] = self.get_pose(joint_angles, node)
        return T

    @abstractmethod
    def random_configuration(self):
        """
        Returns a random set of joint values within the joint limits
        determined by lb and ub.
        """
        raise NotImplementedError

    def end_effector_pos(self, q: dict) -> dict:
        goals = {}
        for ee in self.end_effectors:
            goals[ee[0]] = self.get_pose(q, ee[0]).trans
            goals[ee[1]] = self.get_pose(q, ee[1]).trans
        return goals

    @property
    def n(self) -> int:
        """
        :return: number of links, or joints (including root)
        """
        return self._n

    @n.setter
    def n(self, n: int):
        self._n = n

    @property
    def dim(self) -> int:
        """
        :return: dimension of the robot (2 or 3)
        """
        return self._dim

    @dim.setter
    def dim(self, dim: int):
        self._dim = dim

    @property
    def joint_ids(self):
        try:
            return self._joint_ids
        except AttributeError:
            self._joint_ids = list(self.kinematic_map.keys())
            return self._joint_ids

    @joint_ids.setter
    def joint_ids(self, ids: list):
        self._joint_ids = ids

    @property
    def structure(self) -> nx.DiGraph:
        """
        :return: graph representing the robot's structure
        """
        return self._structure

    @structure.setter
    def structure(self, structure: nx.DiGraph):
        self._structure = structure

    @property
    def kinematic_map(self) -> dict:
        """
        :return: topological graph of the robot's structure
        """
        return self._kinematic_map

    @kinematic_map.setter
    def kinematic_map(self, kinematic_map: dict):
        self._kinematic_map = kinematic_map

    @property
    @abstractmethod
    def end_effectors(self) -> list:
        """
        :return: all end-effector nodes
        """
        raise NotImplementedError

    @property
    def limit_edges(self) -> list:
        """
        :return: list of limited edges
        """
        return self._limit_edges

    @limit_edges.setter
    def limit_edges(self, lim: list):
        self._limit_edges = lim

    @property
    def T_base(self) -> SEMatrixBase:
        """
        :return: SE(dim) Transform to robot base frame
        """
        return self._T_base

    @T_base.setter
    def T_base(self, T_base: SEMatrixBase):
        self._T_base = T_base

    ########################################
    #         KINEMATIC PARAMETERS
    ########################################
    @property
    def ub(self) -> dict:
        """
        :return: Upper limits on joint values
        """
        return self._ub

    @ub.setter
    def ub(self, ub: dict):
        self._ub = ub if type(ub) is dict else list_to_variable_dict(flatten([ub]))

    @property
    def lb(self) -> dict:
        """
        :return: Lower limits on joint values
        """
        return self._lb

    @lb.setter
    def lb(self, lb: dict):
        self._lb = lb if type(lb) is dict else list_to_variable_dict(flatten([lb]))

    @property
    def d(self) -> dict:
        return self._d

    @d.setter
    def d(self, d: dict):
        self._d = d if type(d) is dict else list_to_variable_dict(flatten([d]))

    @property
    def al(self) -> dict:
        return self._al

    @al.setter
    def al(self, al: dict):
        self._al = al if type(al) is dict else list_to_variable_dict(flatten([al]))

    @property
    def a(self) -> dict:
        return self._a

    @a.setter
    def a(self, a: dict):
        self._a = a if type(a) is dict else list_to_variable_dict(flatten([a]))

    @property
    def th(self) -> dict:
        return self._th

    @th.setter
    def th(self, th: dict):
        self._th = th if type(th) is dict else list_to_variable_dict(flatten([th]))

    @property
    def spherical(self) -> bool:
        return False

    ########################################
    #         LAMBDIFICATION
    ########################################

    @property
    def lambdified(self) -> bool:
        return self._lambdified

    @lambdified.setter
    def lambdified(self, lambdified: bool):
        self._lambdified = lambdified

    def lambdify_get_pose(self):
        """
        Sets the fast full joint kinematics function with lambdify.
        """
        full_pose_expression = sp.symarray(
            "dummy", (self.dim + 1, self.dim + 1, self.n)
        )
        sym_vars = {}
        variable_angles = list(list_to_variable_dict(self.n * [0.0]).keys())
        sym_vars_list = []
        if not self.spherical:
            for var in variable_angles:
                sym_vars[var] = sp.symbols(var)
                sym_vars_list.append(sym_vars[var])
        else:
            for var in variable_angles:
                sym_vars[var] = sp.symbols([var + "_1", var + "_2"])
                sym_vars_list.append(sym_vars[var][0])
                sym_vars_list.append(sym_vars[var][1])
        for idx, var in enumerate(variable_angles):
            if self.dim == 2 or self.spherical:
                full_pose_expression[:, :, idx] = self.get_pose(
                    sym_vars, var
                ).as_matrix()

            else:
                full_pose_expression[:, :, idx] = self.get_pose(
                    sym_vars, var
                ).as_matrix()
        # if not self.spherical:
        #     x = sp.symarray("x", (self.n,))
        # else:
        #     x = sp.symarray("x", (self.n*2,))
        self.get_full_pose_lambdified = sp.lambdify(
            [sym_vars_list], full_pose_expression, "numpy"
        )
        self.lambdified = True

    def get_full_pose_fast_lambdify(self, node_inputs: dict):
        assert (
            self.lambdified
        ), "This robot has not yet been lambdified: call robot.lambdifiy_get_pose() first."
        input_list = list(node_inputs.values())
        pose_tensor = np.array(self.get_full_pose_lambdified(input_list))
        pose_dict = {}
        if self.spherical:
            for idx in range(self.n):
                pose_dict[f"p{idx+1}"] = pose_tensor[:, :, idx]
        else:
            for idx, key in enumerate(node_inputs):
                pose_dict[key] = pose_tensor[:, :, idx]
        return pose_dict


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
        # nx.set_node_attributes(chain_graph, None, POS)
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
        # nx.set_node_attributes(tree_graph, None, POS)
        return tree_graph

    def generate_structure_graph(self):
        self.structure = self.tree_graph(self.parents)

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

    def get_pose(self, node_inputs: dict, query_node: str):
        """
        Returns an SE2 element corresponding to the location
        of the query_node in the configuration determined by
        node_inputs.
        """
        if query_node == "p0":
            return SE2.identity()

        path_nodes = self.kinematic_map["p0"][query_node][1:]
        q = np.array([node_inputs[node] for node in path_nodes])
        a = np.array([self.a[node] for node in path_nodes])
        th = np.array([self.th[node] for node in path_nodes])
        return fk_2d(a, th, q)

    def joint_variables(self, G: nx.Graph) -> dict:
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
        self.limit_edges = []
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
                self.limit_edges += [[u, v]]  # TODO remove/fix

    def random_configuration(self):
        q = {}
        for key in self.structure:
            if key != "p0":
                q[key] = self.lb[key] + (self.ub[key] - self.lb[key]) * np.random.rand()
        return q

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
        self.limit_edges = []
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
                self.limit_edges += [[u, v]]  # TODO remove/fix

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

    def jacobian_linear(self, joint_angles: dict, query_frame: str = "") -> np.ndarray:
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


class RobotRevolute(Robot):
    def __init__(self, params):
        self.dim = 3
        self.n = params.get("num_joints", len(params["lb"]))  # number of joints
        self.axis_length = params.get("axis_length", 1)  # distance between p and q
        self.T_base = params.get("T_base", SE3.identity())  # base frame
        # self.modified_dh = params.get("modified_dh", False)

        # Topological "map" of the robot, if not provided assume chain
        if "parents" in params:
            self.parents = nx.DiGraph(params["parents"])
        else:
            self.parents = nx.path_graph(
                [f"p{idx}" for idx in range(self.n + 1)], nx.DiGraph
            )

        # A dict of shortest paths between joints for forward kinematics
        self.kinematic_map = nx.shortest_path(self.parents)

        # joint limits NOTE currently assuming symmetric around 0
        self.lb = params.get("lb", dict(zip(self.joint_ids, self.n * [-pi])))
        self.ub = params.get("ub", dict(zip(self.joint_ids, self.n * [pi])))

        # Use frame poses at zero conf if provided, otherwise construct from DH
        if "T_zero" in params:
            self.T_zero = params["T_zero"]
        else:
            try:
                self.a, self.d, self.al, self.th, self.modified_dh = (
                    params["a"],
                    params["d"],
                    params["alpha"],
                    params["theta"],
                    params["modified_dh"],
                )
            except KeyError:
                raise Exception("Robot description not provided.")

        self.generate_structure_graph()
        self.set_limits()
        super(RobotRevolute, self).__init__()

    @property
    def end_effectors(self) -> list:
        """
        Returns a list of end effector node pairs, since it's the
        last two points that are defined for a full pose.
        """
        S = self.parents
        return [[x, f"q{x[1:]}"] for x in S if S.out_degree(x) == 0]

    @property
    def T_zero(self) -> dict:
        if not hasattr(self, "_T_zero"):
            T = {"p0": self.T_base}
            kinematic_map = self.kinematic_map
            for ee in self.end_effectors:
                for node in kinematic_map["p0"][ee[0]][1:]:
                    path_nodes = kinematic_map["p0"][node][1:]

                    q = np.array([0 for node in path_nodes])
                    a = np.array([self.a[node] for node in path_nodes])
                    alpha = np.array([self.al[node] for node in path_nodes])
                    th = np.array([self.th[node] for node in path_nodes])
                    d = np.array([self.d[node] for node in path_nodes])

                    if not self.modified_dh:
                        T[node] = fk_3d(a, alpha, d, q + th)
                    else:
                        T[node] = modified_fk_3d(a, alpha, d, q + th)
            self._T_zero = T

        return self._T_zero

    @T_zero.setter
    def T_zero(self, T_zero: dict):
        self._T_zero = T_zero

    @property
    def parents(self) -> nx.DiGraph:
        return self._parents

    @parents.setter
    def parents(self, parents: nx.DiGraph):
        self._parents = parents

    def get_pose(self, joint_angles: dict, query_node: str) -> SE3:
        """
        Returns an SE3 element corresponding to the location
        of the query_node in the configuration determined by
        node_inputs.
        """
        kinematic_map = self.kinematic_map["p0"]["p" + query_node[1:]]

        T = self.T_base

        for idx in range(len(kinematic_map) - 1):
            pred, cur = kinematic_map[idx], kinematic_map[idx + 1]
            T_rel = self.T_zero[pred].inv().dot(self.T_zero[cur])
            T = T.dot(rot_axis(joint_angles[cur], "z")).dot(T_rel)

        if query_node[0] == "q":
            T = T.dot(trans_axis(self.axis_length, "z"))

        return T

    def generate_structure_graph(self):
        trans_z = trans_axis(self.axis_length, "z")
        T = self.T_zero

        S = nx.empty_graph(create_using=nx.DiGraph)

        for ee in self.end_effectors:
            k_map = self.kinematic_map["p0"][ee[0]]
            for idx in range(len(k_map)):
                cur, aux_cur = k_map[idx], f"q{k_map[idx][1:]}"
                cur_pos, aux_cur_pos = T[cur].trans, T[cur].dot(trans_z).trans
                dist = norm(cur_pos - aux_cur_pos)

                # Add nodes for joint and edge between them
                S.add_nodes_from([(cur, {POS: cur_pos}), (aux_cur, {POS: aux_cur_pos})])
                S.add_edge(cur, aux_cur, **{DIST: dist, LOWER: dist, UPPER: dist})

                # If there exists a preceeding joint, connect it to new
                if idx != 0:
                    pred, aux_pred = (k_map[idx - 1], f"q{k_map[idx-1][1:]}")
                    for u in [pred, aux_pred]:
                        for v in [cur, aux_cur]:
                            dist = norm(S.nodes[u][POS] - S.nodes[v][POS])
                            S.add_edge(u, v, **{DIST: dist, LOWER: dist, UPPER: dist})

        # Delete positions used for weights
        for u in S.nodes:
            del S.nodes[u][POS]

        # Set node type to robot
        nx.set_node_attributes(S, "robot", TYPE)

        # Set structure graph attribute
        self.structure = S

    def max_min_distance(self, T0: SE3, T1: SE3, T2: SE3) -> (float, float, str):
        """
        Given three frames, find the maximum and minimum distances between the
        frames T0 and T2. It is assumed that the two frames are connected by an
        unlimited revolute joint with its rotation axis being the z-axis
        of the frame T1.
        """
        tol = 10e-10
        # T_rel_01 = T0.inv().dot(T1)
        T_rel_12 = T1.inv().dot(T2)

        p0 = T0.as_matrix()[0:3, 3]
        z1 = T1.as_matrix()[0:3, 2]
        x1 = T1.as_matrix()[0:3, 0]
        p1 = T1.as_matrix()[0:3, 3]
        p2 = T2.as_matrix()[0:3, 3]

        p0_proj = p0 - (z1.dot(p0 - p1)) * z1  # p0 projected onto T1 plane
        p2_proj = p2 - (z1.dot(p2 - p1)) * z1  # p2 projected onto T1 plane

        if norm(p1 - p0_proj) < tol or norm(p2_proj - p1) < tol:
            d = norm(T2.trans - T0.trans)
            return d, d, False

        r = norm(p2_proj - p1)  # radius of circle p2_proj is on
        delta_th = arctan2(cross(x1, p2_proj - p1).dot(z1), np.dot(x1, p2_proj - p1))

        # closest and farthest point from p0_proj
        sol_1 = r * (p0_proj - p1) / norm(p0_proj - p1) + p1
        sol_2 = -r * (p0_proj - p1) / norm(p0_proj - p1) + p1
        sol_min = min(sol_1 - p0_proj, sol_2 - p0_proj, key=norm) + p0_proj
        sol_max = max(sol_1 - p0_proj, sol_2 - p0_proj, key=norm) + p0_proj

        th_max = arctan2(cross(x1, sol_max - p1).dot(z1), np.dot(x1, sol_max - p1))
        th_min = arctan2(cross(x1, sol_min - p1).dot(z1), np.dot(x1, sol_min - p1))

        rot_min = rot_axis(th_min - delta_th, "z")
        d_min = norm(T1.dot(rot_min).dot(T_rel_12).trans - T0.trans)

        rot_max = rot_axis(th_max - delta_th, "z")
        d_max = norm(T1.dot(rot_max).dot(T_rel_12).trans - T0.trans)

        if abs(th_max - delta_th) < tol and d_max > d_min:
            return d_max, d_min, "below"
        elif abs(th_min - delta_th) < tol and d_max > d_min:
            return d_max, d_min, "above"
        else:
            return d_max, d_min, False

    def set_limits(self):
        """
        Sets known bounds on the distances between joints.
        This is induced by link length and joint limits.
        """
        S = self.structure
        T = self.T_zero
        self.limit_edges = []  # edges enforcing joint limits
        self.limited_joints = []  # joint limits that can be enforced
        kinematic_map = self.kinematic_map
        T_axis = trans_axis(self.axis_length, "z")

        for ee in self.end_effectors:
            k_map = self.kinematic_map["p0"][ee[0]]
            for idx in range(2, len(k_map)):
                cur, prev = k_map[idx], k_map[idx - 2]
                names = [
                    (f"p{prev[1:]}", f"p{cur[1:]}"),
                    (f"p{prev[1:]}", f"q{cur[1:]}"),
                    (f"q{prev[1:]}", f"p{cur[1:]}"),
                    (f"q{prev[1:]}", f"q{cur[1:]}"),
                ]
                for ids in names:
                    path = kinematic_map[prev][cur]
                    T0, T1, T2 = [T[path[0]], T[path[1]], T[path[2]]]

                    if "q" in ids[0]:
                        T0 = T0.dot(T_axis)
                    if "q" in ids[1]:
                        T2 = T2.dot(T_axis)

                    d_max, d_min, limit = self.max_min_distance(T0, T1, T2)

                    if limit:

                        rot_limit = rot_axis(self.ub[cur], "z")

                        T_rel = T1.inv().dot(T2)

                        d_limit = norm(T1.dot(rot_limit).dot(T_rel).trans - T0.trans)

                        if limit == "above":
                            d_max = d_limit
                        else:
                            d_min = d_limit

                        self.limited_joints += [cur]
                        self.limit_edges += [[ids[0], ids[1]]]  # TODO remove/fix

                    S.add_edge(ids[0], ids[1])
                    if d_max == d_min:
                        S[ids[0]][ids[1]][DIST] = d_max
                    S[ids[0]][ids[1]][BOUNDED] = [limit]
                    S[ids[0]][ids[1]][UPPER] = d_max
                    S[ids[0]][ids[1]][LOWER] = d_min

    def joint_variables(self, G: nx.Graph, T_final: dict = None) -> np.ndarray:
        """
        Calculate joint angles from a complete set of point positions.
        """
        # TODO: make this more readable
        tol = 1e-10
        q_zero = list_to_variable_dict(self.n * [0])
        get_pose = self.get_pose
        axis_length = self.axis_length

        T = {}
        T["p0"] = self.T_base
        theta = {}

        for ee in self.end_effectors:
            k_map = self.kinematic_map["p0"][ee[0]]
            for idx in range(1, len(k_map)):
                cur, aux_cur = k_map[idx], f"q{k_map[idx][1:]}"
                pred, aux_pred = (k_map[idx - 1], f"q{k_map[idx-1][1:]}")

                T_prev = T[pred]

                T_prev_0 = get_pose(q_zero, pred)
                T_0 = get_pose(q_zero, cur)
                T_rel = T_prev_0.inv().dot(T_0)
                T_0_q = get_pose(q_zero, cur).dot(trans_axis(axis_length, "z"))
                T_rel_q = T_prev_0.inv().dot(T_0_q)

                p = G.nodes[cur][POS] - T_prev.trans
                q = G.nodes[aux_cur][POS] - T_prev.trans
                ps = T_prev.inv().as_matrix()[:3, :3].dot(p)
                qs = T_prev.inv().as_matrix()[:3, :3].dot(q)

                zs = skew(np.array([0, 0, 1]))
                cp = (T_rel.trans - ps) + zs.dot(zs).dot(T_rel.trans)
                cq = (T_rel_q.trans - qs) + zs.dot(zs).dot(T_rel_q.trans)
                ap = zs.dot(T_rel.trans)
                aq = zs.dot(T_rel_q.trans)
                bp = zs.dot(zs).dot(T_rel.trans)
                bq = zs.dot(zs).dot(T_rel_q.trans)

                c0 = cp.dot(cp) + cq.dot(cq)
                c1 = 2 * (cp.dot(ap) + cq.dot(aq))
                c2 = 2 * (cp.dot(bp) + cq.dot(bq))
                c3 = ap.dot(ap) + aq.dot(aq)
                c4 = bp.dot(bp) + bq.dot(bq)
                c5 = 2 * (ap.dot(bp) + aq.dot(bq))

                # poly = [c0 -c2 +c4, 2*c1 - 2*c5, 2*c0 + 4*c3 -2*c4, 2*c1 + 2*c5, c0 + c2 + c4]
                diff = np.array(
                    [
                        c1 - c5,
                        2 * c2 + 4 * c3 - 4 * c4,
                        3 * c1 + 3 * c5,
                        8 * c2 + 4 * c3 - 4 * c4,
                        -4 * c1 + 4 * c5,
                    ]
                )
                if all(diff < tol):
                    theta[cur] = 0
                else:
                    sols = np.roots(
                        diff
                    )  # solutions to the Whaba problem for fixed axis

                    def error_test(x):
                        if abs(x.imag) > 0:
                            return 1e6
                        x = -2 * arctan2(x.real, 1)
                        return (
                            c0
                            + c1 * sin(x)
                            - c2 * cos(x)
                            + c3 * sin(x) ** 2
                            + c4 * cos(x) ** 2
                            - c5 * sin(2 * x) / 2
                        )

                    sol = min(sols, key=error_test)
                    theta[cur] = -2 * arctan2(sol.real, 1)

                T[cur] = (T_prev.dot(rot_axis(theta[cur], "z"))).dot(T_rel)

            if T_final is None:
                return theta

            if (
                T_final[ee[0]] is not None
                and norm(cross(T_rel.trans, np.array([0, 0, 1]))) < tol
            ):
                T_th = (T[cur]).inv().dot(T_final[ee[0]]).as_matrix()
                theta[ee[0]] += np.arctan2(T_th[1, 0], T_th[0, 0])

        return theta

    def random_configuration(self):
        """
        Returns a random set of joint values within the joint limits
        determined by lb and ub.
        """
        q = {}
        for key in self.joint_ids:
            if key != "p0":
                q[key] = self.lb[key] + (self.ub[key] - self.lb[key]) * np.random.rand()
        return q

    def euclidean_cost_hessian(self, J: dict, K: dict, r: dict):
        """
        Based on 'Solving Inverse Kinematics Using Exact Hessian Matrices', Erleben, 2019
        :param J: dictionary of linear velocity kinematic Jacobians
        :param K: dictionary of tensors representing second order derivative information
        :param r: dictionary where each value for key ee is goal_ee - F_ee(theta)
        :return:
        """
        H = 0
        for e in J.keys():
            J_e = J[e]
            N = J_e.shape[1]
            H += J_e.T @ J_e
            # TODO: Try with einsum for speed, maybe?
            for idx in range(N):
                for jdx in range(idx, N):
                    dH = K[e][:, idx, jdx].T @ r[e]
                    H[idx, jdx] -= dH
                    if idx != jdx:
                        H[jdx, idx] -= dH
        return H

    def jacobian_linear_symb(
        self, joint_angles: dict, pose_term=False, ee_keys=None
    ) -> dict:
        """
        Calculate the robot's linear velocity Jacobian for all end-effectors.
        """
        kinematic_map = self.kinematic_map["p0"]  # get map to all nodes from root

        if ee_keys is None:
            end_effector_nodes = []
            for ee in self.end_effectors:  # get p nodes in end-effectors
                if ee[0][0] == "p":
                    end_effector_nodes += [ee[0]]
                else:
                    end_effector_nodes += [ee[1]]
        else:
            end_effector_nodes = ee_keys
        # Ts = self.get_all_poses_symb(joint_angles)  # all frame poses
        Ts = self.get_all_poses(joint_angles)  # all frame poses
        J = {}  # np.zeros([0, len(node_names) - 1])
        for ee in end_effector_nodes:  # iterate through end-effector nodes
            ee_path = kinematic_map[ee][
                1:
            ]  # [:-1]  # no last node, only phys. joint locations

            T_0_ee = Ts[ee].as_matrix()  # ee frame
            if pose_term:
                dZ = np.array([0.0, 0.0, 1.0])
                p_ee = T_0_ee[0:3, 0:3] @ dZ + T_0_ee[0:3, -1]
            else:
                p_ee = T_0_ee[0:3, -1]  # ee position

            Jp = np.zeros([3, self.n], dtype=object)  # translation jac
            for joint in ee_path:  # algorithm fills Jac per column
                T_0_i = Ts[list(self.parents.predecessors(joint))[0]].as_matrix()
                z_hat_i = T_0_i[:3, 2]
                if pose_term:
                    p_i = T_0_i[0:3, 0:3] @ dZ + T_0_i[0:3, -1]
                else:
                    p_i = T_0_i[:3, -1]
                j_idx = int(joint[1:]) - 1  # node_names.index(joint) - 1
                Jp[:, j_idx] = cross_symb(z_hat_i, p_ee - p_i)
            J[ee] = Jp
        return J

    def hessian_linear_symb(
        self,
        joint_angles: dict,
        J=None,
        query_frame: str = "",
        pose_term=False,
        ee_keys=None,
    ) -> np.ndarray:
        """
        Calculates the Hessian at query_frame geometrically.
        """
        # dZ = np.array([0., 0., 1.])  # For the pose_term = True case
        if J is None:
            J = self.jacobian_linear_symb(joint_angles, pose_term=pose_term)
        kinematic_map = self.kinematic_map["p0"]  # get map to all nodes from root

        if ee_keys is None:
            end_effector_nodes = []
            for ee in self.end_effectors:  # get p nodes in end-effectors
                if ee[0][0] == "p":
                    end_effector_nodes += [ee[0]]
                if ee[1][0] == "p":
                    end_effector_nodes += [ee[1]]
        else:
            end_effector_nodes = ee_keys

        N = len(joint_angles)
        M = 3  # 3 translation
        H = {}
        # Ts = self.get_all_poses_symb(joint_angles)
        Ts = self.get_all_poses(joint_angles)
        for ee in end_effector_nodes:
            J_ee = J[ee]
            H_ee = np.zeros((M, N, N), dtype=object)
            ee_path = kinematic_map[ee][1:]

            visited = []

            for joint in ee_path:
                visited += [joint]
                jdx = int(joint[1:]) - 1
                for joint_base in visited:
                    idx = int(joint_base[1:]) - 1
                    T_0_base = Ts[
                        list(self.parents.predecessors(joint_base))[0]
                    ].as_matrix()
                    z_hat_base = T_0_base[:3, 2]
                    h = cross_symb(z_hat_base, J_ee[0:3, jdx])
                    H_ee[:, idx, jdx] = h
                    H_ee[:, jdx, idx] = H_ee[:, idx, jdx]

            H[ee] = H_ee
        return H
