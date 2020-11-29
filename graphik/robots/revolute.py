#!/usr/bin/env python3
import networkx as nx
import numpy as np
from graphik.robots.robot_base import (
    RobotSpherical,
    RobotPlanar,
    RobotRevolute,
)
from graphik.utils.utils import (
    flatten,
    list_to_variable_dict,
    transZ,
    spherical_angle_bounds_to_revolute,
)
from graphik.utils.kinematics_helpers import roty
from liegroups.numpy import SE3, SO3
from numpy import arctan2, cos, pi, sin, sqrt

DIST = "weight"
POS = "pos"
UPPER = "upper_limit"
LOWER = "lower_limit"


class Spherical3dChain(RobotSpherical):
    def __init__(self, params, T_base=np.identity(4)):
        super(Spherical3dChain, self).__init__()

        self.T_base = SE3.from_matrix(T_base)
        self.a = params["a"]
        self.al = params["alpha"]
        self.d = params["d"]
        self.th = params["theta"]
        self.ub = params["joint_limits_upper"]
        self.lb = params["joint_limits_lower"]
        self.n = len(self.th)  # number of links

        self.structure = self.chain_graph()
        self.kinematic_map = nx.shortest_path(self.structure.copy())
        self.set_limits()

    def chain_graph(self) -> nx.DiGraph:
        """
        Directed graph representing the robots chain structure
        """
        edg_lst = [
            (f"p{idx}", f"p{idx+1}", self.d[f"p{idx+1}"]) for idx in range(self.n)
        ]
        chain_graph = nx.DiGraph()
        chain_graph.add_weighted_edges_from(edg_lst)
        return chain_graph

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

        Ts = self.get_full_pose_fast_lambdify(joint_angles)  # all frame poses
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

    @property
    def spherical(self) -> bool:
        return True

    def to_revolute(self):
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
            T_zero[new_node2] = T_zero[new_node1].dot(Ry_back).dot(transZ(d))

            joint_prev = new_node2

        # for key in T_zero:
        #     if key not in ub.keys() and key is not 'p0':
        #         ub[key] = np.pi
        #         lb[key] = -np.pi

        params = {"T_zero": T_zero, "ub": ub, "lb": lb}
        return RobotRevolute(params), old_to_new_names, ang_lims_map


class Spherical3dTree(RobotSpherical):
    def __init__(self, params, T_base=np.identity(4)):

        self.T_base = SE3.from_matrix(T_base)
        self.a = params["a"]
        self.al = params["alpha"]
        self.d = params["d"]
        self.th = params["theta"]
        self.ub = params["joint_limits_upper"]
        self.lb = params["joint_limits_lower"]
        self.parents = params["parents"]
        self.n = len(self.th)  # number of links
        self.dim = 3

        self.structure = self.tree_graph()
        self.kinematic_map = nx.shortest_path(self.structure.copy())
        self.set_limits()

        super(Spherical3dTree, self).__init__()

    def tree_graph(self) -> nx.DiGraph:
        """
        Needed for forward kinematics (computing the shortest path).
        :return: Directed graph representing the robot's tree structure.
        """
        tree_graph = nx.DiGraph(self.parents)
        for parent, child in tree_graph.edges():
            tree_graph.edges[parent, child]["weight"] = self.d[child]
        return tree_graph

    @property
    def spherical(self) -> bool:
        # TODO is this used?
        return True

    def to_revolute(self):  # -> Revolute3dTree:
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
                T_zero[new_grand_child] = T_zero[new_child].dot(Ry_back).dot(transZ(d))
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


class Revolute3dChain(RobotRevolute):
    def __init__(self, params):
        super(Revolute3dChain, self).__init__(params)

        if "T_base" in params:
            self.T_base = params["T_base"]
        else:
            self.T_base = SE3.from_matrix(np.identity(4))

        # Use frame poses at zero conf if provided, if not use DH
        if "T_zero" in params:
            self.T_zero = params["T_zero"]
            self.n = len(self.T_zero) - 1  # number of links
        else:
            if "modified_dh" in params:
                self.modified_dh = params["modified_dh"]
            else:
                self.modified_dh = False

            if all(k in params for k in ("a", "d", "alpha", "theta")):
                self.a = params["a"]
                self.d = params["d"]
                self.al = params["alpha"]
                self.th = params["theta"]
                self.n = len(self.al)  # number of links
            else:
                raise Exception("Robot description not provided.")

        # Topological "map" of the robot
        if "parents" in params:
            self.parents = nx.DiGraph(params["parents"])
        else:
            names = [f"p{idx}" for idx in range(self.n + 1)]
            self.parents = nx.path_graph(names, nx.DiGraph)

        self.kinematic_map = nx.shortest_path(self.parents)

        # joint limits TODO currently assuming symmetric around 0
        if "lb" and "ub" in params:
            self.lb = params["lb"]
            self.ub = params["ub"]
        else:
            self.lb = list_to_variable_dict(self.n * [-pi])
            self.ub = list_to_variable_dict(self.n * [pi])

        self.structure = self.structure_graph()
        self.limit_edges = []  # edges enforcing joint limits
        self.limited_joints = []  # joint limits that can be enforced
        self.set_limits()


class Revolute3dTree(RobotRevolute):
    def __init__(self, params):
        super(Revolute3dTree, self).__init__(params)
        if "T_base" in params:
            self.T_base = params["T_base"]
        else:
            self.T_base = SE3.from_matrix(np.identity(4))

        # Use frame poses at zero conf if provided, if not use DH
        if "T_zero" in params:
            self.T_zero = params["T_zero"]
            self.n = len(self.T_zero) - 1  # number of links
        else:
            if "modified_dh" in params:
                self.modified_dh = params["modified_dh"]
            else:
                self.modified_dh = False

            if all(k in params for k in ("a", "d", "alpha", "theta")):
                self.a = params["a"]
                self.d = params["d"]
                self.al = params["alpha"]
                self.th = params["theta"]
                self.n = len(self.al)  # number of links
            else:
                raise Exception("Robot description not provided.")

        # Topological "map" of the robot
        if "parents" in params:
            self.parents = nx.DiGraph(params["parents"])
        else:
            names = [f"p{idx}" for idx in range(self.n + 1)]
            self.parents = nx.path_graph(names, nx.DiGraph)

        self.kinematic_map = nx.shortest_path(self.parents)

        # joint limits TODO currently assuming symmetric around 0
        if "lb" and "ub" in params:
            self.lb = params["lb"]
            self.ub = params["ub"]
        else:
            self.lb = list_to_variable_dict(self.n * [-pi])
            self.ub = list_to_variable_dict(self.n * [pi])

        self.structure = self.structure_graph()
        self.limit_edges = []  # edges enforcing joint limits
        self.limited_joints = []  # joint limits that can be enforced
        self.set_limits()


if __name__ == "__main__":
    n = np.random.randint(3, high=20)
    a = list_to_variable_dict(np.ones(n))
    th = list_to_variable_dict(np.zeros(n))

    params = {"a": a, "theta": th}

    robot = RobotPlanar(params)
    print(robot.structure.edges())
