from typing import Any, Dict, Union, List
from numpy.typing import ArrayLike

import networkx as nx
import numpy as np
from graphik.robots.robot_base import Robot, SEMatrix
from graphik.utils import *

from liegroups.numpy import SE3


class RobotRevolute(Robot):
    def __init__(self, params):
        super(RobotRevolute, self).__init__(params)

        self.dim = 3  # 3d workspace

        # Use frame poses at zero conf if provided, otherwise construct from DH
        if "T_zero" in params:
            T_zero = params["T_zero"]
        else:
            try:
                T_zero = self.from_params()
            except KeyError:
                raise Exception("Robot description not provided.")

        # Poses of frames at zero config as node attributes
        nx.set_node_attributes(self, values=T_zero, name="T0")

        # Set node and edge attributes describing geometry
        self.set_geometric_attributes()

    def set_geometric_attributes(self):
        end_effectors = self.end_effectors
        kinematic_map = self.kinematic_map

        for ee in end_effectors:
            k_map = kinematic_map[ROOT][ee]
            for idx in range(len(k_map)):

                # Twists representing rotation axes as node attributes
                cur = k_map[idx]
                omega = self.nodes[cur]["T0"].as_matrix()[:3, 2]
                q = self.nodes[cur]["T0"].as_matrix()[:3, 3]
                self.nodes[cur]["S"] = np.hstack((np.cross(-omega, q), omega))

                # Relative transforms between coordinate frames as edge attributes
                if idx != 0:
                    pred = k_map[idx - 1]
                    self[pred][cur][TRANSFORM] = (
                        self.nodes[pred]["T0"].inv().dot(self.nodes[cur]["T0"])
                    )

    def pose(
        self, joint_angles: Dict[str, float], query_node: Union[List[str], str]
    ) -> Union[Dict[str, SE3], SE3]:
        """
        Given a list of N joint variables, calculate the Nth joint's pose.

        :param node_inputs: joint variables node names as keys mapping to values
        :param query_node: node ID of node whose pose we want
        :returns: SE2 or SE3 pose
        :rtype: lie.SE3Matrix
        """
        # TODO support multiple query nodes
        # TODO avoid for loop by vectorizing matrix exponential
        kinematic_map = self.kinematic_map[ROOT][query_node]
        T = self.nodes[ROOT]["T0"]
        for idx in range(len(kinematic_map) - 1):
            pred, cur = kinematic_map[idx], kinematic_map[idx + 1]
            T = T.dot(SE3.exp(self.nodes[pred]["S"] * joint_angles[cur]))
        T = T.dot(self.nodes[query_node]["T0"])

        return T

    def jacobian(
        self,
        joint_angles: Dict[str, float],
        query_nodes: Union[List[str], str],
    ) -> Dict[str, ArrayLike]:
        """
        Calculate the robot's Jacobian for all end-effectors.

        :param joint_angles: dictionary describing the current joint configuration
        :param query_nodes: list of nodes that the Jacobian should be computed for
        :return: dictionary of Jacobians indexed by relevant node
        """
        kinematic_map = self.kinematic_map[ROOT]  # get map to all nodes from root

        # find end-effector nodes
        if query_nodes is None:
            query_nodes = self.end_effectors

        J = {}
        for node in query_nodes:  # iterate through nodes
            path = kinematic_map[node]  # find joints that move node
            T = self.nodes[ROOT]["T0"]

            J[node] = np.zeros([6, self.n])
            for idx in range(len(path) - 1):
                pred, cur = path[idx], path[idx + 1]
                if idx == 0:
                    J[node][:, idx] = self.nodes[pred]["S"]
                else:
                    ppred = list(self.predecessors(pred))[0]
                    T = T.dot(SE3.exp(self.nodes[ppred]["S"] * joint_angles[pred]))
                    Ad = T.adjoint()
                    J[node][:, idx] = Ad.dot(self.nodes[pred]["S"])
        return J

    def from_params(self):
        self.a, self.d, self.al, self.th, self.modified_dh = (
            self.params["a"],
            self.params["d"],
            self.params["alpha"],
            self.params["theta"],
            self.params["modified_dh"],
        )
        T = {ROOT: SE3.identity()}
        kinematic_map = self.kinematic_map
        for ee in self.end_effectors:
            for node in kinematic_map[ROOT][ee][1:]:
                path_nodes = kinematic_map[ROOT][node][1:]

                q = np.asarray([0 for node in path_nodes])
                a = np.asarray([self.a[node] for node in path_nodes])
                alpha = np.asarray([self.al[node] for node in path_nodes])
                th = np.asarray([self.th[node] for node in path_nodes])
                d = np.asarray([self.d[node] for node in path_nodes])

                if not self.modified_dh:
                    T[node] = fk_3d(a, alpha, d, q + th)
                else:
                    T[node] = modified_fk_3d(a, alpha, d, q + th)
        return T

    def jacobian_geometric(
        self,
        joint_angles: Dict[str, float],
        nodes: Union[List[str], str],
        Ts: Dict[str, SEMatrix] = None,
    ) -> Dict[str, ArrayLike]:

        """
        Calculate the robot's linear velocity Jacobian for all end-effectors.

        :param joint_angles: dictionary describing the current joint configuration
        :param nodes: list of nodes that the Jacobian shoulf be computed for
        :param Ts: the current list of frame poses for all nodes, speeds up computation
        :return: dictionary of Jacobians indexed by relevant node
        """
        kinematic_map = self.kinematic_map[ROOT]  # get map to all nodes from root

        # find end-effector nodes #FIXME so much code and relies on notation
        if nodes is None:
            nodes = []
            for ee in self.end_effectors:  # get p nodes in end-effectors
                if ee[0][0] == MAIN_PREFIX:
                    nodes += [ee[0]]
                elif ee[1][0] == MAIN_PREFIX:
                    nodes += [ee[1]]

        if Ts is None:
            Ts = self.get_all_poses(joint_angles)  # compute all poses

        J = {}
        for node in nodes:  # iterate through nodes
            path = kinematic_map[node][1:]  # find joints that move node
            p_ee = Ts[node].trans

            J[node] = np.zeros([6, self.n])
            for idx, joint in enumerate(path):  # algorithm fills Jac per column
                T_0_i = Ts[list(self.parents.predecessors(joint))[0]]
                z_hat_i = T_0_i.rot.mat[:3, 2]
                p_i = T_0_i.trans
                J[node][:3, idx] = np.cross(z_hat_i, p_ee - p_i)
                J[node][3:, idx] = z_hat_i
        return J

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

    def hessian_linear_symb(
        self,
        joint_angles: dict,
        J=None,
        query_frame: str = "",
        pose_term=False,
        ee_keys=None,
    ) -> ArrayLike:
        """
        Calculates the Hessian at query_frame geometrically.
        """
        # dZ = np.array([0., 0., 1.])  # For the pose_term = True case
        if J is None:
            J = self.jacobian(joint_angles, pose_term=pose_term)
        kinematic_map = self.kinematic_map[ROOT]  # get map to all nodes from root

        if ee_keys is None:
            end_effector_nodes = []
            for ee in self.end_effectors:  # get p nodes in end-effectors
                if ee[0][0] == MAIN_PREFIX:
                    end_effector_nodes += [ee[0]]
                if ee[1][0] == MAIN_PREFIX:
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


if __name__ == "__main__":
    from graphik.utils.roboturdf import load_ur10, load_kuka, load_schunk_lwa4d

    # robot, graph = load_schunk_lwa4d()
    robot, graph = load_ur10()
    # print(robot.nodes(data=True))
    # print(robot.edges(data=True))
    # print(robot.random_configuration())
    q = robot.random_configuration()
    print(robot.pose(q, "p6"))
    print(graph.get_pose(q, "p6"))
    print(robot.jacobian(q, ["p6"]))
    # print(robot.edges())
    # q = robot.random_configuration()
    # q = robot.random_configuration()
    # T = robot.get_pose(q, "p6")
    # print(robot.get_pose(q, "p6"))
    # print(robot.pose_exp(q, "p6"))
    # print(robot.get_jacobian(q, ["p6"])["p6"])
    # print("---------------------------")
    # print(SE3.adjoint(T.inv()).dot(robot.jacobian(q, ["p6"])["p6"]))
    # print("---------------------------")
    # print(robot.jacobian(q, ["p6"])["p6"] - robot.get_jacobian(q, ["p6"])["p6"])
