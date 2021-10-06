from typing import Dict, List, Union, Any
from numpy.typing import ArrayLike

import networkx as nx
import numpy as np
from graphik.robots.robot_base import Robot
from graphik.utils import *
from liegroups.numpy import SE2, SO2
from numpy import cos, pi, sqrt, inf


class RobotPlanar(Robot):
    def __init__(self, params):
        super(RobotPlanar, self).__init__(params)
        self.dim = 2

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

                # Relative transforms between coordinate frames as edge attributes
                if idx != 0:
                    pred = k_map[idx - 1]
                    self[pred][cur][TRANSFORM] = (
                        self.nodes[pred]["T0"].inv().dot(self.nodes[cur]["T0"])
                    )

    def from_params(self):
        self.l = self.params["link_lengths"]
        T = {ROOT: SE2.identity()}
        q0 = self.zero_configuration()
        kinematic_map = self.kinematic_map
        for ee in self.end_effectors:
            for node in kinematic_map[ROOT][ee][1:]:
                path_nodes = kinematic_map[ROOT][node][1:]
                T[node] = fk_tree_2d(self.l, q0, q0, path_nodes)
        return T

    def pose(
        self, joint_angles: Dict[str, float], query_node: Union[List[str], str]
    ) -> Union[Dict[str, SE2], SE2]:
        """
        Returns an SE2 element corresponding to the location
        of the query_node in the configuration determined by
        node_inputs.
        """
        if query_node == "p0":
            return SE2.identity()

        path_nodes = self.kinematic_map["p0"][query_node][1:]
        q = np.asarray([joint_angles[node] for node in path_nodes])
        l = np.asarray([self.l[node] for node in path_nodes])
        th = np.asarray([0 for node in path_nodes])
        return fk_2d(l, th, q)

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
                T_0_i = Ts[list(self.predecessors(joint))[0]].as_matrix()
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


if __name__ == "__main__":

    n = 10

    l = list_to_variable_dict(np.ones(n))
    # th = list_to_variable_dict(np.zeros(n))
    lim_u = list_to_variable_dict(np.pi * np.ones(n))
    lim_l = list_to_variable_dict(-np.pi * np.ones(n))
    params = {
        "link_lengths": l,
        "ub": lim_u,
        "lb": lim_l,
        "num_joints": n,
    }

    # robot = Revolute2dChain(params)
    robot = RobotPlanar(params)
    # print(robot.nodes(data=True))
    # print(robot.edges(data=True))
    q = robot.random_configuration()
    # print(robot.pose(q, "p9"))
    # print(robot.jacobian(q, ["p9"]))

    from graphik.graphs.graph_planar import ProblemGraphPlanar
    graph = ProblemGraphPlanar(robot)
    # print(graph.directed.nodes(data=True))
    # print(graph.directed.edges(data=True))
    G = graph.realization(q)
    q_ = graph.joint_variables(G)
    print(q)
    print(q_)
