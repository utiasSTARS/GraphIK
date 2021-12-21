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

        for ee in self.end_effectors:
            k_map = self.kinematic_map[ROOT][ee]
            for idx in range(len(k_map)):

                # Twists representing rotation axes as node attributes
                cur = k_map[idx]
                omega = np.array([0,0,1])
                q = np.hstack((self.nodes[cur]["T0"].as_matrix()[:2, 2], 0))
                self.nodes[cur]["S"] = np.hstack((np.cross(-omega, q), omega))[[0,1,5]]

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

    def pose(self, joint_angles: Dict[str, float], query_node: str) -> SE3:
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
            T = T.dot(SE2.exp(self.nodes[pred]["S"] * joint_angles[cur]))
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

            J[node] = np.zeros([3, self.n])
            for idx in range(len(path) - 1):
                pred, cur = path[idx], path[idx + 1]
                if idx == 0:
                    J[node][:, idx] = self.nodes[pred]["S"]
                else:
                    ppred = list(self.predecessors(pred))[0]
                    T = T.dot(SE2.exp(self.nodes[ppred]["S"] * joint_angles[pred]))
                    Ad = T.adjoint()
                    J[node][:, idx] = Ad.dot(self.nodes[pred]["S"])
        return J

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
    # print(q)
    # print(q_)
    print(robot.jacobian(q, robot.end_effectors))
    print(robot.jacobian_old(q, robot.end_effectors))
