from typing import Any, Dict, Union, List
from numpy.typing import ArrayLike

import networkx as nx
import numpy as np
from graphik.robots.robot_base import Robot, SEMatrix
from graphik.utils import list_to_variable_dict, flatten, fk_3d, modified_fk_3d
from graphik.utils.constants import ROOT, TRANSFORM, MAIN_PREFIX
# from graphik.utils import *

from liegroups.numpy import SE3


class RobotRevolute(Robot):
    def __init__(self, params):
        super(RobotRevolute, self).__init__(params)

        self.dim = 3  # 3d workspace

        # Use frame poses at zero conf if provided, otherwise find using DH
        if "T_zero" in params:
            T_zero = params["T_zero"]
        elif all(k in params for k in ("a", "d", "alpha", "theta", "modified_dh")):
            T_zero = self.from_dh_params(params)
        else:
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
                omega = self.nodes[cur]["T0"].as_matrix()[:3, 2]
                q = self.nodes[cur]["T0"].as_matrix()[:3, 3]
                self.nodes[cur]["S"] = np.hstack((np.cross(-omega, q), omega))

                # Relative transforms between coordinate frames as edge attributes
                if idx != 0:
                    pred = k_map[idx - 1]
                    self[pred][cur][TRANSFORM] = (
                        self.nodes[pred]["T0"].inv().dot(self.nodes[cur]["T0"])
                    )

    def from_dh_params(self, params):

        a, d, al, th, modified_dh = (
            params["a"],
            params["d"],
            params["alpha"],
            params["theta"],
            params["modified_dh"],
        )
        a = a if type(a) is dict else list_to_variable_dict(flatten([a]))
        d = d if type(d) is dict else list_to_variable_dict(flatten([d]))
        al = al if type(al) is dict else list_to_variable_dict(flatten([al]))
        th = th if type(th) is dict else list_to_variable_dict(flatten([th]))

        T = {ROOT: SE3.identity()}
        kinematic_map = self.kinematic_map
        for ee in self.end_effectors:
            for node in kinematic_map[ROOT][ee][1:]:
                path_nodes = kinematic_map[ROOT][node][1:]

                q = np.asarray([0 for node in path_nodes])
                a_ = np.asarray([a[node] for node in path_nodes])
                alpha_ = np.asarray([al[node] for node in path_nodes])
                th_ = np.asarray([th[node] for node in path_nodes])
                d_ = np.asarray([d[node] for node in path_nodes])

                if not modified_dh:
                    T[node] = fk_3d(a_, alpha_, d_, q + th_)
                else:
                    T[node] = modified_fk_3d(a_, alpha_, d_, q + th_)
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
