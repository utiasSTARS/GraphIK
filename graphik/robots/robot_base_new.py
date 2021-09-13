#!/usr/bin/env python3
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union, Tuple
from numpy.typing import ArrayLike

import networkx as nx
from graphik.utils import *
from liegroups.numpy import SE2, SE3
from math import pi

SEMatrix = Union[SE2, SE3]


class Robot(nx.DiGraph):
    """
    Describes the kinematic parameters for a robot whose joints and links form a tree (no loops like in parallel
    mechanisms).
    """

    def __init__(self, params: Dict):
        super(Robot, self).__init__()
        self.lambdified = False
        self.params = params
        self.n = params["num_joints"]

        # Topological "map" of the robot, if not provided assume chain
        if "parents" in params:
            topology = nx.DiGraph(params["parents"])
        else:
            topology = nx.path_graph(
                [f"p{idx}" for idx in range(self.n + 1)], nx.DiGraph
            )
        self.add_nodes_from(topology.nodes())
        self.add_edges_from(topology.edges())

        # A dict of shortest paths between joints for forward kinematics
        self.kinematic_map = nx.shortest_path(self)

        # Lower and upper joint limits
        self.lb = params.get("lb", dict(zip(self.joint_ids, self.n * [-pi])))
        self.ub = params.get("ub", dict(zip(self.joint_ids, self.n * [pi])))
        nx.set_node_attributes(self, values=self.lb, name="lb")
        nx.set_node_attributes(self, values=self.ub, name="ub")

    @abstractmethod
    def set_geometric_attributes(self):
        raise NotImplementedError

    @abstractmethod
    def pose(self, joint_angles: Dict[str, Any], query_node: str) -> SEMatrix:
        """
        Given a list of N joint variables, calculate the Nth joint's pose.

        :param node_inputs: joint variables node names as keys mapping to values
        :param query_node: node ID of node whose pose we want
        :returns: SE2 or SE3 pose
        :rtype: lie.SE3Matrix
        """
        raise NotImplementedError

    @abstractmethod
    def jacobian(
        self,
        joint_angles: Dict[str, float],
        query_nodes: Union[List[str], str],
    ) -> Dict[str, ArrayLike]:
        """
        TODO planar doesn't have an isolated jacobian method
        Calculate the robot's Jacobian for nodes
        """
        raise NotImplementedError

    def random_configuration(self):
        """
        Returns a random set of joint values within the joint limits
        determined by lb and ub.
        """
        q = {}
        for key in self.joint_ids:
            if key != ROOT:
                q[key] = self.lb[key] + (self.ub[key] - self.lb[key]) * np.random.rand()
        return q

    def zero_configuration(self):
        """
        Returns zero joint values within the joint limits
        determined by lb and ub.
        """
        q = {}
        for key in self.joint_ids:
            if key != ROOT:
                q[key] = 0
        return q

    @property
    def end_effectors(self) -> List:
        """
        Returns a list of end effector node pairs, since it's the
        last two points that are defined for a full pose.
        """
        if not hasattr(self, "_end_effectors"):
            self._end_effectors = [x for x in self.nodes() if self.out_degree(x) == 0]
        return self._end_effectors

    ########################################
    #         ATTRIBUTES
    ########################################
    @property
    def kinematic_map(self) -> dict:
        """
        :return: topological graph of the robot's structure, but not auxiliary points q
        Used for forward kinematics on multi end-effector manipulators.
        """
        return self._kinematic_map

    @kinematic_map.setter
    def kinematic_map(self, kinematic_map: dict):
        self._kinematic_map = kinematic_map

    @property
    def joint_ids(self) -> List[str]:
        try:
            return self._joint_ids
        except AttributeError:
            self._joint_ids = list(self.kinematic_map.keys())
            return self._joint_ids

    @property
    def T_base(self) -> SEMatrix:
        """
        :return: SE(dim) Transform to robot base frame
        """
        try:
            return self._T_base
        except AttributeError:
            self._T_base = self.nodes[ROOT]["T0"]
        return self._T_base

    @property
    def limited_joints(self) -> List[str]:
        """
        :return: list of limited edges
        """
        return self._limited_joints

    @limited_joints.setter
    def limited_joints(self, lim: List[str]):
        self._limited_joints = lim

    ########################################
    #         KINEMATIC PARAMETERS
    ########################################
    @property
    def ub(self) -> Dict[str, Any]:
        """
        :return: Upper limits on joint values
        """
        return self._ub

    @ub.setter
    def ub(self, ub: dict):
        self._ub = ub if type(ub) is dict else list_to_variable_dict(flatten([ub]))

    @property
    def lb(self) -> Dict[str, Any]:
        """
        :return: Lower limits on joint values
        """
        return self._lb

    @lb.setter
    def lb(self, lb: dict):
        self._lb = lb if type(lb) is dict else list_to_variable_dict(flatten([lb]))

    @property
    def spherical(self) -> bool:
        return False

    ########################################
    #         CONVENIENCE METHODS
    ########################################

    def get_all_poses(self, joint_angles: Dict[str, Any]) -> Dict[str, SEMatrix]:
        """
        Convenient method for getting all poses of coordinate systems attached to each point in the robot's graph description.
        """
        T = {ROOT: self.T_base}
        for ee in self.end_effectors:
            for node in self.kinematic_map[ROOT][ee][1:]:
                T[node] = self.pose(joint_angles, node)
        return T

    def end_effector_pos(self, q: Dict[str, float]) -> Dict[str, ArrayLike]:
        """
        Gets the positions of all end-effector nodes in a dictionary.
        """
        goals = {}
        for ee in self.end_effectors:
            for node in ee:
                goals[node] = self.pose(q, node).trans
        return goals
