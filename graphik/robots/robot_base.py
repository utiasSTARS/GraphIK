#!/usr/bin/env python3
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union, Tuple
from numpy.typing import ArrayLike

import networkx as nx
from graphik.utils.constants import *
from graphik.utils.utils import flatten, list_to_variable_dict
from liegroups.numpy import SE2, SE3

SEMatrix = Union[SE2, SE3]


class Robot(ABC):
    """
    Describes the kinematic parameters for a robot whose joints and links form a tree (no loops like in parallel
    mechanisms).
    """

    def __init__(self):
        self.lambdified = False

    @abstractmethod
    def get_pose(self, joint_angles: Dict[str, Any], query_node: str) -> SEMatrix:
        """
        TODO rename to pose, requires extensive refactor
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
        nodes: Union[List[str], str],
        Ts: Dict[str, SEMatrix] = None,
    ) -> Dict[str, ArrayLike]:
        """
        TODO planar doesn't have an isolated jacobian method
        Calculate the robot's Jacobian for nodes
        """
        raise NotImplementedError

    @abstractmethod
    def random_configuration(self) -> Dict[str, float]:
        """
        Returns a random set of joint values within the joint limits
        determined by lb and ub.
        """
        raise NotImplementedError

    @abstractmethod
    def joint_variables(
        self, G: nx.DiGraph, T: Dict[str, SEMatrix] = None
    ) -> Dict[str, Any]:
        """
        Finds the set of decision variables corresponding to the
        graph realization G.

        :param G: networkx.DiGraph with known node positions
        :returns: Dictionary of joint variables
        :rtype: np.ndarray
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def end_effectors(self) -> List[Any]:
        """
        :return: all end-effector nodes
        """
        raise NotImplementedError

    ########################################
    #         ATTRIBUTES
    ########################################
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

    @joint_ids.setter
    def joint_ids(self, ids: list):
        self._joint_ids = ids

    @property
    def T_base(self) -> SEMatrix:
        """
        :return: SE(dim) Transform to robot base frame
        """
        return self._T_base

    @T_base.setter
    def T_base(self, T_base: SEMatrix):
        self._T_base = T_base

    @property
    def limit_edges(self) -> List[List[str]]:
        """
        FIXME switch to tuples
        :return: list of limited edges
        """
        return self._limit_edges

    @limit_edges.setter
    def limit_edges(self, lim: List[List[str]]):
        self._limit_edges = lim

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
    def d(self) -> Dict[str, Any]:
        return self._d

    @d.setter
    def d(self, d: dict):
        self._d = d if type(d) is dict else list_to_variable_dict(flatten([d]))

    @property
    def al(self) -> Dict[str, Any]:
        return self._al

    @al.setter
    def al(self, al: dict):
        self._al = al if type(al) is dict else list_to_variable_dict(flatten([al]))

    @property
    def a(self) -> Dict[str, Any]:
        return self._a

    @a.setter
    def a(self, a: dict):
        self._a = a if type(a) is dict else list_to_variable_dict(flatten([a]))

    @property
    def th(self) -> Dict[str, Any]:
        return self._th

    @th.setter
    def th(self, th: dict):
        self._th = th if type(th) is dict else list_to_variable_dict(flatten([th]))

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
            for node in self.kinematic_map[ROOT][ee[0]][1:]:
                T[node] = self.get_pose(joint_angles, node)
        return T

    def end_effector_pos(self, q: Dict[str, float]) -> Dict[str, ArrayLike]:
        """
        Gets the positions of all end-effector nodes in a dictionary.
        """
        goals = {}
        for ee in self.end_effectors:
            for node in ee:
                goals[node] = self.get_pose(q, node).trans
        return goals


from graphik.robots.robot_planar import RobotPlanar
from graphik.robots.robot_spherical import RobotSpherical
from graphik.robots.robot_revolute import RobotRevolute
