import numpy as np
import networkx as nx
from numpy import pi
from numpy.testing import assert_array_equal
import unittest

from graphik.graphs.graph_base import RobotSphericalGraph
from graphik.robots.robot_base import RobotPlanar

from graphik.utils.dgp import adjacency_matrix_from_graph
from graphik.utils.utils import list_to_variable_dict


class TestAdjacencyMatrices(unittest.TestCase):
    def test_planar_chain_pose_goal(self):
        n = 3
        a = list_to_variable_dict(np.ones(n))
        th = list_to_variable_dict(np.zeros(n))
        lim_u = list_to_variable_dict(pi * np.ones(n))
        lim_l = list_to_variable_dict(-pi * np.ones(n))
        params = {
            "a": a,
            "theta": th,
            "joint_limits_upper": lim_u,
            "joint_limits_lower": lim_l,
        }

        # Adjacency matrix derived by hand
        F_gt = np.array(
            [
                [0, 1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1, 1],
                [1, 1, 0, 0, 1, 1],
                [1, 0, 0, 0, 1, 0],
                [1, 1, 1, 1, 0, 1],
                [1, 1, 1, 0, 1, 0],
            ]
        )

        robot = RobotPlanar(params)
        graph = RobotSphericalGraph(robot)

        q_goal = graph.robot.random_configuration()
        goals = {
            f"p{n}": robot.get_pose(q_goal, f"p{n}").trans,
            f"p{n-1}": robot.get_pose(q_goal, f"p{n-1}").trans,
        }
        G = graph.complete_from_pos(goals)

        F = adjacency_matrix_from_graph(G)

        assert_array_equal(F, F_gt)

    def test_planar_chain_position_goal(self):
        n = 3
        a = list_to_variable_dict(np.ones(n))
        th = list_to_variable_dict(np.zeros(n))
        lim_u = list_to_variable_dict(pi * np.ones(n))
        lim_l = list_to_variable_dict(-pi * np.ones(n))
        params = {
            "a": a,
            "theta": th,
            "joint_limits_upper": lim_u,
            "joint_limits_lower": lim_l,
        }

        # Adjacency matrix derived by hand
        F_gt = np.array(
            [
                [0, 1, 1, 1, 0, 1],
                [1, 0, 1, 0, 0, 1],
                [1, 1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 1],
                [1, 1, 1, 0, 1, 0],
            ]
        )

        robot = RobotPlanar(params)
        graph = RobotSphericalGraph(robot)

        q_goal = graph.robot.random_configuration()
        goals = {
            f"p{n}": robot.get_pose(q_goal, f"p{n}").trans,
            # f"p{n-1}": robot.get_pose(q_goal, f"p{n-1}").trans,
        }
        G = graph.complete_from_pos(goals)

        F = adjacency_matrix_from_graph(G)

        assert_array_equal(F, F_gt)

    def test_planar_tree_position_goal(self):
        height = 3
        gen = nx.balanced_tree(2, height, create_using=nx.DiGraph)
        gen = nx.relabel_nodes(gen, {node: f"p{node}" for node in gen})
        n = gen.number_of_edges()
        parents = nx.to_dict_of_lists(gen)
        a = list_to_variable_dict(np.ones(n))
        th = list_to_variable_dict(np.zeros(n))
        lim_u = list_to_variable_dict(np.pi * np.ones(n))
        lim_l = list_to_variable_dict(-np.pi * np.ones(n))
        params = {
            "a": a,
            "theta": th,
            "parents": parents,
            "joint_limits_upper": lim_u,
            "joint_limits_lower": lim_l,
        }

        # Adjacency matrix derived by hand
        F_gt = np.array(
            [
                [
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                ],
            ]
        )
        robot = RobotPlanar(params)
        graph = RobotSphericalGraph(robot)

        q_goal = robot.random_configuration()
        goals = {}
        for idx, ee_pair in enumerate(robot.end_effectors):
            goals[ee_pair[0]] = robot.get_pose(q_goal, ee_pair[0]).trans
            # goals[ee_pair[1]] = robot.get_pose(q_goal, ee_pair[1]).trans

        G = graph.complete_from_pos(goals)

        F = adjacency_matrix_from_graph(G)

        assert_array_equal(F, F_gt)

    def test_planar_tree_pose_goal(self):
        height = 3
        gen = nx.balanced_tree(2, height, create_using=nx.DiGraph)
        gen = nx.relabel_nodes(gen, {node: f"p{node}" for node in gen})
        n = gen.number_of_edges()
        parents = nx.to_dict_of_lists(gen)
        a = list_to_variable_dict(np.ones(n))
        th = list_to_variable_dict(np.zeros(n))
        lim_u = list_to_variable_dict(np.pi * np.ones(n))
        lim_l = list_to_variable_dict(-np.pi * np.ones(n))
        params = {
            "a": a,
            "theta": th,
            "parents": parents,
            "joint_limits_upper": lim_u,
            "joint_limits_lower": lim_l,
        }

        # Adjacency matrix derived by hand
        F_gt = np.array(
            [
                [
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    0,
                    1,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    1,
                    0,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                ],
            ]
        )
        robot = RobotPlanar(params)
        graph = RobotSphericalGraph(robot)

        q_goal = robot.random_configuration()
        goals = {}
        for idx, ee_pair in enumerate(robot.end_effectors):
            goals[ee_pair[0]] = robot.get_pose(q_goal, ee_pair[0]).trans
            goals[ee_pair[1]] = robot.get_pose(q_goal, ee_pair[1]).trans

        G = graph.complete_from_pos(goals)

        F = adjacency_matrix_from_graph(G)

        assert_array_equal(F, F_gt)
