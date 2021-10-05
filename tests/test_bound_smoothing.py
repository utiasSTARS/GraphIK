#!/usr/bin/env python3
import numpy as np
from numpy import pi
from numpy.testing import assert_array_less
import unittest
import networkx as nx
from graphik.graphs import (
    ProblemGraphPlanar,
    ProblemGraphRevolute,
)
from graphik.robots import RobotRevolute, RobotPlanar
from graphik.utils.dgp import pos_from_graph, graph_from_pos, bound_smoothing
from graphik.utils.utils import list_to_variable_dict
from graphik.utils.geometry import trans_axis
from graphik.utils.roboturdf import load_ur10


TOL = 1e-6


class TestBoundSmoothing(unittest.TestCase):
    def test_random_params_2d_chain(self):
        n = 5
        for _ in range(100):
            a = list_to_variable_dict(np.ones(n))
            th = list_to_variable_dict(np.zeros(n))
            lim = np.minimum(np.random.rand(n) * pi + 0.20, pi)
            lim_u = lim * np.ones(n)
            lim_l = -lim * np.ones(n)
            params = {
                "link_lengths": a,
                "theta": th,
                "joint_limits_upper": lim_u,
                "joint_limits_lower": lim_l,
                "num_joints": n,
            }

            robot = RobotPlanar(params)
            graph = ProblemGraphPlanar(robot)

            q_goal = graph.robot.random_configuration()
            G_goal = graph.realization(q_goal)
            X_goal = pos_from_graph(G_goal)
            D_goal = graph.distance_matrix_from_joints(q_goal)

            goals = {f"p{n-1}": X_goal[-2, :], f"p{n}": X_goal[-1, :]}
            G = graph.complete_from_pos(goals)

            lb, ub = bound_smoothing(G)
            self.assertIsNone(
                assert_array_less(D_goal, ub ** 2 + TOL * np.ones(D_goal.shape))
            )
            self.assertIsNone(
                assert_array_less(lb ** 2 - TOL * np.ones(D_goal.shape), D_goal)
            )

    def test_random_params_2d_tree(self):
        for _ in range(100):
            height = 4
            gen = nx.balanced_tree(2, height, create_using=nx.DiGraph)
            gen = nx.relabel_nodes(gen, {node: f"p{node}" for node in gen})
            n = gen.number_of_edges()
            parents = nx.to_dict_of_lists(gen)
            a = list_to_variable_dict(np.ones(n))
            th = list_to_variable_dict(np.zeros(n))
            lim = np.minimum(np.random.rand(n) * pi + 0.20, pi)
            lim_u = lim * np.ones(n)
            lim_l = -lim * np.ones(n)
            params = {
                "link_lengths": a,
                "theta": th,
                "parents": parents,
                "joint_limits_upper": lim_u,
                "joint_limits_lower": lim_l,
                "num_joints": n,
            }

            robot = RobotPlanar(params)
            graph = ProblemGraphPlanar(robot)

            q_goal = graph.robot.random_configuration()
            G_goal = graph.realization(q_goal)
            D_goal = graph.distance_matrix_from_joints(q_goal)

            goals = {}
            for idx, ee_pair in enumerate(robot.end_effectors):
                goals[ee_pair] = robot.pose(q_goal, ee_pair).trans

            G = graph.complete_from_pos(goals)

            lb, ub = bound_smoothing(G)
            self.assertIsNone(
                assert_array_less(D_goal, ub ** 2 + TOL * np.ones(D_goal.shape))
            )
            self.assertIsNone(
                assert_array_less(lb ** 2 - TOL * np.ones(D_goal.shape), D_goal)
            )

    def test_random_params_3d_revolute_chain(self):
        robot, graph = load_ur10()

        for _ in range(100):
            q_goal = graph.robot.random_configuration()
            D_goal = graph.distance_matrix_from_joints(q_goal)
            T_goal = robot.pose(q_goal, "p" + str(robot.n))

            G = graph.complete_from_pos(
                {f"p{robot.n}": T_goal.trans, f"q{robot.n}": T_goal.dot(trans_axis(1, "z")).trans}
            )
            lb, ub = bound_smoothing(G)

            self.assertIsNone(
                assert_array_less(D_goal, ub ** 2 + TOL * np.ones(D_goal.shape))
            )
            self.assertIsNone(
                assert_array_less(lb ** 2 - TOL * np.ones(D_goal.shape), D_goal)
            )


if __name__ == "__main__":
    np.random.seed(22)
    test = TestBoundSmoothing()
    # test.test_random_params_3d_revolute_chain()
    # print("DONE!")
