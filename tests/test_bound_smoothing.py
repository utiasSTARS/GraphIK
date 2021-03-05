#!/usr/bin/env python3
import numpy as np
from numpy import pi
from numpy.testing import assert_array_less
import unittest
import networkx as nx
from graphik.graphs import (
    RobotPlanarGraph,
    RobotRevoluteGraph,
    RobotSphericalGraph,
)
from graphik.robots import RobotRevolute, RobotSpherical, RobotPlanar
from graphik.utils.dgp import pos_from_graph, graph_from_pos, bound_smoothing
from graphik.utils.utils import list_to_variable_dict
from graphik.utils.geometry import trans_axis


TOL = 1e-6


class TestBoundSmoothing(unittest.TestCase):
    def test_random_params_2d_chain(self):
        n = 5
        for _ in range(100):
            a = list_to_variable_dict(np.ones(n))
            th = list_to_variable_dict(np.zeros(n))
            lim = np.minimum(np.random.rand(n) * pi + 0.20, pi)
            lim_u = list_to_variable_dict(lim * np.ones(n))
            lim_l = list_to_variable_dict(-lim * np.ones(n))
            params = {
                "a": a,
                "theta": th,
                "joint_limits_upper": lim_u,
                "joint_limits_lower": lim_l,
            }

            robot = RobotPlanar(params)
            graph = RobotPlanarGraph(robot)

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
            lim_u = list_to_variable_dict(lim * np.ones(n))
            lim_l = list_to_variable_dict(-lim * np.ones(n))
            params = {
                "a": a,
                "theta": th,
                "parents": parents,
                "joint_limits_upper": lim_u,
                "joint_limits_lower": lim_l,
            }

            robot = RobotPlanar(params)
            graph = RobotPlanarGraph(robot)

            q_goal = graph.robot.random_configuration()
            G_goal = graph.realization(q_goal)
            D_goal = graph.distance_matrix_from_joints(q_goal)

            goals = {}
            for idx, ee_pair in enumerate(robot.end_effectors):
                goals[ee_pair[0]] = robot.get_pose(q_goal, ee_pair[0]).trans

            G = graph.complete_from_pos(goals)

            lb, ub = bound_smoothing(G)
            self.assertIsNone(
                assert_array_less(D_goal, ub ** 2 + TOL * np.ones(D_goal.shape))
            )
            self.assertIsNone(
                assert_array_less(lb ** 2 - TOL * np.ones(D_goal.shape), D_goal)
            )

    def test_random_params_3d_spherical_chain(self):
        n = 5
        for _ in range(100):
            a = list_to_variable_dict(0 * np.random.rand(n))
            d = list_to_variable_dict(np.random.rand(n))
            al = list_to_variable_dict(0 * np.random.rand(n))
            th = list_to_variable_dict(0 * np.random.rand(n))
            lim = np.minimum(np.random.rand(n) * pi, pi)
            lim_u = list_to_variable_dict(lim * np.ones(n))
            lim_l = list_to_variable_dict(-lim * np.ones(n))

            params = {
                "a": a,
                "alpha": al,
                "d": d,
                "theta": th,
                "joint_limits_lower": lim_l,
                "joint_limits_upper": lim_u,
            }
            robot = RobotSpherical(params)  # instantiate robot
            graph = RobotSphericalGraph(robot)  # instantiate graph

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

    def test_random_params_3d_spherical_tree(self):
        n = 5
        for _ in range(100):
            height = 4
            gen = nx.balanced_tree(2, height, create_using=nx.DiGraph)
            gen = nx.relabel_nodes(gen, {node: f"p{node}" for node in gen})
            n = gen.number_of_edges()
            # Generate random DH parameters
            a = list_to_variable_dict(0 * np.random.rand(n))
            d = list_to_variable_dict(np.random.rand(n))
            al = list_to_variable_dict(0 * np.random.rand(n))
            th = list_to_variable_dict(0 * np.random.rand(n))
            parents = nx.to_dict_of_lists(gen)
            lim_u = list_to_variable_dict(pi * np.ones(n))
            lim_l = list_to_variable_dict(-pi * np.ones(n))

            params = {
                "a": a,
                "alpha": al,
                "d": d,
                "theta": th,
                "parents": parents,
                "joint_limits_lower": lim_l,
                "joint_limits_upper": lim_u,
            }
            robot = RobotSpherical(params)  # instantiate robot
            graph = RobotSphericalGraph(robot)  # instantiate graph

            q_goal = graph.robot.random_configuration()
            G_goal = graph.realization(q_goal)
            D_goal = graph.distance_matrix_from_joints(q_goal)

            goals = {}
            for idx, ee_pair in enumerate(robot.end_effectors):
                goals[ee_pair[0]] = robot.get_pose(q_goal, ee_pair[0]).trans
                goals[ee_pair[1]] = robot.get_pose(q_goal, ee_pair[1]).trans

            G = graph.complete_from_pos(goals)
            lb, ub = bound_smoothing(G)

            self.assertIsNone(
                assert_array_less(D_goal, ub ** 2 + TOL * np.ones(D_goal.shape))
            )
            self.assertIsNone(
                assert_array_less(lb ** 2 - TOL * np.ones(D_goal.shape), D_goal)
            )

    def test_random_params_3d_revolute_chain(self):
        # UR10 DH params theta, d, alpha, a
        n = 6
        a = [0, -0.612, -0.5723, 0, 0, 0]
        d = [0.1273, 0, 0, 0.1639, 0.1157, 0.0922]
        al = [pi / 2, 0, 0, pi / 2, -pi / 2, 0]
        th = [0, pi, 0, 0, 0, 0]
        ub = (pi / 4) * np.ones(n)
        lb = -ub

        dZ = 1

        params = {
            "a": a,
            "alpha": al,
            "d": d,
            "theta": th,
            "modified_dh": False,
            "lb": lb,
            "ub": ub,
        }
        robot = RobotRevolute(params)  # instantiate robot
        graph = RobotRevoluteGraph(robot)  # instantiate graph

        for _ in range(100):
            q_goal = graph.robot.random_configuration()
            D_goal = graph.distance_matrix_from_joints(q_goal)
            T_goal = robot.get_pose(list_to_variable_dict(q_goal), "p" + str(n))

            G = graph.complete_from_pos(
                {f"p{n}": T_goal.trans, f"q{n}": T_goal.dot(trans_axis(dZ, "z")).trans}
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
