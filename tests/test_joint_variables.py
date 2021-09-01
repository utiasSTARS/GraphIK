#!/usr/bin/env python3
from graphik.utils.dgp import graph_from_pos, pos_from_graph
import numpy as np
import networkx as nx
import unittest
import graphik
from numpy.testing import assert_allclose
from numpy.random import rand, randint
from numpy import pi
from graphik.graphs import (
    RobotPlanarGraph,
    RobotRevoluteGraph,
    RobotSphericalGraph,
)
from graphik.robots import RobotPlanar, RobotRevolute, RobotSpherical
from graphik.utils.roboturdf import RobotURDF
from graphik.utils import *


class TestJointVariables(unittest.TestCase):
    def test_scale_invariance(self):
        n = 7
        ub = (pi) * np.ones(n)
        lb = -ub
        # fname = graphik.__path__[0] + "/robots/urdfs/ur10_mod.urdf"
        fname = graphik.__path__[0] + "/robots/urdfs/panda_arm.urdf"
        # fname = graphik.__path__[0] + "/robots/urdfs/kuka_iiwr.urdf"
        # fname = graphik.__path__[0] + "/robots/urdfs/kuka_lwr.urdf"
        # fname = graphik.__path__[0] + "/robots/urdfs/jaco2arm6DOF_no_hand.urdf"
        urdf_robot = RobotURDF(fname)
        robot_urdf = urdf_robot.make_Revolute3d(
            ub, lb
        )  # make the Revolute class from a URDF
        params = {
            "lb": lb[:n],
            "ub": ub[:n],
            "T_zero": robot_urdf.T_zero,
            "num_joints": urdf_robot.n_q_joints
        }
        robot = RobotRevolute(params)

        graph = RobotRevoluteGraph(robot)
        for _ in range(100):
            q_goal = robot.random_configuration()
            T_goal = {}
            T_goal[f"p{n}"] = robot.pose(q_goal, "p" + str(n))
            X = graph.realization(q_goal)
            P = normalize_positions(pos_from_graph(X))
            q_rec = graph.joint_variables(graph_from_pos(P, node_ids=list(X)), T_goal)
            self.assertIsNone(
                assert_allclose(list(q_goal.values()), list(q_rec.values()), rtol=1e-5)
            )

    def test_urdf_params_3d_chain(self):
        n = 7
        ub = (pi) * np.ones(n)
        lb = -ub
        # fname = graphik.__path__[0] + "/robots/urdfs/ur10_mod.urdf"
        fname = graphik.__path__[0] + "/robots/urdfs/panda_arm.urdf"
        # fname = graphik.__path__[0] + "/robots/urdfs/kuka_iiwr.urdf"
        # fname = graphik.__path__[0] + "/robots/urdfs/kuka_lwr.urdf"
        # fname = graphik.__path__[0] + "/robots/urdfs/jaco2arm6DOF_no_hand.urdf"
        urdf_robot = RobotURDF(fname)
        robot_urdf = urdf_robot.make_Revolute3d(
            ub, lb
        )  # make the Revolute class from a URDF
        # Tz = robot_urdf.T_zero["p4"].as_matrix()
        # Tz[2, -1] = robot_urdf.T_zero["p3"].as_matrix()[2, -1]
        # robot_urdf.T_zero["p4"] = SE3.from_matrix(Tz)
        params = {
            "lb": lb[:n],
            "ub": ub[:n],
            "T_zero": robot_urdf.T_zero,
            "num_joints": urdf_robot.n_q_joints
        }
        robot = RobotRevolute(params)

        graph = RobotRevoluteGraph(robot)
        for _ in range(100):
            q_goal = robot.random_configuration()
            T_goal = {}
            T_goal[f"p{n}"] = robot.pose(q_goal, "p" + str(n))
            X = graph.realization(q_goal)
            q_rec = graph.joint_variables(X, T_goal)
            self.assertIsNone(
                assert_allclose(list(q_goal.values()), list(q_rec.values()), rtol=1e-5)
            )

    def test_random_params_3d_chain(self):
        # TODO include randomized theta to FK and such
        print("Testing randomly generated params 3d ... \n")
        modified_dh = False
        for _ in range(100):
            n = randint(3, high=20)  # number of joints

            # Generate random DH parameters
            a = rand(n)
            d = rand(n)
            al = rand(n) * pi / 2 - 2 * rand(n) * pi / 2
            th = 0 * np.ones(n)
            ub = np.ones(n) * pi - 2 * pi * rand(n)
            lb = -ub

            params = {
                "a": a,
                "alpha": al,
                "d": d,
                "theta": th,
                "lb": lb,
                "ub": ub,
                "modified_dh": modified_dh,
                "num_joints": n
            }
            robot = RobotRevolute(params)  # instantiate robot
            graph = RobotRevoluteGraph(robot)  # instantiate graph

            q_goal = robot.random_configuration()
            T_goal = {}
            T_goal[f"p{n}"] = robot.pose(q_goal, "p" + str(n))
            X = graph.realization(q_goal)
            q_rec = graph.joint_variables(X, T_goal)
            self.assertIsNone(
                assert_allclose(list(q_goal.values()), list(q_rec.values()), rtol=1e-5)
            )

    def test_random_params_3d_spherical(self):
        # TODO include randomized theta to FK and such
        print("Testing randomly generated params 3d ... \n")

        for _ in range(100):

            n = randint(3, high=20)

            # Generate random DH parameters
            a = list_to_variable_dict(0 * rand(n))
            d = list_to_variable_dict(rand(n))
            al = list_to_variable_dict(0 * rand(n))
            th = list_to_variable_dict(0 * rand(n))
            lim_u = list_to_variable_dict(pi * np.ones(n))
            lim_l = list_to_variable_dict(-pi * np.ones(n))

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

            q_goal = robot.random_configuration()
            X = graph.realization(q_goal)
            q_rec = robot.joint_variables(X)
            self.assertIsNone(
                assert_allclose(list(q_goal.values()), list(q_rec.values()), rtol=1e-5)
            )

    def test_random_params_3d_spherical_tree(self):
        # TODO include randomized theta to FK and such
        print("Testing randomly generated params 3d ... \n")

        for _ in range(100):

            height = randint(2, high=5)
            gen = nx.balanced_tree(2, height, create_using=nx.DiGraph)
            gen = nx.relabel_nodes(gen, {node: f"p{node}" for node in gen})
            n = gen.number_of_edges()
            # Generate random DH parameters
            a = list_to_variable_dict(0 * rand(n))
            d = list_to_variable_dict(rand(n))
            al = list_to_variable_dict(0 * rand(n))
            th = list_to_variable_dict(0 * rand(n))
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

            q_goal = robot.random_configuration()
            X = graph.realization(q_goal)
            q_rec = robot.joint_variables(X)
            self.assertIsNone(
                assert_allclose(list(q_goal.values()), list(q_rec.values()), rtol=1e-5)
            )

    def test_random_params_2d(self):

        for _ in range(100):

            n = randint(3, high=20)

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

            robot = RobotPlanar(params)
            graph = RobotPlanarGraph(robot)

            q_goal = robot.random_configuration()
            # TODO: was T_goal supposed to be tested too?
            T_goal = robot.get_pose(q_goal, f"p{n}").trans
            X = graph.realization(q_goal)

            q_rec = robot.joint_variables(X)
            self.assertIsNone(
                assert_allclose(list(q_goal.values()), list(q_rec.values()), rtol=1e-5)
            )

    def test_random_params_2d_tree(self):
        for _ in range(10):
            height = randint(2, high=5)
            gen = nx.balanced_tree(2, height, create_using=nx.DiGraph)
            gen = nx.relabel_nodes(gen, {node: f"p{node}" for node in gen})
            n = gen.number_of_edges()
            a = list_to_variable_dict(np.ones(n))
            th = list_to_variable_dict(np.zeros(n))
            parents = nx.to_dict_of_lists(gen)
            lim_u = list_to_variable_dict(pi * np.ones(n))
            lim_l = list_to_variable_dict(-pi * np.ones(n))
            params = {
                "a": a,
                "theta": th,
                "parents": parents,
                "joint_limits_upper": lim_u,
                "joint_limits_lower": lim_l,
            }
            robot = RobotPlanar(params)
            graph = RobotPlanarGraph(robot)
            q_goal = robot.random_configuration()
            T_goal = robot.get_pose(q_goal, f"p{n}").trans
            X = graph.realization(q_goal)
            q_rec = robot.joint_variables(X)
            self.assertIsNone(
                assert_allclose(list(q_goal.values()), list(q_rec.values()), rtol=1e-5)
            )

    def test_special_cases_3d_chain(self):
        # TODO extend to randomly zeroed out params, maybe common manipulators
        print("\n Testing cases with zeroed out parameters ...")

        n = 6

        # # UR10 coordinates for testing
        modified_dh = False
        a = [0, -0.612, 0.5723, 0, 0, 0]
        d = [0.1273, 0, 0, 0.1639, 0.1157, 0.0922]
        al = [pi / 2, 0, 0, pi / 2, -pi / 2, 0]
        th = [0, pi, 0, 0, 0, 0]
        ub = np.ones(n) * pi - 2 * pi * rand(n)
        lb = -ub

        params = {
            "a": a,
            "alpha": al,
            "d": d,
            "theta": th,
            "lb": lb,
            "ub": ub,
            "modified_dh": modified_dh,
        }
        robot = RobotRevolute(params)
        graph = RobotRevoluteGraph(robot)

        for _ in range(100):
            q_goal = robot.random_configuration()
            T_goal = {}
            T_goal[f"p{n}"] = robot.get_pose(
                list_to_variable_dict(q_goal), "p" + str(n)
            )
            X = graph.realization(q_goal)
            # q_rec = robot.joint_variables(X, T_goal.as_matrix())
            q_rec = graph.joint_variables(X, T_goal)
            self.assertIsNone(
                assert_allclose(list(q_goal.values()), list(q_rec.values()), rtol=1e-5)
            )

    def test_special_cases_3d_tree(self):
        # TODO extend to randomly zeroed out params, maybe common manipulators
        print("\n Testing cases with zeroed out parameters ...")
        modified_dh = False

        n = 5
        parents = {"p0": ["p1"], "p1": ["p2", "p3"], "p2": ["p4"], "p3": ["p5"]}
        a = {"p1": 0, "p2": -0.612, "p3": -0.612, "p4": -0.5732, "p5": -0.5732}
        d = {"p1": 0.1237, "p2": 0, "p3": 0, "p4": 0, "p5": 0}
        al = {"p1": pi / 2, "p2": 0, "p3": 0, "p4": 0, "p5": 0}
        th = {"p1": 0, "p2": 0, "p3": 0, "p4": 0, "p5": 0}
        ub = list_to_variable_dict((pi) * np.ones(n))
        lb = list_to_variable_dict(-(pi) * np.ones(n))

        params = {
            "a": a,
            "alpha": al,
            "d": d,
            "theta": th,
            "lb": lb,
            "ub": ub,
            "modified_dh": modified_dh,
            "parents": parents,
        }
        robot = RobotRevolute(params)
        graph = RobotRevoluteGraph(robot)

        for _ in range(100):
            q_goal = robot.random_configuration()
            T_goal = {}
            for ee in robot.end_effectors:
                T_goal[ee[0]] = robot.get_pose(list_to_variable_dict(q_goal), ee[0])
            X = graph.realization(q_goal)
            q_rec = graph.joint_variables(X, T_goal)
            q_rec = dict(sorted(q_rec.items()))
            self.assertIsNone(
                assert_allclose(list(q_goal.values()), list(q_rec.values()), rtol=1e-5)
            )


if __name__ == "__main__":
    unittest.main()
