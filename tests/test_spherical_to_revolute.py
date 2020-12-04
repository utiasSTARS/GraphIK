"""
Test the method that converts a spherical chain to a revolute chain. Mostly checking forward kinematics.
"""

import numpy as np
import time
import unittest
import networkx as nx
from graphik.solvers.local_solver import LocalSolver
from graphik.robots.robot_base import RobotSpherical
from graphik.graphs.graph_base import RobotSphericalGraph, RobotRevoluteGraph
from graphik.utils.utils import list_to_variable_dict, list_to_variable_dict_spherical


class TestForwardKinematics(unittest.TestCase):
    def test_tree_solver(self):
        pass

    def test_chain_solver(self):
        n = 2
        d = list(np.random.rand(n) + 0.5)
        d_dict = list_to_variable_dict(d)
        theta = n * [0.0]
        theta_dict = list_to_variable_dict(theta)
        al = n * [0.0]
        al_dict = list_to_variable_dict(al)
        a = n * [0.0]
        a_dict = list_to_variable_dict(a)
        lim_u = list_to_variable_dict(np.random.rand(n) * np.pi)
        lim_l = list_to_variable_dict(np.random.rand(n) * np.pi)
        params = {
            "d": d_dict,
            "a": a_dict,
            "alpha": al_dict,
            "theta": theta_dict,
            "joint_limits_upper": lim_u,
            "joint_limits_lower": lim_l,
        }
        robot = RobotSpherical(params)
        robot.lambdify_get_pose()

        # Revolute robot
        robot_revolute, names_map, ang_lims_map = robot.to_revolute()

        q_goal = robot.random_configuration()

        ee_goals = {}
        ee_goals_revolute = {}
        for ee in robot.end_effectors:
            ee_goals[ee[0]] = robot.get_pose(q_goal, ee[0])
            ee_goals_revolute[names_map[ee[0]]] = ee_goals[ee[0]]

        # Create solvers
        solver_params = {"solver": "trust-exact", "tol": 1e-6, "maxiter": 200}
        solver = LocalSolver(solver_params)
        solver.set_symbolic_cost_function(robot, ee_goals, robot.lb.keys())

        solver_revolute = LocalSolver(solver_params)
        solver_revolute.set_revolute_cost_function(
            robot_revolute, ee_goals_revolute, robot_revolute.lb.keys()
        )

        problem_params = {"initial_guess": list_to_variable_dict(n * [[0.0, 0.0]])}
        problem_params_revolute = {
            "initial_guess": list_to_variable_dict(2 * n * [0.0])
        }
        results = solver.solve(RobotSphericalGraph(robot), problem_params)
        results_revolute = solver_revolute.solve(
            RobotRevoluteGraph(robot_revolute), problem_params_revolute
        )

        self.assertTrue(np.all(np.isclose(results.x, results_revolute.x)))

    def test_random_chain(self):
        n_runs = 100
        n = 10
        d = list(np.random.rand(n) + 0.5)
        d_dict = list_to_variable_dict(d)
        theta = n * [0.0]
        theta_dict = list_to_variable_dict(theta)
        al = n * [0.0]
        al_dict = list_to_variable_dict(al)
        a = n * [0.0]
        a_dict = list_to_variable_dict(a)
        lim_u = list_to_variable_dict(np.random.rand(n) * np.pi)
        lim_l = list_to_variable_dict(np.random.rand(n) * np.pi)
        params = {
            "d": d_dict,
            "a": a_dict,
            "alpha": al_dict,
            "theta": theta_dict,
            "joint_limits_upper": lim_u,
            "joint_limits_lower": lim_l,
        }
        robot = RobotSpherical(params)

        for _ in range(n_runs):
            input_list = list(np.random.rand(2 * n))
            poses = {}
            input_pairs = list_to_variable_dict_spherical(input_list, in_pairs=True)
            for key in input_pairs:
                poses[key] = robot.get_pose(input_pairs, key)

            # Revolute
            robot_revolute, names_map, _ = robot.to_revolute()
            poses_revolute = {}
            input_revolute = list_to_variable_dict(input_list)
            for key in input_revolute:
                poses_revolute[key] = robot_revolute.get_pose(input_revolute, key)

            for key in input_pairs:
                pose_revolute = poses_revolute[names_map[key]].as_matrix()
                try:
                    self.assertTrue(
                        np.all(np.isclose(pose_revolute, poses[key].as_matrix()))
                    )
                except AssertionError:
                    print("Pose revolute: {:}".format(pose_revolute))
                    print("Pose standard: {:}".format(poses[key].as_matrix()))

                # Test angular limits
                try:
                    self.assertAlmostEqual(
                        robot.lb[key], robot_revolute.lb[names_map[key]]
                    )
                    self.assertAlmostEqual(
                        robot.ub[key], robot_revolute.ub[names_map[key]]
                    )
                except AssertionError:
                    print(
                        "Lower bound revolute: {:}".format(
                            robot_revolute.lb[names_map[key]]
                        )
                    )
                    print("Lower bound standard: {:}".format(robot.lb[key]))
                    print(
                        "Upper bound revolute: {:}".format(
                            robot_revolute.ub[names_map[key]]
                        )
                    )
                    print("Upper bound standard: {:}".format(robot.ub[key]))

    def test_random_tree(self):
        n_runs = 100
        height = 3
        gen = nx.balanced_tree(2, height, create_using=nx.DiGraph)
        gen = nx.relabel_nodes(gen, {node: f"p{node}" for node in gen})
        n = gen.number_of_edges()
        parents = nx.to_dict_of_lists(gen)
        d = list(np.random.rand(n) + 0.5)
        d_dict = list_to_variable_dict(d)
        theta = n * [0.0]
        theta_dict = list_to_variable_dict(theta)
        al = n * [0.0]
        al_dict = list_to_variable_dict(al)
        a = n * [0.0]
        a_dict = list_to_variable_dict(a)
        lim_u = list_to_variable_dict(np.random.rand(n) * np.pi)
        lim_l = list_to_variable_dict(np.random.rand(n) * np.pi)
        params = {
            "d": d_dict,
            "a": a_dict,
            "alpha": al_dict,
            "theta": theta_dict,
            "joint_limits_upper": lim_u,
            "joint_limits_lower": lim_l,
            "parents": parents,
        }
        robot = RobotSpherical(params)
        robot.lambdify_get_pose()

        for _ in range(n_runs):
            input_list = list(np.random.rand(2 * n))
            # poses_fast = robot.get_full_pose_fast_lambdify(input)

            poses = {}
            input_pairs = list_to_variable_dict_spherical(input_list, in_pairs=True)
            for key in input_pairs:
                poses[key] = robot.get_pose(input_pairs, key)

            # Revolute
            robot_revolute, names_map, _ = robot.to_revolute()
            poses_revolute = {}
            input_revolute = list_to_variable_dict(input_list)
            for key in input_revolute:
                poses_revolute[key] = robot_revolute.get_pose(input_revolute, key)

            for key in input_pairs:
                pose_revolute = poses_revolute[names_map[key]].as_matrix()
                # try:
                self.assertTrue(
                    np.all(np.isclose(pose_revolute, poses[key].as_matrix()))
                )
                # except AssertionError:
                #     print("Pose revolute: {:}".format(pose_revolute))
                #     print("Pose standard: {:}".format(poses[key].as_matrix()))

                # Test angular limits
                try:
                    self.assertAlmostEqual(
                        robot.lb[key], robot_revolute.lb[names_map[key]]
                    )
                    self.assertAlmostEqual(
                        robot.ub[key], robot_revolute.ub[names_map[key]]
                    )
                except AssertionError:
                    print(
                        "Lower bound revolute: {:}".format(
                            robot_revolute.lb[names_map[key]]
                        )
                    )
                    print("Lower bound standard: {:}".format(robot.lb[key]))
                    print(
                        "Upper bound revolute: {:}".format(
                            robot_revolute.ub[names_map[key]]
                        )
                    )
                    print("Upper bound standard: {:}".format(robot.ub[key]))


if __name__ == "__main__":
    unittest.main()
