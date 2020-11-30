import unittest
import numpy as np
import sympy as sp

from graphik.robots.robot_base import RobotSpherical
from graphik.solvers.local_solver import LocalSolver
from graphik.graphs.graph_base import SphericalRobotGraph
from graphik.utils.utils import list_to_variable_dict, list_to_variable_dict_spherical

from liegroups.numpy import SO3, SE3


class TestParameterization(unittest.TestCase):
    def test_1dof_simple(self):
        a = [0.0, 0.0]
        d = [1.0, 1.0]
        theta = [0.0, 0.0]
        al = [0.0, 0.0]
        lb = [-np.pi / 2, -np.pi / 2]
        ub = [np.pi / 2, np.pi / 2]
        params = {
            "a": list_to_variable_dict(a),
            "alpha": list_to_variable_dict(al),
            "d": list_to_variable_dict(d),
            "theta": list_to_variable_dict(theta),
            "joint_limits_upper": list_to_variable_dict(ub),
            "joint_limits_lower": list_to_variable_dict(lb),
        }
        robot = RobotSpherical(params)
        # robot_rev = Revolute3dChain(params)
        solver_params = {"solver": "L-BFGS-B", "maxiter": 100, "tol": 1e-6}
        solver = LocalSolver(solver_params)
        end_effector_assignment = {"p2": SE3(SO3.identity(), np.array([1.0, 1.0, 1.0]))}
        variables = ["p1", "p2"]
        # solver.set_symbolic_cost_function(robot_rev, end_effector_assignment, variables)
        solver.set_symbolic_cost_function(robot, end_effector_assignment, variables)

        graph = SphericalRobotGraph(robot)
        problem_params = {
            "angular_limits": graph.robot.ub,
            "initial_guess": list_to_variable_dict([[0.0, 0.0], [0.0, 0.0]]),
        }
        results = solver.solve(graph, problem_params)
        # print(results)
        t = robot.get_pose(
            list_to_variable_dict([results.x[0:2], results.x[2:]]), "p2"
        ).trans
        self.assertTrue(np.allclose(end_effector_assignment["p2"].trans, t))

    def test_random_cases(self):
        N_runs = 10
        # np.random.seed(1234567)
        for _ in range(N_runs):
            n = np.random.randint(2, 3)
            a = n * [0.0]
            d = np.random.rand(n) + 0.5
            theta = n * [0.0]
            al = n * [0.0]
            ub = np.random.rand(n) * (np.pi - 0.2) + 0.2
            lb = -ub
            params = {
                "a": list_to_variable_dict(a),
                "alpha": list_to_variable_dict(al),
                "d": list_to_variable_dict(d),
                "theta": list_to_variable_dict(theta),
                "joint_limits_upper": list_to_variable_dict(ub),
                "joint_limits_lower": list_to_variable_dict(lb),
            }
            robot = RobotSpherical(params)
            # solver_params = {"solver": "L-BFGS-B", "maxiter": 100, "tol": 1e-9}
            solver_params = {"solver": "trust-constr", "maxiter": 100, "tol": 1e-9}
            solver = LocalSolver(solver_params)
            goal_pose = robot.get_pose(robot.random_configuration(), "p" + str(n))
            end_effector_assignment = {"p" + str(n): goal_pose}
            variables = ["p" + str(idx) for idx in range(1, n + 1)]
            solver.set_symbolic_cost_function(robot, end_effector_assignment, variables)
            graph = SphericalRobotGraph(robot)
            problem_params = {
                "angular_limits": graph.robot.ub,
                "initial_guess": list_to_variable_dict(n * [[0.0, 0.0]]),
            }
            results = solver.solve(graph, problem_params)
            x_list = [
                list(results.x[n : n + 2])
                for n, val in enumerate(results.x)
                if n % 2 == 0
            ]
            t = robot.get_pose(list_to_variable_dict(x_list), "p" + str(n)).trans
            self.assertTrue(
                np.allclose(goal_pose.trans, t, rtol=1e-5, atol=1e-3),
                msg=f"Goal position {goal_pose} vs. solved position {t}",
            )


if __name__ == "__main__":
    unittest.main()
