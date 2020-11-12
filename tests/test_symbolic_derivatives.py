"""
Test the symbolic cost, grad, and Hessian computations. Mostly comparing to hand-computed values for simple robots,
some comparisons with a procedural cost function too.
Finally, one test checking that the BFGS solver is initialization dependent.
"""
import numpy as np
import sympy as sp
import unittest
from graphik.solvers.local_solver import LocalSolver
from graphik.robots.robot_base import RobotRevolute
from graphik.robots.revolute import Revolute2dTree, Revolute2dChain
from graphik.utils.utils import list_to_variable_dict
from graphik.graphs.graph_base import SphericalRobotGraph

from liegroups.numpy import SO2, SE2, SO3, SE3


class TestGradient(unittest.TestCase):
    def test_simplest_case(self):
        dim = 2
        link_lengths = list_to_variable_dict([1.0])
        parent_nodes = None
        upper_angular_limits = {"p1": np.pi}
        lower_angular_limits = {"p1": -np.pi}
        params = {
            "a": link_lengths,
            "theta": list_to_variable_dict(len(link_lengths) * [0.0]),
            "joint_limits_upper": upper_angular_limits,
            "joint_limits_lower": lower_angular_limits,
        }
        end_effector_assignment = {"p1": SE2(SO2.identity(), np.array([0.0, 0.0]))}
        initial_guess_good = {"p1": 0.4}
        robot = Revolute2dChain(params)
        f_cost, f_grad, f_hess = LocalSolver.symbolic_cost_function(
            robot, end_effector_assignment, initial_guess_good, use_trigsimp=True
        )
        angles = [sp.symbols("p1")]
        # print("Cost: {:}".format(f_cost(angles)))
        # print("Grad: {:}".format(f_grad(angles)))
        # print("Hess: {:}".format(f_hess(angles)))

        self.assertTrue(f_cost(angles) == 1.0)
        self.assertTrue(f_grad(angles)[0] == 0)
        self.assertTrue(f_hess(angles)[0] == 0)

    def test_simple_chain_by_inspection(self):
        dim = 2
        link_lengths = list_to_variable_dict([1.0, 1.0])
        parent_nodes = None
        upper_angular_limits = {"p1": np.pi, "p2": np.pi}
        lower_angular_limits = {"p1": -np.pi, "p2": -np.pi}
        params = {
            "a": link_lengths,
            "theta": list_to_variable_dict(len(link_lengths) * [0.0]),
            "joint_limits_upper": upper_angular_limits,
            "joint_limits_lower": lower_angular_limits,
        }
        end_effector_assignment = {"p2": SE2(SO2.identity(), np.array([0.0, 0.0]))}
        initial_guess_good = {"p1": 0.4, "p2": -2.0}
        initial_guess_bad = {"p1": -0.6, "p2": 1.0}
        robot = Revolute2dChain(params)
        f_cost, f_grad, f_hess = LocalSolver.symbolic_cost_function(
            robot, end_effector_assignment, initial_guess_good, use_trigsimp=True
        )
        angles = [sp.symbols("p1"), sp.symbols("p2")]
        # print("Cost: {:}".format(f_cost(angles)))
        # print("Grad: {:}".format(f_grad(angles)))
        # print("Hess: {:}".format(f_hess(angles)))

        # Only depends on angle p2 due to symmetry
        self.assertTrue(f_cost(angles) == 2 * sp.cos(angles[1]) + 2.0)
        self.assertTrue(f_grad(angles)[1] == -2 * sp.sin(angles[1]))
        self.assertTrue(f_grad(angles)[0] == 0)
        self.assertTrue(f_hess(angles)[1, 1] == -2 * sp.cos(angles[1]))
        self.assertTrue(f_hess(angles)[0, 0] == 0)
        self.assertTrue(f_hess(angles)[0, 1] == 0)
        self.assertTrue(f_hess(angles)[1, 0] == 0)

    def test_bfgs_solver(self):
        # Try the simplest case, see if local minimum induced by angular limits 'captures' local method's progress
        dim = 2
        link_lengths = list_to_variable_dict([1.0, 1.0])
        parent_nodes = None
        upper_angular_limits = {"p1": np.pi / 4.0, "p2": np.pi}
        lower_angular_limits = {"p1": -np.pi / 4.0, "p2": -np.pi}
        params = {
            "a": link_lengths,
            "theta": list_to_variable_dict(len(link_lengths) * [0.0]),
            "joint_limits_upper": upper_angular_limits,
            "joint_limits_lower": lower_angular_limits,
        }

        end_effector_assignment = {"p2": SE2(SO2.identity(), np.array([1.2, -0.3]))}

        initial_guess_good = {"p1": 0.4, "p2": -2.0}
        initial_guess_bad = {"p1": -0.6, "p2": 1.0}

        robot = Revolute2dChain(params)

        graph = SphericalRobotGraph(robot)
        solver = LocalSolver()

        # Get symbolic results
        solver.set_symbolic_cost_function(
            robot, end_effector_assignment, upper_angular_limits.keys()
        )
        problem_params = {
            "angular_limits": upper_angular_limits,
            "initial_guess": initial_guess_good,
        }
        results_good = solver.solve(graph, problem_params)
        self.assertTrue(np.abs(results_good.fun) < 1e-9)
        problem_params["initial_guess"] = initial_guess_bad
        results_bad = solver.solve(graph, problem_params)
        self.assertTrue(np.abs(results_bad.fun) > 0.13)

    def test_random_symbolic_costs(self):
        seed = 8675309
        np.random.seed(seed)
        n = 3
        N_runs = 10
        N_angle_sets = 20
        dim = 2
        upper_angular_limits = list_to_variable_dict(n * [np.pi])
        lower_angular_limits = list_to_variable_dict(n * [-np.pi])

        for idx in range(N_runs):
            # print("{:}/{:} runs".format(idx+1, N_runs))
            link_lengths = list_to_variable_dict(list(np.random.rand(n) + 0.2))
            params = {
                "a": link_lengths,
                "theta": list_to_variable_dict(len(link_lengths) * [0.0]),
                "joint_limits_upper": upper_angular_limits,
                "joint_limits_lower": lower_angular_limits,
            }

            end_effector_assignment = {"p" + str(n): SE2(SO2.identity(), np.random.rand(dim))}
            robot = Revolute2dChain(params)

            solver1 = LocalSolver()
            solver1.set_symbolic_cost_function(
                robot, end_effector_assignment, upper_angular_limits.keys()
            )

            solver2 = LocalSolver()
            # Get symbolic results
            solver2.set_procedural_cost_function(robot, end_effector_assignment)

            for jdx in range(N_angle_sets):
                # print("{:}/{:} angle sets".format(jdx + 1, N_angle_sets))
                angles = np.random.rand(n) * 2.0 * np.pi - np.pi
                f_symb_val = solver1.f_cost(angles)
                f_proc_val = solver2.f_cost(angles)
                self.assertAlmostEqual(f_symb_val, f_proc_val)

    def test_2d_tree(self):
        pass

    def test_3d_chain(self):
        # Number of DOF's used
        n = 3
        N_runs = 100
        dim = 3
        # Modified (that's why a and al are 1 shorter) DH params theta, d, alpha, a
        a = [1.0, -0.612, 0.5723, 0, 0, 0]
        d = [0.1273, 0, 0, 0.1639, 0.1157, 0.0922]
        al = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]
        th = [0, 0, 0, 0, 0, 0]
        params = {"a": a, "alpha": al, "d": d, "theta": th}
        robot = RobotRevolute(params)
        # graph = SpatialRobotGraph(robot, as_equality=True)

        end_effector_assignment = {
            "p" + str(n): SE3(SO3.identity(), np.random.rand(dim) * n * 2.0 - n * 1.0)
        }
        angle_joints = ["p" + str(idx + 1) for idx in range(n)]

        solver1 = LocalSolver()
        solver1.set_symbolic_cost_function(robot, end_effector_assignment, angle_joints)

        solver2 = LocalSolver()
        # Get symbolic results
        solver2.set_procedural_cost_function(robot, end_effector_assignment)

        for idx in range(N_runs):
            angles = np.random.rand(n) * 2.0 * np.pi - np.pi
            f_symb_val = solver1.f_cost(angles)
            f_proc_val = solver2.f_cost(angles)
            self.assertAlmostEqual(f_symb_val, f_proc_val)


if __name__ == "__main__":
    unittest.main()
