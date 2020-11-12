import numpy as np
import unittest

# from graphik.graphs.planar_robot_ik import PlanarRobot
from graphik.graphs.graph_base import SphericalRobotGraph
from graphik.robots.revolute import Revolute2dChain, Revolute2dTree
from graphik.solvers.solver_base import SdpRelaxationSolver
from graphik.solvers.constraints import nearest_neighbour_cost
from graphik.utils.utils import constraint_violations, list_to_variable_dict


class TestPlanarRobotIK(unittest.TestCase):
    @property
    def dim(self):
        return 2

    def test_simple_chain(self):
        link_lengths = list_to_variable_dict([1.0, 1.0])
        parent_nodes = None
        verbose = False
        as_equality = False
        force_dense = True
        eps_perturb = 1.9  # Starts to give the position vals trouble above 1.0 for as_equality = True
        # TODO: as_equality = False appears much more robust to eps_perturb!!! Make many plots (after more tests)
        angular_limits = list_to_variable_dict([np.pi, np.pi])
        end_effector_vals = {"p2": np.array([0.0, np.sqrt(2.0)])}
        nearest_neighbour_vals = {
            "p1": np.array([np.sqrt(2.0) / 2.0, np.sqrt(2.0) / 2.0])
        }
        nearest_neighbour_ang_vals = (
            {"p1": np.sqrt(2.0) / 2.0, "p2": np.sqrt(2.0) / 2.0}
            if as_equality
            else None
        )
        self.check_instance(
            link_lengths,
            parent_nodes,
            angular_limits,
            as_equality,
            verbose,
            force_dense,
            end_effector_vals,
            nearest_neighbour_vals,
            nearest_neighbour_ang_vals,
        )

    def test_simple_tree(self):
        link_lengths = list_to_variable_dict([1.0, 1.0, 1.0, 1.0])
        # parent_nodes = {1: 0, 2: 1, 3: 1, 4: 3}  # [0, 1, 1, 3]
        parent_nodes = {
            "p0": ["p1"],
            "p1": ["p2", "p3"],
            "p2": [],
            "p3": ["p4"],
            "p4": [],
        }
        verbose = False
        as_equality = False
        force_dense = False
        # Case 1 - no angular limits (all seems fine)
        angular_limits = list_to_variable_dict([np.pi, np.pi, np.pi, np.pi])
        end_effector_vals = np.array([[0.5, 0.5], [1.5, 2.4]])
        nearest_neighbour_vals = np.array([[0.0, 0.0], [1.0, 2.0]])

        end_effector_vals = {"p2": np.array([0.5, 1.5]), "p4": np.array([0.5, 2.4])}
        nearest_neighbour_vals = {
            "p1": np.array([0.0, 1.0]),
            "p3": np.array([0.0, 2.0]),
        }

        self.check_instance(
            link_lengths,
            parent_nodes,
            angular_limits,
            as_equality,
            verbose,
            force_dense,
            end_effector_vals,
            nearest_neighbour_vals,
        )

    def test_simple_tree_maxed(self):
        link_lengths = list_to_variable_dict([1.0, 1.0, 1.0, 1.0])
        parent_nodes = {
            "p0": ["p1"],
            "p1": ["p2", "p3"],
            "p2": [],
            "p3": ["p4"],
            "p4": [],
        }
        verbose = False
        as_equality = False
        force_dense = False
        angular_limits = list_to_variable_dict(
            [np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2]
        )
        # end_effector_vals = np.array([[1.0, 1.0], [1.0, 2.0]])
        # nearest_neighbour_vals = np.array([[0.5, 0.5],
        #                                    [1.0, 2.0]])
        # nearest_neighbour_vals = np.array([[0.0, 0.0], [1.0, 2.0]])
        end_effector_vals = {"p2": np.array([1.0, 1.0]), "p4": np.array([1.0, 2.0])}
        nearest_neighbour_vals = {
            "p1": np.array([0.0, 1.0]),
            "p3": np.array([0.0, 2.0]),
        }
        self.check_instance(
            link_lengths,
            parent_nodes,
            angular_limits,
            as_equality,
            verbose,
            force_dense,
            end_effector_vals,
            nearest_neighbour_vals,
        )

    def check_instance(
        self,
        link_lengths,
        parent_nodes,
        angular_limits,
        as_equality,
        verbose,
        force_dense,
        end_effector_vals,
        nearest_neighbour_vals,
        nearest_neighbour_ang_vals=None,
    ):
        params = {
            "theta": list_to_variable_dict(len(angular_limits) * [0.0]),
            "a": link_lengths,
        }
        if parent_nodes is not None:
            params["parents"] = parent_nodes
            robot = Revolute2dTree(params, leaves_only_end_effectors=True)
        else:
            robot = Revolute2dChain(params, leaves_only_end_effectors=True)
        robot_graph = SphericalRobotGraph(robot)
        # robot_graph.set_end_effectors(end_effector_vals)
        # robot_graph.set_angular_limits(angular_limits)

        solver = SdpRelaxationSolver(verbose=verbose, force_dense=force_dense)
        # solver.cost = robot_graph.nearest_neighbour_cost(
        #     nearest_neighbour_vals, nearest_angular_residuals=nearest_neighbour_ang_vals
        # )
        solver.cost = nearest_neighbour_cost(
            robot_graph, nearest_neighbour_vals, nearest_neighbour_ang_vals
        )
        prob_params = {
            "end_effector_assignment": end_effector_vals,
            "angular_limits": angular_limits,
            # "angular_offsets": None,
            "as_equality": as_equality,
        }
        solution_dict, ranks, prob, constraints = solver.solve(robot_graph, prob_params)
        violations = constraint_violations(constraints, solution_dict)
        # robot_graph.plot_solution(solution_dict)
        cost = solver.cost
        self.assertAlmostEqual(cost.subs(solution_dict), prob.value, places=6)
        eq_resid = [resid for (resid, is_eq) in violations if is_eq]
        ineq_resid = [resid for (resid, is_eq) in violations if not is_eq]
        # try:
        self.assertTrue(
            np.all(np.isclose(np.array(eq_resid, dtype=float), 0.0, atol=1e-6))
        )
        self.assertTrue(np.all(np.array(ineq_resid, dtype=float) >= 0.0))
        # except AssertionError:
        #     print("Equality residuals: {:}".format(eq_resid))
        #     print("Inequality residuals: {:}".format(ineq_resid))


if __name__ == "__main__":
    unittest.main()
