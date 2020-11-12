import numpy as np
import numpy.linalg as la
import math
import unittest
import random

from graphik.solvers.solver_fabrik import solver_fabrik


class TestFabrik(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestFabrik, self).__init__(*args, **kwargs)
        self.max_iteration = 100
        self.n_tests = 1000

    def initialize_solver_plug(self, immediateParent=False):
        # Number of joints
        if immediateParent:
            N = 6
        else:
            N = 8

        # Dimension
        dim = 3

        # Link lengths
        r = []
        for i in range(N):
            r += [random.random() + 0.5]

        # Setting the joint angle limits
        angle_limit = []
        for i in range(N):
            angle_limit += [math.pi]

        # Consisted of the indices of the endeffectors
        if immediateParent:
            goal_index = [4, 5]
        else:
            goal_index = [5, 7]

        # The coordinatest of the goal of the endeffectors chosen in "goal_index" in the order at which they were inserted
        goal_position = [[2.5, 0.5, 0.5], [3, 0.5, 0.5]]

        # The index of the parent joint of each joint; the base joint(the 0th joint) has no parents and hence parents[0]=-1
        # NOTE: Make sure the index of the parent always preceeds that of its child joints; so joint[n]<n for all n.
        if immediateParent:
            parents = [-1, 0, 1, 2, 3, 3]
        else:
            parents = [-1, 0, 1, 2, 3, 4, 3, 6]

        params = {
            "N": N,
            "r": r,
            "parents": parents,
            "angle_limit": angle_limit,
            "goal_index": goal_index,
            "goal_position": goal_position,
            "dim": dim,
        }
        solver = solver_fabrik(params)

        return solver

    def initialize_solver_chain(self, dim):
        # Number of joints
        N = random.randint(3, 10)

        # Link lengths
        r = []
        for i in range(N):
            r += [random.random() + 0.5]

        # Setting the joint angle limits
        angle_limit = []
        for i in range(N):
            angle_limit += [math.pi]

        # Consisted of the indices of the endeffectors
        goal_index = [N - 1]

        # The coordinatest of the goal of the endeffectors chosen in "goal_index" in the order at which they were inserted
        goal_position = [[2.5, 0.5, 0.5 * (dim - 2)]]

        # The index of the parent joint of each joint; the base joint(the 0th joint) has no parents and hence parents[0]=-1
        # NOTE: Make sure the index of the parent always preceeds that of its child joints; so joint[n]<n for all n.
        parents = [-1]
        for i in range(N - 1):
            parents += [i]

        params = {
            "N": N,
            "r": r,
            "parents": parents,
            "angle_limit": angle_limit,
            "goal_index": goal_index,
            "goal_position": goal_position,
            "dim": dim,
        }
        solver = solver_fabrik(params)

        return solver

    def test_chain3d(self):
        n_tests = self.n_tests

        max_iteration = self.max_iteration
        initial_guess = None
        error_threshold = 0.010
        sensitivity = 0.0000001
        sensitivity_range = 5

        solver = self.initialize_solver_chain(dim=3)
        for i in range(n_tests):
            solver.goal_position = solver.generate_random_configuration()
            solution = solver.solve(
                initial_guess,
                max_iteration,
                error_threshold,
                sensitivity,
                sensitivity_range,
            )

            p = solution["positions"]
            for i in range(len(solver.goal_index)):
                distance = la.norm(solver.goal_position[i] - p[solver.goal_index[i], :])

                self.assertAlmostEqual(0, distance, delta=0.02)

    def test_chain2d(self):
        n_tests = self.n_tests

        max_iteration = self.max_iteration
        initial_guess = None
        error_threshold = 0.010
        sensitivity = 0.0000001
        sensitivity_range = 5

        solver = self.initialize_solver_chain(dim=2)
        for i in range(n_tests):
            solver.goal_position = solver.generate_random_configuration()
            solution = solver.solve(
                initial_guess,
                max_iteration,
                error_threshold,
                sensitivity,
                sensitivity_range,
            )

            p = solution["positions"]
            for i in range(len(solver.goal_index)):
                distance = la.norm(solver.goal_position[i] - p[solver.goal_index[i], :])

                self.assertAlmostEqual(0, distance, delta=0.2)

    def test_chain2d_orientation(self):
        n_tests = self.n_tests

        max_iteration = self.max_iteration
        initial_guess = None
        error_threshold = 0.001
        sensitivity = 0.0000001
        sensitivity_range = 5

        solver = self.initialize_solver_chain(dim=2)
        solver.goal_index = [solver.N - 1, solver.N - 2]

        for i in range(n_tests):
            configuration = solver.generate_random_configuration(
                returnAllPositions=True
            )
            solver.goal_position = [configuration[-1, :], configuration[-2, :]]

            solution = solver.solve(
                initial_guess,
                max_iteration,
                error_threshold,
                sensitivity,
                sensitivity_range,
            )

            p = solution["positions"]
            for i in range(len(solver.goal_index)):
                distance = la.norm(solver.goal_position[i] - p[solver.goal_index[i], :])

                self.assertAlmostEqual(0, distance, delta=0.2)

    def test_chain2d_orientation_angle_constraints(self):
        n_tests = self.n_tests

        max_iteration = self.max_iteration
        initial_guess = None
        error_threshold = 0.001
        sensitivity = 0.0000001
        sensitivity_range = 5

        solver = self.initialize_solver_chain(dim=2)
        solver.goal_index = [solver.N - 1, solver.N - 2]

        solver.angle_limit = [0]
        for i in range(solver.N - 1):
            solver.angle_limit += [np.random.random() * math.pi / 2 + math.pi / 2]

        for i in range(n_tests):
            configuration = solver.generate_random_configuration(
                returnAllPositions=True
            )
            solver.goal_position = [configuration[-1, :], configuration[-2, :]]

            solution = solver.solve(
                initial_guess,
                max_iteration,
                error_threshold,
                sensitivity,
                sensitivity_range,
            )

            a = solution["angles"]
            for i in range(1, solver.N):
                self.assertLess(a[i], solver.angle_limit[i] + 0.03)

    def test_plug(self):
        n_tests = self.n_tests

        max_iteration = self.max_iteration
        initial_guess = None
        error_threshold = 0.020
        sensitivity = 0.0000001
        sensitivity_range = 5

        solver = self.initialize_solver_plug(immediateParent=False)

        for i in range(n_tests):
            solver.goal_position = solver.generate_random_configuration()
            solution = solver.solve(
                initial_guess,
                max_iteration,
                error_threshold,
                sensitivity,
                sensitivity_range,
            )

            p = solution["positions"]
            for i in range(len(solver.goal_index)):
                distance = la.norm(solver.goal_position[i] - p[solver.goal_index[i], :])

                self.assertAlmostEqual(0, distance, delta=0.5)


if __name__ == "__main__":
    unittest.main()
