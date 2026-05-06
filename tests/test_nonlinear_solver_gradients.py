"""Finite-difference checks for the analytical gradient and Hessian
exposed by ``NonlinearSolver.create_cost``.

Was previously a ``if __name__ == "__main__":`` block at the bottom of
``graphik/solvers/nonlinear_solver.py``; lifted here so it runs under
the regular test suite.
"""
from __future__ import annotations

import unittest

import numpy as np

from graphik.solvers.nonlinear_solver import NonlinearSolver
from graphik.utils.dgp import (
    adjacency_matrix_from_graph,
    distance_matrix_from_pos,
    pos_from_graph,
)
from graphik.utils.roboturdf import load_ur10


def _numerical_gradient(Y, f, eps=1e-6):
    g = np.zeros_like(Y)
    perturb = np.zeros_like(Y)
    for i in range(Y.size):
        perturb[i] = eps
        loss_plus, _ = f(Y + perturb)
        loss_minus, _ = f(Y - perturb)
        g[i] = (loss_plus - loss_minus) / (2 * eps)
        perturb[i] = 0
    return g


def _numerical_hessian(Y, f, eps=1e-5):
    H = np.zeros((Y.size, Y.size))
    for i in range(Y.size):
        Y[i] += eps
        _, plus = f(Y)
        Y[i] -= 2 * eps
        _, minus = f(Y)
        Y[i] += eps
        H[:, i] = (plus - minus) / (2 * eps)
    return H.ravel()


class NonlinearSolverGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        robot, graph = load_ur10()
        rng = np.random.default_rng(0)
        # Realize a random configuration to get a fully-known position
        # graph (every edge has a distance).
        joint_names = list(robot.random_configuration().keys())
        q = {name: float(rng.uniform(-np.pi, np.pi)) for name in joint_names}
        G = graph.realization(q)
        cls.graph = graph
        cls.omega = adjacency_matrix_from_graph(G)
        cls.D_goal = distance_matrix_from_pos(pos_from_graph(G))
        cls.dim = robot.dim
        cls.n_points = cls.omega.shape[0]
        cls.rng = np.random.default_rng(1)

    def _check_gradient(self, cost_type, atol=1e-6):
        solver = NonlinearSolver(self.graph, cost_type=cost_type)
        cost_and_grad, _, _ = solver.create_cost(self.D_goal, self.omega)
        f = lambda Y: cost_and_grad(Y.flatten())
        for trial in range(5):
            Y = self.rng.standard_normal((self.n_points, self.dim))
            _, g = f(Y)
            g_fd = _numerical_gradient(Y.flatten(), f)
            self.assertTrue(
                np.allclose(g, g_fd, atol=atol),
                f"{cost_type}: trial {trial} gradient mismatch",
            )

    def _check_hessian(self, cost_type, atol=1e-3):
        solver = NonlinearSolver(self.graph, cost_type=cost_type)
        cost_and_grad, _, hess = solver.create_cost(self.D_goal, self.omega)
        f = lambda Y: cost_and_grad(Y.flatten())
        for trial in range(5):
            Y = self.rng.standard_normal((self.n_points, self.dim))
            H = hess(Y.flatten()).flatten()
            H_fd = _numerical_hessian(Y.flatten(), f)
            self.assertTrue(
                np.allclose(H, H_fd, atol=atol),
                f"{cost_type}: trial {trial} Hessian mismatch",
            )

    def test_gradient_loop(self):
        self._check_gradient("loop")

    def test_gradient_sparse(self):
        self._check_gradient("sparse")

    def test_gradient_dense(self):
        self._check_gradient("dense")

    def test_hessian_loop(self):
        self._check_hessian("loop")

    def test_hessian_sparse(self):
        self._check_hessian("sparse")


if __name__ == "__main__":
    unittest.main()
