"""Finite-difference checks for the analytical gradient and HVP exposed by
``NonlinearSolver.create_cost``.

After the loss-consolidation refactor there is a single dense backend; the
prior ``loop`` / ``sparse`` / ``dense`` test variants collapse to one.
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


def _numerical_hvp(Y, grad_fn, w, eps=1e-5):
    return (grad_fn(Y + eps * w) - grad_fn(Y - eps * w)) / (2 * eps)


class NonlinearSolverGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        robot, graph = load_ur10()
        rng = np.random.default_rng(0)
        joint_names = list(robot.random_configuration().keys())
        q = {name: float(rng.uniform(-np.pi, np.pi)) for name in joint_names}
        G = graph.realization(q)
        cls.graph = graph
        cls.omega = adjacency_matrix_from_graph(G)
        cls.D_goal = distance_matrix_from_pos(pos_from_graph(G))
        cls.dim = robot.dim
        cls.n_points = cls.omega.shape[0]
        cls.rng = np.random.default_rng(1)

    def test_gradient_matches_fd(self):
        solver = NonlinearSolver(self.graph)
        cost_and_grad, _ = solver.create_cost(self.D_goal, self.omega)
        for trial in range(5):
            Y = self.rng.standard_normal((self.n_points, self.dim)).ravel()
            _, g = cost_and_grad(Y)
            g_fd = _numerical_gradient(Y, cost_and_grad)
            np.testing.assert_allclose(
                g, g_fd, atol=1e-6,
                err_msg=f"trial {trial}: analytical gradient != FD",
            )

    def test_hvp_matches_fd_of_grad(self):
        solver = NonlinearSolver(self.graph)
        cost_and_grad, hessp = solver.create_cost(self.D_goal, self.omega)
        grad_only = lambda Y: cost_and_grad(Y)[1]
        for trial in range(5):
            Y = self.rng.standard_normal((self.n_points, self.dim)).ravel()
            w = self.rng.standard_normal(Y.size)
            hv = hessp(Y, w)
            hv_fd = _numerical_hvp(Y, grad_only, w)
            np.testing.assert_allclose(
                hv, hv_fd, atol=1e-3,
                err_msg=f"trial {trial}: analytical HVP != FD",
            )


if __name__ == "__main__":
    unittest.main()
