"""Finite-difference parity tests for graphik.solvers.costs.

The dense backend is checked against an FD approximation of its own
gradient and HVP. egrad equals the analytical gradient of cost; ehvp
equals the analytical Hessian-vector product of cost (verified by FD
on egrad).
"""
from __future__ import annotations

import unittest

import numpy as np

from graphik.solvers import costs
from graphik.utils.dgp import distance_matrix_from_pos


def _fd_gradient(Y, fn, eps=1e-6):
    """Central-difference gradient of fn(Y) at Y. Y can be (N, d) or flat."""
    g = np.zeros_like(Y)
    perturb = np.zeros_like(Y)
    flat = perturb.reshape(-1)
    g_flat = g.reshape(-1)
    for i in range(flat.size):
        flat[i] = eps
        fp = fn(Y + perturb)
        fm = fn(Y - perturb)
        flat[i] = 0
        g_flat[i] = (fp - fm) / (2 * eps)
    return g


def _fd_jvp(Y, grad_fn, w, eps=1e-5):
    """Central-difference HVP via FD on grad: (grad(Y+eps*w) - grad(Y-eps*w)) / (2*eps)."""
    return (grad_fn(Y + eps * w) - grad_fn(Y - eps * w)) / (2 * eps)


def _make_problem(n=6, d=3, seed=0):
    rng = np.random.default_rng(seed)
    Y_target = rng.standard_normal((n, d))
    D_goal = distance_matrix_from_pos(Y_target)
    omega = (np.ones((n, n)) - np.eye(n)).astype(float)  # full graph
    return D_goal, omega, n, d, rng


class TestDenseEquality(unittest.TestCase):
    def setUp(self):
        self.D_goal, self.omega, self.n, self.d, self.rng = _make_problem(seed=0)
        self.cost, self.egrad, self.ehvp = costs._dense_equality(
            self.D_goal, self.omega, cache=True
        )

    def test_egrad_matches_fd_of_cost(self):
        for trial in range(3):
            Y = self.rng.standard_normal((self.n, self.d))
            g = self.egrad(Y)
            g_fd = _fd_gradient(Y, self.cost)
            np.testing.assert_allclose(g, g_fd, atol=1e-6,
                err_msg=f"trial {trial}: egrad != FD(cost)")

    def test_ehvp_matches_fd_of_egrad(self):
        for trial in range(3):
            Y = self.rng.standard_normal((self.n, self.d))
            w = self.rng.standard_normal((self.n, self.d))
            hv = self.ehvp(Y, w)
            hv_fd = _fd_jvp(Y, self.egrad, w)
            np.testing.assert_allclose(hv, hv_fd, atol=1e-4,
                err_msg=f"trial {trial}: ehvp != FD(egrad)")


class TestDenseLimits(unittest.TestCase):
    def setUp(self):
        self.D_goal, self.omega, self.n, self.d, self.rng = _make_problem(seed=1)
        # Use two distinct pairs so we exercise both active-set branches:
        # pair_L gets a wide lower bound (psi_L = 5x D_goal) that random N(0,1)
        # Y reliably violates (D < psi_L → A1 active, m4 contributes to ehvp).
        # pair_U gets a tight upper bound (psi_U = 0.2x D_goal) that random Y
        # reliably violates (D > psi_U → A2 active, m5 contributes to ehvp).
        # Both pairs are in omega; the test_egrad_matches_fd_of_cost assertion
        # validates the joint active-set behavior under FD.
        psi_L = np.zeros_like(self.D_goal)
        psi_U = np.zeros_like(self.D_goal)
        pair_L = (0, self.n - 1)
        pair_U = (1, self.n - 2)
        psi_L[pair_L] = psi_L[pair_L[::-1]] = 5.0 * self.D_goal[pair_L]
        psi_U[pair_U] = psi_U[pair_U[::-1]] = 0.2 * self.D_goal[pair_U]
        self.psi_L, self.psi_U = psi_L, psi_U
        self.cost, self.egrad, self.ehvp = costs._dense_limits(
            self.D_goal, self.omega, self.psi_L, self.psi_U, cache=True
        )

    def test_egrad_matches_fd_of_cost(self):
        for trial in range(3):
            Y = self.rng.standard_normal((self.n, self.d))
            g = self.egrad(Y)
            g_fd = _fd_gradient(Y, self.cost)
            np.testing.assert_allclose(g, g_fd, atol=1e-6,
                err_msg=f"trial {trial}: egrad != FD(cost)")

    def test_ehvp_matches_fd_of_egrad(self):
        for trial in range(3):
            Y = self.rng.standard_normal((self.n, self.d))
            w = self.rng.standard_normal((self.n, self.d))
            hv = self.ehvp(Y, w)
            hv_fd = _fd_jvp(Y, self.egrad, w)
            np.testing.assert_allclose(hv, hv_fd, atol=1e-4,
                err_msg=f"trial {trial}: ehvp != FD(egrad)")

    def test_active_set_exercises_both_branches(self):
        """Sanity check: across 3 trials, both A1 (lower-bound) and A2
        (upper-bound) branches are active at least once. If this regresses,
        the egrad/ehvp tests above stop covering the m4 / m5 paths and a
        sign bug in either branch becomes invisible.
        """
        # Use an independent rng (seed=99) so this check is not sensitive to
        # how many draws the two preceding tests consume from self.rng.
        rng = np.random.default_rng(99)
        a1_seen = a2_seen = False
        for _ in range(3):
            Y = rng.standard_normal((self.n, self.d))
            from graphik.utils.dgp import distance_matrix_from_pos
            D = distance_matrix_from_pos(Y)
            LL = (self.psi_L != self.psi_U) * (self.psi_L > 0)
            UU = (self.psi_L != self.psi_U) * (self.psi_U > 0)
            A1 = np.maximum(self.psi_L - LL * D, 0)
            A2 = -np.maximum(-self.psi_U + UU * D, 0)
            if (A1 > 0).any():
                a1_seen = True
            if (A2 < 0).any():
                a2_seen = True
        self.assertTrue(a1_seen, "no trial activated lower-bound branch (A1 > 0)")
        self.assertTrue(a2_seen, "no trial activated upper-bound branch (A2 < 0)")


class TestForRiemannian(unittest.TestCase):
    def setUp(self):
        self.D_goal, self.omega, self.n, self.d, self.rng = _make_problem(seed=4)

    def test_dense_dispatch_equality(self):
        cost, egrad, _ = costs.for_riemannian(self.D_goal, self.omega)
        ref_cost, ref_egrad, _ = costs._dense_equality(self.D_goal, self.omega, cache=True)
        Y = self.rng.standard_normal((self.n, self.d))
        np.testing.assert_allclose(cost(Y), ref_cost(Y), atol=1e-12, rtol=0)
        np.testing.assert_allclose(egrad(Y), ref_egrad(Y), atol=1e-12, rtol=0)

    def test_dense_dispatch_limits(self):
        psi_L = np.zeros_like(self.D_goal)
        psi_U = np.zeros_like(self.D_goal)
        pair_L = (0, self.n - 1)
        pair_U = (1, self.n - 2)
        psi_L[pair_L] = psi_L[pair_L[::-1]] = 5.0 * self.D_goal[pair_L]
        psi_U[pair_U] = psi_U[pair_U[::-1]] = 0.2 * self.D_goal[pair_U]
        cost, _, _ = costs.for_riemannian(
            self.D_goal, self.omega, psi_L=psi_L, psi_U=psi_U
        )
        ref_cost, _, _ = costs._dense_limits(
            self.D_goal, self.omega, psi_L, psi_U, cache=True
        )
        Y = self.rng.standard_normal((self.n, self.d))
        np.testing.assert_allclose(cost(Y), ref_cost(Y), atol=1e-12, rtol=0)


class TestForMinimize(unittest.TestCase):
    def setUp(self):
        self.D_goal, self.omega, self.n, self.d, self.rng = _make_problem(seed=5)

    def test_cost_and_grad_shapes(self):
        cost_and_grad, hessp = costs.for_minimize(
            self.D_goal, self.omega, dim=self.d
        )
        Y_flat = self.rng.standard_normal(self.n * self.d)
        f, g = cost_and_grad(Y_flat)
        self.assertIsInstance(f, float)
        self.assertEqual(g.shape, (self.n * self.d,))
        w_flat = self.rng.standard_normal(self.n * self.d)
        hv = hessp(Y_flat, w_flat)
        self.assertEqual(hv.shape, (self.n * self.d,))

    def test_cost_and_grad_matches_riemannian(self):
        # for_minimize must return the same numbers as for_riemannian, just flattened.
        cost, egrad, _ = costs.for_riemannian(self.D_goal, self.omega)
        cost_and_grad, _ = costs.for_minimize(
            self.D_goal, self.omega, dim=self.d
        )
        Y = self.rng.standard_normal((self.n, self.d))
        f_ref = cost(Y)
        g_ref = egrad(Y).ravel()
        f, g = cost_and_grad(Y.ravel())
        np.testing.assert_allclose(f, f_ref, atol=1e-12, rtol=0)
        np.testing.assert_allclose(g, g_ref, atol=1e-12, rtol=0)

    def test_limits_dispatch_shapes(self):
        psi_L = np.zeros_like(self.D_goal)
        psi_U = np.zeros_like(self.D_goal)
        pair_L = (0, self.n - 1)
        pair_U = (1, self.n - 2)
        psi_L[pair_L] = psi_L[pair_L[::-1]] = 5.0 * self.D_goal[pair_L]
        psi_U[pair_U] = psi_U[pair_U[::-1]] = 0.2 * self.D_goal[pair_U]
        cost_and_grad, hessp = costs.for_minimize(
            self.D_goal, self.omega, dim=self.d,
            psi_L=psi_L, psi_U=psi_U,
        )
        Y_flat = self.rng.standard_normal(self.n * self.d)
        f, g = cost_and_grad(Y_flat)
        self.assertIsInstance(f, float)
        self.assertEqual(g.shape, (self.n * self.d,))
        w_flat = self.rng.standard_normal(self.n * self.d)
        hv = hessp(Y_flat, w_flat)
        self.assertEqual(hv.shape, (self.n * self.d,))


class TestPoseCost(unittest.TestCase):
    """FD parity for the SE(n) log pose loss."""

    def test_cost_and_grad_matches_finite_differences_2d(self):
        from graphik.graphs import ProblemGraph
        from graphik.robots import Robot
        from graphik.utils.utils import list_to_variable_dict

        n = 4
        params = {
            "link_lengths": list_to_variable_dict(np.ones(n)),
            "theta": list_to_variable_dict(np.zeros(n)),
            "joint_limits_upper": list_to_variable_dict(np.pi * np.ones(n)),
            "joint_limits_lower": list_to_variable_dict(-np.pi * np.ones(n)),
            "num_joints": n,
            "dim": 2,
        }
        robot = Robot(params)
        ProblemGraph(robot)
        point = f"p{n}"
        q_goal = np.array([0.4, -0.3, 0.5, 0.2])
        T_goal = robot.pose(list_to_variable_dict(q_goal), point)
        cost_and_grad = costs.pose_cost(robot, point, T_goal)

        q = np.array([-0.2, 0.6, -0.1, 0.3])
        _, grad = cost_and_grad(q)

        eps = 1e-6
        fd = np.zeros(n)
        for i in range(n):
            dq = np.zeros(n)
            dq[i] = eps
            f_plus, _ = cost_and_grad(q + dq)
            f_minus, _ = cost_and_grad(q - dq)
            fd[i] = (f_plus - f_minus) / (2 * eps)

        np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-6)


class TestForMinimizeUR10(unittest.TestCase):
    """UR10-scale FD checks for the flat for_minimize interface."""

    @classmethod
    def setUpClass(cls):
        from graphik.utils.dgp import adjacency_matrix_from_graph, pos_from_graph
        from graphik.utils.roboturdf import load_ur10

        robot, graph = load_ur10()
        rng = np.random.default_rng(0)
        joint_names = list(robot.random_configuration().keys())
        q = {name: float(rng.uniform(-np.pi, np.pi)) for name in joint_names}
        G = graph.realization(q)
        cls.omega = adjacency_matrix_from_graph(G)
        cls.D_goal = distance_matrix_from_pos(pos_from_graph(G))
        cls.dim = robot.dim
        cls.n_points = cls.omega.shape[0]
        cls.rng = np.random.default_rng(1)

    def test_gradient_matches_fd(self):
        cost_and_grad, _ = costs.for_minimize(self.D_goal, self.omega, dim=self.dim)
        for trial in range(5):
            Y = self.rng.standard_normal((self.n_points, self.dim)).ravel()
            _, g = cost_and_grad(Y)
            g_fd = _fd_gradient(Y, lambda Yf: cost_and_grad(Yf)[0])
            np.testing.assert_allclose(
                g, g_fd, atol=1e-6,
                err_msg=f"trial {trial}: analytical gradient != FD",
            )

    def test_hvp_matches_fd_of_grad(self):
        cost_and_grad, hessp = costs.for_minimize(self.D_goal, self.omega, dim=self.dim)
        grad_only = lambda Y: cost_and_grad(Y)[1]
        for trial in range(5):
            Y = self.rng.standard_normal((self.n_points, self.dim)).ravel()
            w = self.rng.standard_normal(Y.size)
            hv = hessp(Y, w)
            hv_fd = _fd_jvp(Y, grad_only, w)
            np.testing.assert_allclose(
                hv, hv_fd, atol=1e-3,
                err_msg=f"trial {trial}: analytical HVP != FD",
            )


if __name__ == "__main__":
    unittest.main()
