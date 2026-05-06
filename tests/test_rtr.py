"""Layer-1 tests for graphik.solvers.rtr.

Tests use a Euclidean shim manifold so tCG and trust_regions can be
exercised on quadratics without dragging in PSDFixedRank.
"""
from __future__ import annotations

import unittest

import numpy as np

from graphik.solvers.rtr import (
    RTRResult,
    StopReason,
    _truncated_cg,
    trust_regions,
)


class _EuclideanShim:
    """Minimal Euclidean R^n manifold satisfying the rtr.py contract."""

    def __init__(self, n: int):
        self.dim = n
        self.typical_dist = float(np.sqrt(n))

    def inner_product(self, x, u, v):
        return float(np.dot(u, v))

    def norm(self, x, u):
        return float(np.linalg.norm(u))

    def projection(self, x, z):
        return z

    def to_tangent_space(self, x, u):
        return u

    def retraction(self, x, u):
        return x + u

    def zero_vector(self, x):
        return np.zeros_like(x)

    def random_point(self):
        return np.random.randn(self.dim)

    def random_tangent_vector(self, x):
        u = np.random.randn(self.dim)
        return u / np.linalg.norm(u)


def _quadratic(A: np.ndarray, b: np.ndarray):
    """Build cost / rgrad / rhess closures for f(x) = 0.5 x^T A x + b^T x.

    On the Euclidean shim, projection is identity, so rgrad == egrad and
    rhess == ehess.
    """

    def cost(x):
        return float(0.5 * x @ A @ x + b @ x)

    def rgrad(x):
        return A @ x + b

    def rhess(x, u):
        return A @ u

    return cost, rgrad, rhess


class TestTruncatedCG(unittest.TestCase):
    """Exercise each major exit path of _truncated_cg."""

    def test_reached_target(self):
        # Identity Hessian, large Delta, small kappa: tCG converges in
        # one step (residual is zero after step), exits via REACHED_TARGET.
        n = 4
        manifold = _EuclideanShim(n)
        A = np.eye(n)
        b = np.array([1.0, 2.0, 3.0, 4.0])
        x = np.zeros(n)
        _, rgrad, rhess = _quadratic(A, b)
        fgrad = rgrad(x)

        eta, Heta, j, stop = _truncated_cg(
            manifold, x, fgrad, rhess,
            Delta=10.0, theta=1.0, kappa=0.1, mininner=0, maxinner=20,
        )
        self.assertIn(stop, (StopReason.REACHED_TARGET_LINEAR,
                             StopReason.REACHED_TARGET_SUPERLINEAR))
        # Newton step on identity quadratic: eta = -A^-1 grad = -b
        np.testing.assert_allclose(eta, -b, atol=1e-10)

    def test_exceeded_tr(self):
        # Identity Hessian but Delta tiny: tCG hits the TR boundary on
        # the first step.
        n = 3
        manifold = _EuclideanShim(n)
        A = np.eye(n)
        b = np.array([1.0, 0.0, 0.0])
        x = np.zeros(n)
        _, rgrad, rhess = _quadratic(A, b)
        fgrad = rgrad(x)

        eta, _, _, stop = _truncated_cg(
            manifold, x, fgrad, rhess,
            Delta=0.25, theta=1.0, kappa=0.1, mininner=0, maxinner=20,
        )
        self.assertEqual(stop, StopReason.EXCEEDED_TR)
        self.assertAlmostEqual(np.linalg.norm(eta), 0.25, places=10)

    def test_negative_curvature(self):
        # Indefinite Hessian: first CG step encounters d_Hd < 0 and exits.
        n = 2
        manifold = _EuclideanShim(n)
        A = np.diag([1.0, -1.0])
        b = np.array([0.5, 0.5])
        x = np.zeros(n)
        _, rgrad, rhess = _quadratic(A, b)
        fgrad = rgrad(x)

        _, _, _, stop = _truncated_cg(
            manifold, x, fgrad, rhess,
            Delta=10.0, theta=1.0, kappa=0.1, mininner=0, maxinner=20,
        )
        self.assertEqual(stop, StopReason.NEGATIVE_CURVATURE)

    def test_max_inner_iter(self):
        # Tight inner cap forces MAX_INNER_ITER. Use an ill-conditioned SPD
        # Hessian so plain CG needs many steps to converge.
        n = 50
        rng = np.random.default_rng(0)
        Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
        eigs = np.linspace(1.0, 1e6, n)
        A = (Q * eigs) @ Q.T
        b = rng.standard_normal(n)
        manifold = _EuclideanShim(n)
        x = np.zeros(n)
        _, rgrad, rhess = _quadratic(A, b)
        fgrad = rgrad(x)

        _, _, j, stop = _truncated_cg(
            manifold, x, fgrad, rhess,
            Delta=1e9, theta=1.0, kappa=1e-12, mininner=0, maxinner=3,
        )
        self.assertEqual(stop, StopReason.MAX_INNER_ITER)
        self.assertEqual(j, 2)  # last index visited (0-indexed maxinner-1)


class TestTrustRegions(unittest.TestCase):
    """End-to-end RTR on Euclidean quadratics."""

    def test_converges_to_minimum(self):
        # Strictly convex quadratic: RTR should hit the minimum to
        # high precision in a small number of outer iterations.
        n = 5
        rng = np.random.default_rng(42)
        L = rng.standard_normal((n, n))
        A = L @ L.T + np.eye(n)  # SPD
        b = rng.standard_normal(n)
        # Analytic minimizer: x* = -A^-1 b
        x_star = -np.linalg.solve(A, b)

        manifold = _EuclideanShim(n)
        cost, rgrad, rhess = _quadratic(A, b)
        x0 = np.zeros(n)

        result = trust_regions(
            manifold, cost, rgrad, rhess, x0,
            max_iterations=200, min_gradient_norm=1e-10,
        )
        self.assertIsInstance(result, RTRResult)
        np.testing.assert_allclose(result.point, x_star, atol=1e-8)
        self.assertLess(result.gradient_norm, 1e-10)
        self.assertLess(result.iterations, 200)
        self.assertGreater(result.time, 0.0)
        self.assertIn("gradient norm", result.stopping_criterion)

    def test_max_iterations_stops(self):
        # Large cap and strict tolerance plus a single outer-iter cap:
        # RTR halts at the iteration ceiling, not at convergence.
        n = 5
        rng = np.random.default_rng(0)
        L = rng.standard_normal((n, n))
        A = L @ L.T + np.eye(n)
        b = rng.standard_normal(n)
        manifold = _EuclideanShim(n)
        cost, rgrad, rhess = _quadratic(A, b)

        result = trust_regions(
            manifold, cost, rgrad, rhess, np.zeros(n),
            max_iterations=1, min_gradient_norm=1e-30,
        )
        self.assertEqual(result.iterations, 1)
        self.assertIn("max_iterations", result.stopping_criterion)


if __name__ == "__main__":
    unittest.main()
