"""Contract tests for the PSDFixedRank quotient manifold.

The projection onto the horizontal space at Y must be idempotent and land in
the horizontal space {U : Y^T U symmetric}; the hand-coded Lyapunov matrix must
represent the operator Omega -> X@Omega + Omega@X (row-major vec). Both are
checked for k=2 (planar problems) and k=3.
"""
import numpy as np
import pytest

from graphik.utils.manifolds.fixed_rank_psd_sym import PSDFixedRank


@pytest.fixture(params=[2, 3], ids=["k2", "k3"])
def k(request):
    return request.param


def _random_Y(n, k, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, k))


def test_lyapunov_matrix_represents_sylvester_operator(k):
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((4, k))
    X = Y.T @ Y
    A = PSDFixedRank._build_lyap_matrix(X)
    for seed in range(3):
        Omega = np.random.default_rng(seed).standard_normal((k, k))
        expected = X @ Omega + Omega @ X
        np.testing.assert_allclose(
            A @ Omega.ravel(), expected.ravel(), atol=1e-12,
            err_msg=f"Lyapunov matrix wrong for k={k}",
        )


def test_projection_is_idempotent(k):
    n = 8
    man = PSDFixedRank(n, k)
    Y = _random_Y(n, k, seed=1)
    Z = _random_Y(n, k, seed=2)
    P1 = man.projection(Y, Z)
    P2 = man.projection(Y, P1)
    np.testing.assert_allclose(P2, P1, atol=1e-10)


def test_projection_output_is_horizontal(k):
    n = 8
    man = PSDFixedRank(n, k)
    Y = _random_Y(n, k, seed=3)
    Z = _random_Y(n, k, seed=4)
    P = man.projection(Y, Z)
    skew_part = Y.T @ P - P.T @ Y
    np.testing.assert_allclose(skew_part, np.zeros((k, k)), atol=1e-10)
