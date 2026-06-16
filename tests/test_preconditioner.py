"""Contract and end-to-end tests for the tCG preconditioners."""
import numpy as np
import pytest

from graphik.solvers.riemannian import (
    RiemannianSolver,
    make_gn_preconditioner,
    make_jacobi_preconditioner,
)
from graphik.utils.manifolds.fixed_rank_psd_sym import PSDFixedRank
from graphik.utils.roboturdf import load_ur10


FACTORIES = [make_gn_preconditioner, make_jacobi_preconditioner]


def _setup(seed=3, N=10, d=3):
    rng = np.random.default_rng(seed)
    manifold = PSDFixedRank(N, d)
    Y = rng.standard_normal((N, d))
    omega = (rng.random((N, N)) < 0.5).astype(float)
    omega = np.triu(omega, 1)
    omega += omega.T
    return manifold, Y, omega, rng


@pytest.mark.parametrize("factory", FACTORIES)
def test_output_is_tangent(factory):
    manifold, Y, omega, rng = _setup()
    precon = factory(manifold, omega)
    r = manifold.projection(Y, rng.standard_normal(Y.shape))
    z = precon(Y, r)
    asym = Y.T @ z - z.T @ Y
    np.testing.assert_allclose(asym, 0.0, atol=1e-9)


@pytest.mark.parametrize("factory", FACTORIES)
def test_symmetric_positive_definite_on_tangent_space(factory):
    manifold, Y, omega, rng = _setup()
    precon = factory(manifold, omega)
    u = manifold.projection(Y, rng.standard_normal(Y.shape))
    v = manifold.projection(Y, rng.standard_normal(Y.shape))
    np.testing.assert_allclose(
        np.sum(u * precon(Y, v)), np.sum(precon(Y, u) * v), rtol=1e-9
    )
    assert np.sum(u * precon(Y, u)) > 0
    assert np.sum(v * precon(Y, v)) > 0


def test_gn_solve_matches_unpreconditioned_quality():
    np.random.seed(5)
    robot, graph = load_ur10()
    q = robot.random_configuration()
    T_goal = np.asarray(robot.pose(q, f"p{robot.n}"))

    solver = RiemannianSolver(graph, precon="gn")
    result = solver.solve(T_goal)

    T_sol = np.asarray(robot.pose(result.q, f"p{robot.n}"))
    assert np.linalg.norm(T_sol[:3, 3] - T_goal[:3, 3]) < 1e-2
    assert np.linalg.norm(T_sol[:3, :3] - T_goal[:3, :3]) < 1e-2


def test_unknown_precon_raises():
    robot, graph = load_ur10()
    with pytest.raises(ValueError):
        RiemannianSolver(graph, precon="bogus")
