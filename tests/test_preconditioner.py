"""Contract and end-to-end tests for the tCG preconditioners."""
import numpy as np
import pytest

from graphik.solvers.riemannian_solver import (
    RiemannianSolver,
    make_gn_preconditioner,
    make_jacobi_preconditioner,
)
from graphik.utils.dgp import (
    adjacency_matrix_from_graph,
    bound_smoothing,
    distance_matrix_from_graph,
    graph_from_pos,
)
from graphik.utils.manifolds.fixed_rank_psd_sym import PSDFixedRank
from graphik.utils.roboturdf import load_ur10


FACTORIES = [make_gn_preconditioner, make_jacobi_preconditioner]


def _setup(seed=3, N=10, d=3):
    rng = np.random.default_rng(seed)
    manifold = PSDFixedRank(N, d)
    Y = rng.standard_normal((N, d))
    # A connected random edge mask (symmetric, zero diagonal)
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
    # Horizontal-space condition for the PSDFixedRank quotient: Y^T Z symmetric
    asym = Y.T @ z - z.T @ Y
    np.testing.assert_allclose(asym, 0.0, atol=1e-9)


@pytest.mark.parametrize("factory", FACTORIES)
def test_symmetric_positive_definite_on_tangent_space(factory):
    manifold, Y, omega, rng = _setup()
    precon = factory(manifold, omega)
    u = manifold.projection(Y, rng.standard_normal(Y.shape))
    v = manifold.projection(Y, rng.standard_normal(Y.shape))
    # Symmetry: <u, M^{-1} v> == <M^{-1} u, v>
    np.testing.assert_allclose(
        np.sum(u * precon(Y, v)), np.sum(precon(Y, u) * v), rtol=1e-9
    )
    # Positive definiteness along tangent directions
    assert np.sum(u * precon(Y, u)) > 0
    assert np.sum(v * precon(Y, v)) > 0


def test_gn_solve_matches_unpreconditioned_quality():
    np.random.seed(5)
    robot, graph = load_ur10()
    q = robot.random_configuration()
    T_goal = robot.pose(q, f"p{robot.n}")

    G = graph.from_pose(T_goal)
    D_goal = distance_matrix_from_graph(G)
    omega = adjacency_matrix_from_graph(G)
    bounds = bound_smoothing(G)

    solver = RiemannianSolver(graph)
    out = solver.solve(D_goal, omega, use_limits=True, bounds=bounds, precon="gn")
    G_sol = graph_from_pos(out["x"], graph.node_ids)
    q_sol = graph.joint_variables(G_sol, {f"p{robot.n}": T_goal})
    T_sol = robot.pose(q_sol, f"p{robot.n}")
    assert np.linalg.norm(T_sol[:3, 3] - T_goal[:3, 3]) < 1e-2
    assert np.linalg.norm(T_sol[:3, :3] - T_goal[:3, :3]) < 1e-2


def test_unknown_precon_raises():
    robot, graph = load_ur10()
    solver = RiemannianSolver(graph)
    with pytest.raises(ValueError):
        solver.solve(np.eye(4), np.zeros((4, 4)), precon="bogus")
