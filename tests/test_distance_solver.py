"""Pipeline tests for the DistanceSolver mid-layer."""
import numpy as np
import pytest

from graphik.solvers.distance_solver import (
    DistanceProblem,
    DistanceSolver,
    MinimizeInfo,
)
from graphik.solvers.initializations import zero_init
from tests.helpers import planar_chain


class _IdentityMinimizer(DistanceSolver):
    """Returns Y0 untouched; no optimizer is involved."""

    def _minimize(self, problem: DistanceProblem):
        self.last_problem = problem
        return problem.Y0, MinimizeInfo(cost=0.0, iterations=1, status="identity")


def test_invalid_init_rejected_at_construction():
    robot, graph = planar_chain(4)
    with pytest.raises(ValueError):
        _IdentityMinimizer(graph, init="bogus")


def test_pipeline_decodes_exact_realization():
    robot, graph = planar_chain(4)
    solver = _IdentityMinimizer(graph)
    q_zero = robot.zero_configuration()
    T_goal = np.asarray(robot.pose(q_zero, f"p{robot.n}"))
    Y0 = zero_init(graph)

    result = solver.solve(T_goal, Y_init=Y0)

    assert result.Y is Y0
    assert result.status == "identity"
    assert result.iterations == 1
    assert result.time > 0
    assert set(result.q) == set(q_zero)
    np.testing.assert_allclose(np.array(list(result.q.values())), 0.0, atol=1e-8)
    assert result.feasible


def test_problem_state_is_complete():
    robot, graph = planar_chain(4)
    solver = _IdentityMinimizer(graph)
    T_goal = np.asarray(robot.pose(robot.zero_configuration(), f"p{robot.n}"))
    solver.solve(T_goal, Y_init=zero_init(graph))

    p = solver.last_problem
    N = graph.number_of_nodes()
    assert p.D_goal.shape == (N, N)
    assert p.omega.shape == (N, N)
    assert p.psi_L is not None and p.psi_U is not None
    assert p.Y0.shape == (N, graph.dim)


def test_use_limits_false_skips_bound_matrices():
    robot, graph = planar_chain(4)
    solver = _IdentityMinimizer(graph, use_limits=False)
    T_goal = np.asarray(robot.pose(robot.zero_configuration(), f"p{robot.n}"))
    solver.solve(T_goal, Y_init=zero_init(graph))
    assert solver.last_problem.psi_L is None
    assert solver.last_problem.psi_U is None
