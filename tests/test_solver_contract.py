"""Shared-contract tests and cross-solver solve(T_goal) matrix."""
import numpy as np
import pytest

from graphik.solvers import (
    IKResult,
    IKSolver,
    JointAngleSolver,
    RiemannianSolver,
    ScipySolver,
)
from graphik.solvers.distance_solver import DistanceSolver
from tests.helpers import planar_chain


class _StubSolver(IKSolver):
    """Minimal concrete IKSolver for base-class helpers."""

    def solve(self, T_goal, **kwargs):
        return IKResult(
            q={},
            cost=0.0,
            iterations=0,
            time=0.0,
            status="stub",
            limit_violations=[],
        )


def test_ik_solver_is_abstract():
    robot, graph = planar_chain(4)
    with pytest.raises(TypeError):
        IKSolver(graph)


def test_feasible_iff_no_limit_violations():
    kw = dict(q={}, cost=0.0, iterations=0, time=0.0, status="")
    assert IKResult(limit_violations=[], **kw).feasible
    assert not IKResult(limit_violations=[{"edge": ("p1", "p2")}], **kw).feasible


def test_goals_from_wraps_single_transform_with_primary_ee():
    robot, graph = planar_chain(4)
    solver = _StubSolver(graph)
    T = np.eye(3)
    goals = solver.goals_from(T)
    assert goals == {robot.end_effectors[0]: T}
    assert robot.end_effectors[0] == f"p{robot.n}"


def test_goals_from_copies_dict_input():
    robot, graph = planar_chain(4)
    solver = _StubSolver(graph)
    original = {"p2": np.eye(3)}
    goals = solver.goals_from(original)
    assert goals == original
    assert goals is not original


def test_check_limits_empty_at_interior_configuration():
    robot, graph = planar_chain(4)
    solver = _StubSolver(graph)
    q = {f"p{i}": 0.1 for i in range(1, 5)}
    assert solver.check_limits(q) == []


SOLVER_FACTORIES = [
    pytest.param(lambda g: RiemannianSolver(g), id="riemannian"),
    pytest.param(lambda g: ScipySolver(g, method="BFGS"), id="scipy-bfgs"),
    pytest.param(lambda g: ScipySolver(g, method="L-BFGS-B"), id="scipy-lbfgsb"),
    pytest.param(lambda g: JointAngleSolver(g), id="joint-angle"),
]


@pytest.mark.parametrize("factory", SOLVER_FACTORIES)
def test_solve_returns_well_formed_result(factory):
    np.random.seed(42)
    robot, graph = planar_chain(6)
    solver = factory(graph)
    q_goal = robot.random_configuration()
    T_goal = np.asarray(robot.pose(q_goal, f"p{robot.n}"))

    result = solver.solve(T_goal)

    assert isinstance(result, IKResult)
    assert set(result.q.keys()) == set(q_goal.keys())
    assert isinstance(result.cost, float)
    assert result.iterations >= 0
    assert result.time > 0
    assert isinstance(result.status, str)
    assert isinstance(result.limit_violations, list)
    assert result.feasible == (len(result.limit_violations) == 0)
    if isinstance(solver, DistanceSolver):
        assert result.Y.shape == (graph.number_of_nodes(), graph.dim)
    else:
        assert result.Y is None


@pytest.mark.parametrize("factory", SOLVER_FACTORIES)
def test_solve_reaches_reachable_goal(factory):
    np.random.seed(42)
    robot, graph = planar_chain(6)
    solver = factory(graph)
    q_goal = robot.random_configuration()
    T_goal = np.asarray(robot.pose(q_goal, f"p{robot.n}"))

    result = solver.solve(T_goal)

    d = graph.dim
    T_sol = np.asarray(robot.pose(result.q, f"p{robot.n}"))
    assert np.linalg.norm(T_goal[:d, d] - T_sol[:d, d]) < 1e-2
    assert np.linalg.norm(T_goal[:d, :d] - T_sol[:d, :d]) < 1e-2
