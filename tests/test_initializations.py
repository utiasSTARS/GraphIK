"""Behavior tests for the shared Y-initialization strategies."""
import numpy as np

from graphik.solvers.initializations import (
    INIT_STRATEGIES,
    bsmooth_init,
    spectral_init,
    zero_init,
)
from graphik.utils import POS
from graphik.utils.dgp import adjacency_matrix_from_graph, distance_matrix_from_graph
from graphik.utils.roboturdf import load_ur10


def _ur10_problem(seed=0):
    np.random.seed(seed)
    robot, graph = load_ur10()
    T_goal = robot.pose(robot.random_configuration(), f"p{robot.n}")
    G = graph.from_pose(T_goal)
    return graph, G, distance_matrix_from_graph(G), adjacency_matrix_from_graph(G)


def test_strategy_names():
    assert INIT_STRATEGIES == ("spectral", "bsmooth", "zero")


def test_spectral_shape_and_rank():
    graph, G, D_goal, omega = _ur10_problem()
    Y = spectral_init(D_goal, omega, graph.dim)
    assert Y.shape == (D_goal.shape[0], graph.dim)
    assert np.isfinite(Y).all()
    assert np.linalg.matrix_rank(Y) == graph.dim


def test_bsmooth_shape(seed=0):
    graph, G, D_goal, omega = _ur10_problem(seed)
    Y = bsmooth_init(G, omega, graph.dim)
    assert Y.shape == (D_goal.shape[0], graph.dim)
    assert np.isfinite(Y).all()


def test_zero_init_matches_zero_realization():
    graph, G, D_goal, omega = _ur10_problem()
    Y = zero_init(graph)
    G_zero = graph.realization(graph.robot.zero_configuration())
    expected = np.stack(
        [np.asarray(G_zero.nodes[node][POS], dtype=float) for node in graph.node_ids]
    )
    np.testing.assert_array_equal(Y, expected)
