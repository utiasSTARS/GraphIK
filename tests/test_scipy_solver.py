"""Behavior tests for ScipySolver."""
import numpy as np

from graphik.solvers.scipy_solver import ScipySolver
from graphik.utils.constants import POS
from tests.helpers import planar_chain


def test_bfgs_reaches_reachable_goal():
    np.random.seed(42)
    robot, graph = planar_chain(6)
    solver = ScipySolver(graph, method="BFGS")
    q_goal = robot.random_configuration()
    T_goal = np.asarray(robot.pose(q_goal, f"p{robot.n}"))

    result = solver.solve(T_goal)

    d = graph.dim
    T_sol = np.asarray(robot.pose(result.q, f"p{robot.n}"))
    assert np.linalg.norm(T_goal[:d, d] - T_sol[:d, d]) < 1e-2
    assert np.linalg.norm(T_goal[:d, :d] - T_sol[:d, :d]) < 1e-2
    assert result.Y.shape == (graph.number_of_nodes(), d)


def test_position_constraints_pin_goal_nodes():
    np.random.seed(42)
    robot, graph = planar_chain(4)
    solver = ScipySolver(graph, method="L-BFGS-B")
    T_goal = np.asarray(robot.pose(robot.random_configuration(), f"p{robot.n}"))
    G = graph.from_pose(T_goal)
    bnds = solver.position_constraints(G)
    lb = bnds.lb.reshape(-1, graph.dim)

    pinned = {
        node: idx for idx, (node, data) in enumerate(G.nodes(data=True)) if POS in data
    }
    assert f"p{robot.n}" in pinned
    ee_row = lb[pinned[f"p{robot.n}"]]
    np.testing.assert_allclose(ee_row, T_goal[:2, 2], atol=1e-12)
    for idx in range(lb.shape[0]):
        if idx not in pinned.values():
            assert np.all(lb[idx] == -np.inf)


def test_lbfgsb_solution_has_pinned_ee():
    np.random.seed(42)
    robot, graph = planar_chain(6)
    solver = ScipySolver(graph, method="L-BFGS-B")
    q_goal = robot.random_configuration()
    T_goal = np.asarray(robot.pose(q_goal, f"p{robot.n}"))

    result = solver.solve(T_goal)

    node_index = list(graph.node_ids).index(f"p{robot.n}")
    np.testing.assert_allclose(result.Y[node_index], T_goal[:2, 2], atol=1e-12)
