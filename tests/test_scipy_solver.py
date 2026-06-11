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


def test_position_constraints_pin_exactly_the_pos_tagged_nodes():
    robot, graph = planar_chain(4)
    solver = ScipySolver(graph, method="L-BFGS-B")
    bnds = solver.position_constraints(graph)
    lb = bnds.lb.reshape(-1, graph.dim)
    ub = bnds.ub.reshape(-1, graph.dim)
    for idx, (node, data) in enumerate(graph.nodes(data=True)):
        if POS in data:
            np.testing.assert_array_equal(lb[idx], data[POS])
            np.testing.assert_array_equal(ub[idx], data[POS])
        else:
            assert np.all(lb[idx] == -np.inf) and np.all(ub[idx] == np.inf)
