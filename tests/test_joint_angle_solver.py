"""Behavior tests for LocalSolver (joint-space SLSQP solver)."""
import numpy as np
import pytest

from graphik.robots import Robot
from graphik.graphs import ProblemGraph
from graphik.solvers.joint_angle_solver import LocalSolver
from graphik.utils.utils import list_to_variable_dict


def planar_chain(n):
    params = {
        "link_lengths": list_to_variable_dict(np.ones(n)),
        "theta": list_to_variable_dict(np.zeros(n)),
        "joint_limits_upper": list_to_variable_dict(np.pi * np.ones(n)),
        "joint_limits_lower": list_to_variable_dict(-np.pi * np.ones(n)),
        "num_joints": n,
        "dim": 2,
    }
    robot = Robot(params)
    return robot, ProblemGraph(robot)


class TestGradientConsistency2d:
    def test_cost_and_grad_matches_finite_differences(self):
        """The 2D branch must use the true SE2 inverse left Jacobian; a
        stubbed identity makes the returned gradient inconsistent with the
        returned cost away from the goal."""
        n = 4
        robot, graph = planar_chain(n)
        solver = LocalSolver(graph, {})
        point = f"p{n}"

        q_goal = np.array([0.4, -0.3, 0.5, 0.2])
        T_goal = robot.pose(list_to_variable_dict(q_goal), point)
        cost_and_grad = solver.gen_cost_and_grad_ee(point, T_goal)

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


class TestMultiGoalSolve:
    def test_solve_optimizes_all_goals_not_just_the_last(self):
        """Two consistent goals (intermediate point and end effector, both
        from the same configuration). If solve() drops every goal but the
        last dict entry, the end-effector goal here is silently ignored."""
        n = 6
        robot, graph = planar_chain(n)
        solver = LocalSolver(graph, {})

        q_goal = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        q_goal_dict = list_to_variable_dict(q_goal)
        T_ee = robot.pose(q_goal_dict, f"p{n}")
        T_mid = robot.pose(q_goal_dict, "p3")
        # End-effector goal first, intermediate goal last: the historical
        # bug kept only the last entry.
        goals = {f"p{n}": T_ee, "p3": T_mid}

        # Start with the first three joints already at the goal, the rest
        # off: the p3 cost alone is zero here, so a solver that ignores the
        # p{n} goal terminates immediately.
        q0 = np.concatenate([q_goal[:3], np.zeros(3)])
        res = solver.solve(goals, list_to_variable_dict(q0))

        q_sol = list_to_variable_dict(res.x)
        err_ee = np.linalg.norm(robot.pose(q_sol, f"p{n}") - T_ee)
        err_mid = np.linalg.norm(robot.pose(q_sol, "p3") - T_mid)
        assert err_mid < 1e-3
        assert err_ee < 1e-3
