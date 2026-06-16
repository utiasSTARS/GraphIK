"""Behavior tests for JointAngleSolver (joint-space SLSQP solver)."""
from types import SimpleNamespace

import numpy as np

import graphik.solvers.joint_angle as joint_angle_mod
from graphik.solvers import IKResult, JointAngleSolver
from graphik.utils.utils import list_to_variable_dict
from tests.helpers import planar_chain


class TestMultiGoalSolve:
    def test_solve_optimizes_all_goals_not_just_the_last(self):
        """Two consistent goals through the shared solve(T_goal) signature."""
        n = 6
        robot, graph = planar_chain(n)
        solver = JointAngleSolver(graph)

        q_goal = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        q_goal_dict = list_to_variable_dict(q_goal)
        T_ee = robot.pose(q_goal_dict, f"p{n}")
        T_mid = robot.pose(q_goal_dict, "p3")
        goals = {f"p{n}": T_ee, "p3": T_mid}

        q0 = np.concatenate([q_goal[:3], np.zeros(3)])
        result = solver.solve(goals, q_init=list_to_variable_dict(q0))

        assert isinstance(result, IKResult)
        err_ee = np.linalg.norm(robot.pose(result.q, f"p{n}") - T_ee)
        err_mid = np.linalg.norm(robot.pose(result.q, "p3") - T_mid)
        assert err_mid < 1e-3
        assert err_ee < 1e-3


class TestSharedContract:
    def test_q_init_defaults_to_zero_configuration(self):
        n = 4
        robot, graph = planar_chain(n)
        solver = JointAngleSolver(graph)
        q_goal = list_to_variable_dict(np.array([0.3, -0.2, 0.4, 0.1]))
        T_goal = np.asarray(robot.pose(q_goal, f"p{n}"))

        result = solver.solve(T_goal)

        T_sol = np.asarray(robot.pose(result.q, f"p{n}"))
        assert np.linalg.norm(T_sol - T_goal) < 1e-3

    def test_result_fields(self):
        n = 4
        robot, graph = planar_chain(n)
        solver = JointAngleSolver(graph)
        T_goal = np.asarray(
            robot.pose(
                list_to_variable_dict(np.array([0.4, -0.3, 0.5, 0.2])),
                f"p{n}",
            )
        )
        result = solver.solve(T_goal)
        assert result.Y is None
        assert isinstance(result.cost, float)
        assert isinstance(result.status, str)
        assert result.iterations >= 1
        assert result.time > 0
        assert isinstance(result.limit_violations, list)

    def test_q_init_is_ordered_by_joint_name_not_dict_insertion_order(self, monkeypatch):
        n = 4
        robot, graph = planar_chain(n)
        solver = JointAngleSolver(graph)
        q_goal = list_to_variable_dict(np.array([0.2, -0.1, 0.3, -0.2]))
        T_goal = np.asarray(robot.pose(q_goal, f"p{n}"))
        q_init = list_to_variable_dict(np.array([0.4, -0.3, 0.2, -0.1]))
        shuffled = {joint: q_init[joint] for joint in reversed(list(q_init))}
        captured = {}

        def fake_minimize(fun, x0, **kwargs):
            captured["x0"] = np.asarray(x0)
            return SimpleNamespace(
                x=np.asarray(x0), fun=0.0, nit=0, message="fake"
            )

        monkeypatch.setattr(joint_angle_mod, "minimize", fake_minimize)

        result = solver.solve(T_goal, q_init=shuffled)

        expected = np.array([q_init[joint] for joint in solver.joint_order])
        np.testing.assert_array_equal(captured["x0"], expected)
        assert result.q == {
            joint: value for joint, value in zip(solver.joint_order, expected)
        }
