#!/usr/bin/env python3
import numpy as np

from graphik.graphs import ProblemGraph
from graphik.robots import Robot
from graphik.solvers import RiemannianSolver
from graphik.utils import list_to_variable_dict, pos_from_graph


def random_problem_2d_chain():
    e_rot = []
    e_pos = []
    n = 10
    fails = 0

    a = list_to_variable_dict(np.ones(n))
    th = list_to_variable_dict(np.zeros(n))
    angular_limits = np.array((n - 1) * [np.pi / 2] + [np.pi])
    upper_angular_limits = list_to_variable_dict(angular_limits)
    lower_angular_limits = list_to_variable_dict(-angular_limits)
    robot_params = {
        "link_lengths": a,
        "theta": th,
        "joint_limits_upper": upper_angular_limits,
        "joint_limits_lower": lower_angular_limits,
        "num_joints": 10,
    }

    robot = Robot({**robot_params, "dim": 2})
    graph = ProblemGraph(robot)
    solver = RiemannianSolver(graph)

    n_tests = 100
    q_init = list_to_variable_dict(n * [0])
    G_init = graph.realization(q_init)
    Y_init = pos_from_graph(G_init)

    t_sol = []
    for idx in range(n_tests):
        q_goal = robot.random_configuration()
        T_goal = np.asarray(robot.pose(q_goal, f"p{robot.n}"))

        sol = solver.solve(T_goal, Y_init=Y_init)
        q_sol = sol.q

        T_riemannian = np.asarray(robot.pose(q_sol, f"p{robot.n}"))
        err_riemannian_pos = np.linalg.norm(T_goal[:2, 2] - T_riemannian[:2, 2])
        R_delta = T_goal[:2, :2] @ T_riemannian[:2, :2].T
        err_riemannian_rot = abs(np.arctan2(R_delta[1, 0], R_delta[0, 0]))

        t_sol.append(sol.time)
        e_rot.append(err_riemannian_rot)
        e_pos.append(err_riemannian_pos)
        q_abs = np.abs(np.array(list(q_sol.values())))
        if err_riemannian_pos > 0.01 or err_riemannian_rot > 0.01:
            fails += 1
        elif sum(q_abs > (angular_limits + 0.01 * angular_limits)) > 0:
            print("FAIL")
            fails += 1
        print(f"{idx}", end="\r")

    t_sol = np.array(t_sol[1:])
    t_sol = t_sol[abs(t_sol - np.mean(t_sol)) < 2 * np.std(t_sol)]
    print("Average solution time {:}".format(np.average(t_sol)))
    print("Standard deviation of solution time {:}".format(np.std(np.array(t_sol))))
    print("Average pos error {:}".format(np.average(np.array(e_pos))))
    print("Average rot error {:}".format(np.average(np.array(e_rot))))
    print("Number of fails {:}".format(fails))


if __name__ == "__main__":
    np.random.seed(22)
    random_problem_2d_chain()
