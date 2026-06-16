#!/usr/bin/env python3
import numpy as np

from graphik.graphs import ProblemGraph
from graphik.robots import Robot
from graphik.solvers import RiemannianSolver
from graphik.utils import list_to_variable_dict, pos_from_graph


def random_problem_2d_chain():
    e_rot = []
    e_pos = []
    fails = 0
    n = 10

    a = list_to_variable_dict(np.ones(n))
    th = list_to_variable_dict(np.zeros(n))
    lim_u = list_to_variable_dict(np.pi * np.ones(n))
    lim_l = list_to_variable_dict(-np.pi * np.ones(n))
    robot_params = {
        "link_lengths": a,
        "theta": th,
        "joint_limits_upper": lim_u,
        "joint_limits_lower": lim_l,
        "num_joints": 10,
    }

    robot = Robot({**robot_params, "dim": 2})
    graph = ProblemGraph(robot)
    solver = RiemannianSolver(graph, use_limits=False)

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
        if err_riemannian_pos > 0.01 or err_riemannian_rot > 0.01:
            fails += 1

    t_sol = np.array(t_sol[1:])
    t_sol = t_sol[abs(t_sol - np.mean(t_sol)) < 2 * np.std(t_sol)]
    print("Average solution time {:}".format(np.average(t_sol)))
    print("Standard deviation of solution time {:}".format(np.std(np.array(t_sol))))
    print("Average pos error {:}".format(np.average(np.array(e_pos))))
    print("Average rot error {:}".format(np.average(np.array(e_rot))))
    print("Number of fails {:}".format(fails))
    if fails:
        raise SystemExit(f"{fails}/{n_tests} IK solves exceeded tolerance")


if __name__ == "__main__":
    np.random.seed(21)
    random_problem_2d_chain()
