#!/usr/bin/env python3
import numpy as np
from numpy.testing import assert_array_less
import networkx as nx
import time
from graphik.graphs.graph_base import SphericalRobotGraph
from graphik.robots.robot_base import RobotPlanar
from graphik.solvers.riemannian_solver import RiemannianSolver
from graphik.utils.dgp import (
    dist_to_gram,
    linear_projection,
    adjacency_matrix_from_graph,
)
from graphik.utils.utils import best_fit_transform, list_to_variable_dict


def random_problem_2d_chain():
    e_rot = []
    e_pos = []
    t_sol = []
    n = 10
    fails = 0

    a = list_to_variable_dict(np.random.rand(n) * 2.0 + 1.0)
    th = list_to_variable_dict(np.zeros(n))
    angular_limits = np.minimum(np.random.rand(n) * np.pi + 0.20, np.pi)
    upper_angular_limits = list_to_variable_dict(angular_limits)
    lower_angular_limits = list_to_variable_dict(-angular_limits)
    params = {
        "a": a,
        "theta": th,
        "joint_limits_upper": upper_angular_limits,
        "joint_limits_lower": lower_angular_limits,
    }

    robot = RobotPlanar(params)
    graph = SphericalRobotGraph(robot)
    solver = RiemannianSolver(graph)
    n_tests = 100

    q_init = list_to_variable_dict(n * [0])
    G_init = graph.realization(q_init)
    X_init = pos_from_graph(G_init)
    for idx in range(n_tests):

        q_goal = graph.robot.random_configuration()
        G_goal = graph.realization(q_goal)
        X_goal = pos_from_graph(G_goal)
        D_goal = graph.distance_matrix_from_joints(q_goal)
        T_goal = robot.get_pose(q_goal, f"p{n}")

        # goals = {f"p{n}": X_goal[-1, :]}
        goals = {f"p{n-1}": X_goal[-2, :], f"p{n}": X_goal[-1, :]}

        align_ind = list(np.arange(graph.dim + 1))
        for name in goals.keys():
            align_ind.append(graph.node_ids.index(name))

        G = graph.complete_from_pos(goals)
        lb, ub = graph.distance_bounds(G)  # will take goals and jli
        F = adjacency_matrix_from_graph(G)

        sol_info = solver.solve(
            D_goal, F, bounds=(lb, ub), max_attempts=10, use_limits=True
        )
        # sol_info = solver.solve(D_goal, F, Y_init=X_init, max_attempts=10, use_limits = True)
        Y = sol_info["x"]
        t_sol += [sol_info["time"]]

        R, t = best_fit_transform(Y[align_ind, :], X_goal[align_ind, :])
        P_e = (R @ Y.T + t.reshape(2, 1)).T
        # X_e = P_e @ P_e.T
        G_e = graph.graph_from_pos(P_e)

        q_sol = robot.joint_variables(G_e)

        T_riemannian = robot.get_pose(list_to_variable_dict(q_sol), "p" + f"{robot.n}")
        err_riemannian = (T_goal.dot(T_riemannian.inv())).log()
        err_riemannian_pos = np.linalg.norm(T_goal.trans - T_riemannian.trans)
        err_riemannian_rot = np.linalg.norm(err_riemannian[2])
        # print(np.arctan2(T_riemannian.as_matrix()[1,0], T_riemannian.as_matrix()[0,0]))
        # print(np.arctan2(T_goal.as_matrix()[1,0], T_goal.as_matrix()[0,0]))
        # print("----")

        e_rot += [err_riemannian_rot]
        e_pos += [err_riemannian_pos]
        q_abs = np.abs(np.array(list(q_sol.values())))
        if err_riemannian_pos > 0.01 or err_riemannian_rot > 0.01:
            fails += 1
        elif sum(q_abs > (angular_limits + 0.01 * angular_limits)) > 0:
            print("FAIL")
            fails += 1
        print(f"{idx}", end="\r")

    t_sol = np.array(t_sol)
    t_sol = t_sol[abs(t_sol - np.mean(t_sol)) < 2 * np.std(t_sol)]
    print("Average solution time {:}".format(np.average(t_sol)))
    print("Standard deviation of solution time {:}".format(np.std(np.array(t_sol))))
    print("Average pos error {:}".format(np.average(np.array(e_pos))))
    print("Average rot error {:}".format(np.average(np.array(e_rot))))
    print("Number of fails {:}".format(fails))
    # print(robot.get_pose(q_sol, f"p{n}").trans - T_goal.trans)
    # e_sol += [np.linalg.norm(robot.get_pose(q_sol, f"p{n}").trans - T_goal.trans)]
    # q_abs = np.abs(np.array(list(q_sol.values())))
    # if assert_array_less(q_abs, angular_limits + 0.01 * angular_limits):
    #     print("LIMIT VIOLATION")

    # print("Average solution time {:}".format(np.average(np.array(t_sol))))
    # print("Standard deviation of solution time {:}".format(np.std(np.array(t_sol))))
    # print("Average maximum error {:}".format(np.average(np.array(e_sol))))
    # print("Standard deviation of maximum error {:}".format(np.std(np.array(e_sol))))
    # assert_array_less(e_sol, 1e-3 * np.ones(n_tests))


if __name__ == "__main__":
    np.random.seed(22)
    random_problem_2d_chain()
