#!/usr/bin/env python3
import numpy as np
from numpy.testing import assert_array_less
import networkx as nx
import time
from graphik.graphs.graph_base import RobotPlanarGraph

from graphik.robots.robot_base import RobotPlanar

from graphik.solvers.solver_rfr import RiemannianSolver

from graphik.utils.dgp import (
    dist_to_gram,
    adjacency_matrix_from_graph,
    pos_from_graph,
    graph_from_pos,
    bound_smoothing,
)
from graphik.utils.utils import best_fit_transform, list_to_variable_dict


def random_problem_2d_chain():
    e_rot = []
    e_pos = []
    t_sol = []
    fails = 0
    n = 10

    a = list_to_variable_dict(np.ones(n))
    th = list_to_variable_dict(np.zeros(n))
    lim_u = list_to_variable_dict(np.pi * np.ones(n))
    lim_l = list_to_variable_dict(-np.pi * np.ones(n))
    robot_params = {
        "a": a,
        "theta": th,
        "joint_limits_upper": lim_u,
        "joint_limits_lower": lim_l,
    }

    robot = RobotPlanar(robot_params)
    graph = RobotPlanarGraph(robot)

    solver_params = {
        "solver": "TrustRegions",
        "mingradnorm": 1e-6,
        "maxiter": 1000,
        "logverbosity": 2,
    }

    solver = RiemannianSolver(solver_params)
    n_tests = 100

    q_init = list_to_variable_dict(n * [0])
    G_init = graph.realization(q_init)
    X_init = pos_from_graph(G_init)
    for idx in range(n_tests):

        q_goal = graph.robot.random_configuration()
        G_goal = graph.realization(q_goal)
        X_goal = pos_from_graph(G_goal)
        T_goal = robot.get_pose(q_goal, f"p{n}")

        goals = {f"p{n-1}": X_goal[-2, :], f"p{n}": X_goal[-1, :]}  # pose goal
        problem_params = {"goals": goals, "joint_limits": False, "init": X_init}
        # lb, ub = bound_smoothing(G)  # get lower and upper distance bounds for init
        sol_info = solver.solve(graph, problem_params)

        Y = sol_info["x"]
        t_sol += [sol_info["time"]]

        R, t = best_fit_transform(Y[[0, 1, 2], :], X_goal[[0, 1, 2], :])
        P_e = (R @ Y.T + t.reshape(2, 1)).T
        G_e = graph_from_pos(P_e, graph.node_ids)

        q_sol = robot.joint_variables(G_e)

        T_riemannian = robot.get_pose(list_to_variable_dict(q_sol), "p" + f"{robot.n}")
        err_riemannian = T_goal.dot(T_riemannian.inv()).log()
        err_riemannian_pos = np.linalg.norm(T_goal.trans - T_riemannian.trans)
        err_riemannian_rot = np.linalg.norm(err_riemannian[2:])
        e_rot += [err_riemannian_rot]
        e_pos += [err_riemannian_pos]
        if err_riemannian_pos > 0.01 or err_riemannian_rot > 0.01:
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
    # print("Standard deviation of maximum error {:}".format(np.std(np.array(e_sol))))

    assert_array_less(e_pos, 1e-4 * np.ones(n_tests))


if __name__ == "__main__":
    np.random.seed(21)
    random_problem_2d_chain()
