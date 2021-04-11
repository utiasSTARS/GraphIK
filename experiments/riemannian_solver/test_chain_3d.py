#!/usr/bin/env python3
import numpy as np
from graphik.graphs import RobotSphericalGraph
from graphik.robots import RobotSpherical

from graphik.solvers.riemannian_solver import RiemannianSolver
from graphik.utils.utils import best_fit_transform, list_to_variable_dict
from graphik.utils.dgp import (
    adjacency_matrix_from_graph,
    pos_from_graph,
    graph_from_pos,
)


def random_problem_3d_chain():
    e_rot = []
    e_pos = []
    t_sol = []
    n = 100
    fails = 0

    a = list_to_variable_dict(0 * np.random.rand(n))
    d = list_to_variable_dict(np.random.rand(n))
    al = list_to_variable_dict(0 * np.random.rand(n))
    th = list_to_variable_dict(0 * np.random.rand(n))
    lim_u = list_to_variable_dict(np.pi * np.ones(n))
    lim_l = list_to_variable_dict(-np.pi * np.ones(n))

    params = {
        "a": a,
        "alpha": al,
        "d": d,
        "theta": th,
        "joint_limits_lower": lim_l,
        "joint_limits_upper": lim_u,
    }
    robot = RobotSpherical(params)  # instantiate robot
    graph = RobotSphericalGraph(robot)  # instantiate graph
    solver = RiemannianSolver(graph)
    n_tests = 1000

    q_init = graph.robot.random_configuration()
    for key in q_init.keys():
        q_init[key] = [0, 0]
    G_init = graph.realization(q_init)
    X_init = pos_from_graph(G_init)
    for idx in range(n_tests):

        q_goal = graph.robot.random_configuration()

        G_goal = graph.realization(q_goal)
        X_goal = pos_from_graph(G_goal)
        D_goal = graph.distance_matrix_from_joints(q_goal)
        T_goal = robot.get_pose(q_goal, f"p{n}")

        goals = {f"p{n-1}": X_goal[-2, :], f"p{n}": X_goal[-1, :]}
        G = graph.complete_from_pos(goals)
        # lb, ub = bound_smoothing(G)  # will take goals and jli
        F = adjacency_matrix_from_graph(G)

        # sol_info = solver.solve(D_goal, F, use_limits=False, bounds=(lb, ub))
        sol_info = solver.solve(
            D_goal, F, Y_init=X_init, max_attempts=10, use_limits=False
        )
        Y = sol_info["x"]
        t_sol += [sol_info["time"]]
        R, t = best_fit_transform(Y[[0, 1, 2, 3], :], X_goal[[0, 1, 2, 3], :])
        P_e = (R @ Y.T + t.reshape(3, 1)).T
        # X_e = P_e @ P_e.T
        G_e = graph_from_pos(P_e, graph.node_ids)

        q_sol = robot.joint_variables(G_e)

        T_riemannian = robot.get_pose(list_to_variable_dict(q_sol), "p" + f"{robot.n}")
        T_riemannian.rot
        # err_riemannian = (T_goal.dot(T_riemannian.inv())).log()
        err_riemannian_pos = np.linalg.norm(T_goal.trans - T_riemannian.trans)
        err_riemannian_rot = np.linalg.norm(
            T_riemannian.as_matrix()[:3, 2] - T_goal.as_matrix()[:3, 2]
        )
        # print(T_riemannian.as_matrix())
        # print(T_goal.as_matrix())
        # print(err_riemannian)
        # print("------------")

        e_rot += [err_riemannian_rot]
        e_pos += [err_riemannian_pos]
        # q_abs = np.abs(np.array(list(q_sol.values())))
        if err_riemannian_pos > 0.01 or err_riemannian_rot > 0.01:
            fails += 1
        print(f"{idx}", end="\r")

    t_sol = np.array(t_sol)
    t_sol = t_sol[abs(t_sol - np.mean(t_sol)) < 2 * np.std(t_sol)]
    print("Average solution time {:}".format(np.average(t_sol)))
    print("Standard deviation of solution time {:}".format(np.std(np.array(t_sol))))
    print("Average pos error {:}".format(np.average(np.array(e_pos))))
    print("Average rot error {:}".format(np.average(np.array(e_rot))))
    print("Number of fails {:}".format(fails))

    # assert_array_less(e_sol, 1e-3 * np.ones(n_tests))


if __name__ == "__main__":
    np.random.seed(21)
    random_problem_3d_chain()
