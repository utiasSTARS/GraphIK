#!/usr/bin/env python3
import numpy as np
from numpy.testing import assert_array_less
from graphik.graphs import ProblemGraphPlanar

from graphik.robots import RobotPlanar

from graphik.solvers.riemannian_solver import RiemannianSolver

from graphik.utils import *
# from graphik.utils.utils import best_fit_transform, list_to_variable_dict


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
    params = {
        "link_lengths": a,
        "theta": th,
        "ub": lim_u,
        "lb": lim_l,
        "num_joints": n
    }

    # robot = Revolute2dChain(params)
    robot = RobotPlanar(params)
    graph = ProblemGraphPlanar(robot)
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
        T_goal = robot.pose(q_goal, f"p{n}")

        # goals = {p"f{n}": X_goal[-1, :]} # position goal
        goals = {f"p{n-1}": X_goal[-2, :], f"p{n}": X_goal[-1, :]}  # pose goal

        G = graph.complete_from_pos(goals)
        F = adjacency_matrix_from_graph(G)

        # lb, ub = bound_smoothing(G)  # get lower and upper distance bounds for init
        # sol_info = solver.solve(D_goal, F, use_limits=False, bounds=(lb, ub))
        sol_info = solver.solve(D_goal, F, use_limits=False, Y_init=X_init)
        Y = sol_info["x"]
        t_sol += [sol_info["time"]]

        R, t = best_fit_transform(Y[[0, 1, 2], :], X_goal[[0, 1, 2], :])
        P_e = (R @ Y.T + t.reshape(2, 1)).T
        G_e = graph_from_pos(P_e, graph.node_ids)

        q_sol = graph.joint_variables(G_e)
        # G_sol = graph.realization(q_sol)
        # D_sol = graph.distance_matrix_from_joints(q_sol)

        T_riemannian = robot.pose(q_sol, "p" + f"{robot.n}")
        err_riemannian = T_goal.dot(T_riemannian.inv()).log()
        err_riemannian_pos = np.linalg.norm(T_goal.trans - T_riemannian.trans)
        err_riemannian_rot = np.linalg.norm(err_riemannian[2:])
        # print(robot.get_pose(q_sol, f"p{n}").trans - T_goal.trans)
        # print(robot.get_pose(q_sol, f"p{n}").trans - T_goal.trans)
        # e_D = F * (np.sqrt(D_sol) - np.sqrt(D_goal))
        # e_sol += [abs(max(e_D.min(), e_D.max(), key=abs))]
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
