#!/usr/bin/env python3
import time

import numpy as np

from matplotlib import pyplot as plt
from numpy import pi, sqrt

from graphik.graphs.graph_base import RobotGraph, RobotRevoluteGraph
from graphik.robots.robot_base import RobotRevolute
from graphik.solvers.riemannian_solver import RiemannianSolver
from graphik.utils.geometry import trans_axis
from graphik.utils.dgp import (
    pos_from_graph,
    adjacency_matrix_from_graph,
    graph_from_pos,
    bound_smoothing,
)
from graphik.utils.utils import best_fit_transform, list_to_variable_dict, dZ


def solve_random_problem(graph: RobotGraph, solver: RiemannianSolver):
    q_goal = graph.robot.random_configuration()
    G_goal = graph.realization(q_goal)
    X_goal = pos_from_graph(G_goal)
    D_goal = graph.distance_matrix_from_joints(q_goal)

    goals = {}
    T_goal = {}
    for _, ee_pair in enumerate(robot.end_effectors):
        T_goal[ee_pair[0]] = robot.get_pose(q_goal, ee_pair[0])
        goals[ee_pair[0]] = T_goal[ee_pair[0]].trans
        goals[ee_pair[1]] = T_goal[ee_pair[0]].dot(trans_axis(dZ, "z")).trans

    q_rand = list_to_variable_dict(graph.robot.n * [0])
    G_rand = graph.realization(q_rand)
    X_rand = pos_from_graph(G_rand)
    X_init = X_rand

    G = graph.complete_from_pos(goals)
    lb, ub = bound_smoothing(G)
    F = adjacency_matrix_from_graph(G)

    # sol_info = solver.solve(D_goal, F, use_limits=False, Y_init=X_init)
    sol_info = solver.solve(D_goal, F, use_limits=False, bounds=(lb, ub))
    # print(sol_info["f(x)"])
    Y = sol_info["x"]
    t_sol = sol_info["time"]

    R, t = best_fit_transform(Y[[0, 1, 2, 3], :], X_goal[[0, 1, 2, 3], :])
    P_e = (R @ Y.T + t.reshape(3, 1)).T
    X_e = P_e @ P_e.T

    G_sol = graph_from_pos(P_e, graph.node_ids)
    # q_sol = robot.joint_variables(G_sol, T_goal.as_matrix())
    q_sol = robot.joint_variables(G_sol, T_goal)
    q_sol = dict(sorted(q_sol.items()))

    # q_sol_old = robot.joint_variables_old(G_sol, T_goal)
    # q_sol_old = dict(sorted(q_sol_old.items()))
    # print(q_sol_old)
    # print(
    #     robot.get_pose(list_to_variable_dict(q_sol), "p" + str(n)).trans
    #     - T_goal.trans)
    # print(q_sol)

    # D_sol = np.diag(X_e)[np.newaxis, :] + np.diag(X_e)[:, np.newaxis] - 2 * X_e
    # e_D = F * (sqrt(D_sol) - sqrt(D_goal))
    # e_sol = abs(max(e_D.min(), e_D.max(), key=abs))
    e_sol = 0
    for key, value in T_goal.items():
        e_pos = robot.get_pose(q_sol, key).trans - value.trans
        e_sol += e_pos.T @ e_pos
        e_pos = (
            robot.get_pose(q_sol, key).dot(trans_axis(dZ, "z")).trans
            - value.dot(trans_axis(dZ, "z")).trans
        )
        e_sol += e_pos.T @ e_pos
    print(f"Final error: {e_sol}. Solution time: {t_sol}.")
    return e_sol, t_sol


if __name__ == "__main__":

    np.random.seed(21)

    n = 5
    parents = {"p0": ["p1"], "p1": ["p2", "p3"], "p2": ["p4"], "p3": ["p5"]}
    a = {"p1": 0, "p2": -0.612, "p3": -0.612, "p4": -0.5732, "p5": -0.5732}
    d = {"p1": 0.1237, "p2": 0, "p3": 0, "p4": 0, "p5": 0}
    al = {"p1": pi / 2, "p2": 0, "p3": 0, "p4": 0, "p5": 0}
    th = {"p1": 0, "p2": 0, "p3": 0, "p4": 0, "p5": 0}
    ub = list_to_variable_dict((pi) * np.ones(n))
    lb = list_to_variable_dict(-(pi) * np.ones(n))

    # KUKA LWR-IW
    # n = 7
    # a = [0, 0, 0, 0, 0, 0, 0]
    # d = [0, 0, 0.4, 0, 0.39, 0, 0]
    # al = [pi / 2, -pi / 2, -pi / 2, pi / 2, pi / 2, -pi / 2, 0]
    # th = [0, 0, 0, 0, 0, 0, 0]
    # ub = (pi) * np.ones(n)
    # lb = -ub

    # Jaco Arm
    # n = 6
    # D1 = 0.2755
    # D2 = 0.2050
    # D3 = 0.2050
    # D4 = 0.2073
    # D5 = 0.1038
    # D6 = 0.1038
    # D7 = 0.1600
    # e2 = 0.0098
    # a = [0, D2, 0, 0, 0, 0]
    # d = [D1, 0, -e2, -(D3 + D4), 0, -(D5 + D6)]
    # al = [pi / 2, pi, pi / 2, pi / 2, pi / 2, 0]
    # th = [0, 0, 0, 0, 0, 0]
    # ub = pi * np.ones(n)
    # lb = -ub

    params = {
        "a": a,
        "alpha": al,
        "d": d,
        "theta": th,
        "modified_dh": False,
        "lb": lb,
        "ub": ub,
        "parents": parents,
    }
    robot = RobotRevolute(params)
    graph = RobotRevoluteGraph(robot)
    # print(graph.adjacency_matrix())
    solver = RiemannianSolver(graph)
    num_tests = 1000
    t = []
    e = []
    for idx in range(num_tests):
        e_sol, t_sol = solve_random_problem(graph, solver)
        t += [t_sol]
        e += [e_sol]

    print("Average solution time {:}".format(np.average(np.array(t))))
    print("Standard deviation of solution time {:}".format(np.std(np.array(t))))
    print("Average maximum error {:}".format(np.average(np.array(e))))
    print("Standard deviation of maximum error {:}".format(np.std(np.array(e))))
    print("Number of failed instances {:}".format(sum(np.array(e) > 0.01)))

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(P_goal[:, 0], P_goal[:, 1], P_goal[:, 2], marker="o")
    # ax.scatter(P_e[:, 0], P_e[:, 1], P_e[:, 2], marker="^", s=40)
    # # ax.scatter(P_sol[:,0],P_sol[:,1],P_sol[:,2],marker = '*', s=80)
    # plt.show()
    # #
    # F = graph.known_distances()
    # # print(F)
    # print("Goal end-effector transform: \n {:}".format(T_goal.as_matrix()))
    # print("---------------------------------------")
    # print("Solution end-effector transform: \n {:}".format(T_sol.as_matrix()))
    # print("---------------------------------------")

    # print(
    #     "Difference in known distance matrix elements (Y): \n {:}".format(
    #         F * (sqrt(D) - sqrt(D_e))
    #     )
    # )
