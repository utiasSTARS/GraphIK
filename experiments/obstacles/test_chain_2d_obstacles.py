#!/usr/bin/env python3
import numpy as np
from numpy.testing import assert_array_less
import networkx as nx
import time
import matplotlib.pyplot as plt
from graphik.graphs.graph_base import RobotPlanarGraph
from graphik.robots.robot_base import RobotPlanar

from graphik.solvers.riemannian_solver import RiemannianSolver

from graphik.utils.dgp import (
    dist_to_gram,
    adjacency_matrix_from_graph,
    pos_from_graph,
    graph_from_pos,
    bound_smoothing,
)
from graphik.utils.utils import best_fit_transform, list_to_variable_dict


def plot_obstacles(obstacles: list):
    ax = plt.gca()
    ax.cla()
    # fig, ax = plt.plot()
    for obs in obstacles:
        circle = plt.Circle((obs[0][0], obs[0][1]), obs[1], color="r")
        ax.add_artist(circle)
    ax.set_xlim((-10, 10))
    ax.set_ylim((-10, 10))
    return ax


def check_collison(P: np.ndarray, obstacles: list):

    for obs in obstacles:
        center = obs[0]
        radius = obs[1]
        if any(np.diag((P - center) @ np.identity(2) @ (P - center).T) <= radius ** 2):
            return True
    return False


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
        "a": a,
        "theta": th,
        "joint_limits_upper": lim_u,
        "joint_limits_lower": lim_l,
    }

    robot = RobotPlanar(params)
    graph = RobotPlanarGraph(robot)
    obstacles = [
        (np.array([0, 2]), 0.99),
        (np.array([2, 0]), 0.99),
        (np.array([-2, 0]), 0.99),
    ]
    for idx, obs in enumerate(obstacles):
        graph.add_spherical_obstacle(f"o{idx}", obs[0], obs[1])

    ax = plot_obstacles(obstacles)
    solver = RiemannianSolver(graph)
    n_tests = 100
    q_init = list_to_variable_dict(n * [0])
    G_init = graph.realization(q_init)
    X_init = pos_from_graph(G_init)

    ax = plt.gca()
    for idx in range(n_tests):

        q_goal = graph.robot.random_configuration()
        G_goal = graph.realization(q_goal)
        X_goal = pos_from_graph(G_goal)
        D_goal = graph.distance_matrix_from_joints(q_goal)
        T_goal = robot.get_pose(q_goal, f"p{n}")

        # goals = {p"f{n}": X_goal[-1, :]} # position goal
        # goals = {f"p{n-1}": X_goal[-2, :], f"p{n}": X_goal[-1, :]}  # pose goal
        goals = {f"p{n-1}": X_goal[-5, :], f"p{n}": X_goal[-4, :]}  # pose goal

        G = graph.complete_from_pos(goals)
        F = adjacency_matrix_from_graph(G)

        # lb, ub = bound_smoothing(G)  # get lower and upper distance bounds for init
        # sol_info = solver.solve(D_goal, F, use_limits=False, bounds=(lb, ub))
        sol_info = solver.solve(D_goal, F, use_limits=True, Y_init=X_init)
        Y = sol_info["x"]
        t_sol += [sol_info["time"]]

        R, t = best_fit_transform(Y[[0, 1, 2], :], X_goal[[0, 1, 2], :])
        P_e = (R @ Y.T + t.reshape(2, 1)).T
        # ax.scatter(P_e[:, 0], P_e[:, 1])
        G_e = graph_from_pos(P_e, graph.node_ids)

        q_sol = robot.joint_variables(G_e)
        P_sol = pos_from_graph(graph.realization(q_sol), list(robot.structure))
        # ax.scatter(P_sol[:, 0], P_sol[:, 1])
        # ax.scatter(X_goal[:, 0], X_goal[:, 1])

        T_riemannian = robot.get_pose(list_to_variable_dict(q_sol), "p" + f"{robot.n}")
        err_riemannian = T_goal.dot(T_riemannian.inv()).log()
        err_riemannian_pos = np.linalg.norm(T_goal.trans - T_riemannian.trans)
        err_riemannian_rot = np.linalg.norm(err_riemannian[2:])
        e_rot += [err_riemannian_rot]
        e_pos += [err_riemannian_pos]

        infeas = False
        for idx, obs in enumerate(obstacles):
            if (X_goal[-4, :] - obs[0]).T @ np.identity(2) @ (
                X_goal[-4, :] - obs[0]
            ) <= obs[1] ** 2 or (X_goal[-5, :] - obs[0]).T @ np.identity(2) @ (
                X_goal[-5, :] - obs[0]
            ) <= obs[
                1
            ] ** 2:
                infeas = True

        col = False
        if check_collison(P_e[:-3], obstacles):
            col = True

        not_reach = False
        if err_riemannian_pos > 0.01 or err_riemannian_rot > 0.01:
            not_reach = True

        if infeas or col or not_reach:
            print(
                col * "collision"
                + infeas * "+ infeasible goal"
                + not_reach * "+ didn't reach"
            )
            fails += 1
        else:
            ax.plot(P_sol[:-3, 0], P_sol[:-3, 1], "-o")
            plt.pause(0.1)
        print(f"{idx}", end="\r")

    t_sol = np.array(t_sol)
    t_sol = t_sol[abs(t_sol - np.mean(t_sol)) < 2 * np.std(t_sol)]
    print("Average solution time {:}".format(np.average(t_sol)))
    print("Standard deviation of solution time {:}".format(np.std(np.array(t_sol))))
    print("Average pos error {:}".format(np.average(np.array(e_pos))))
    print("Average rot error {:}".format(np.average(np.array(e_rot))))
    print("Number of fails {:}".format(fails))
    # print("Standard deviation of maximum error {:}".format(np.std(np.array(e_sol))))
    plt.show()
    assert_array_less(e_pos, 1e-4 * np.ones(n_tests))


if __name__ == "__main__":
    np.random.seed(23)
    random_problem_2d_chain()
