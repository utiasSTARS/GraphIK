#!/usr/bin/env python3
import graphik
import time
import numpy as np
from graphik.graphs import RobotRevoluteGraph
from graphik.solvers.euclidean_solver import EuclideanSolver
from graphik.utils.dgp import (
    graph_from_pos,
    pos_from_graph,
)
from graphik.utils.roboturdf import RobotURDF, load_ur10
from graphik.utils.utils import best_fit_transform, list_to_variable_dict, safe_arccos
from numpy import pi
from numpy.linalg import norm


def solve_random_problem(graph: RobotRevoluteGraph, solver: EuclideanSolver):
    n = graph.robot.n
    q_rand = list_to_variable_dict(graph.robot.n * [0])
    G_rand = graph.realization(q_rand)
    Y_init = pos_from_graph(G_rand)

    feasible = False
    while not feasible:
        q_goal = robot.random_configuration()
        G_goal = graph.realization(q_goal)
        X_goal = pos_from_graph(G_goal)
        D_goal = graph.distance_matrix_from_joints(q_goal)
        T_goal = robot.get_pose(q_goal, f"p{n}")
        broken_limits = graph.check_distance_limits(G_goal)
        if len(broken_limits) == 0:
            feasible = True

    Y, t_sol, num_iter = solver.solve(D_goal, Y_init=Y_init)

    align_ind = list(np.arange(graph.dim + 1))
    for u, v in robot.end_effectors:
        align_ind.append(graph.node_ids.index(u))
        align_ind.append(graph.node_ids.index(v))

    R, t = best_fit_transform(Y[align_ind, :], X_goal[align_ind, :])
    P_e = (R @ Y.T + t.reshape(3, 1)).T

    G_sol = graph_from_pos(P_e, graph.node_ids)

    T_g = {f"p{n}": T_goal}
    q_sol = robot.joint_variables(G_sol, T_g)

    T_euclidean = robot.get_pose(list_to_variable_dict(q_sol), "p" + str(n))
    err_euclidean_pos = norm(T_goal.trans - T_euclidean.trans)

    z_goal = T_goal.as_matrix()[:3, 2]
    z_euclidean = T_euclidean.as_matrix()[:3, 2]
    err_euclidean_rot = abs(safe_arccos(z_euclidean.dot(z_goal)))

    fail = False
    if err_euclidean_pos > 0.01 or err_euclidean_rot > 0.01:
        fail = True

    print(
        f"Pos. error: {err_euclidean_pos}\nRot. error: {err_euclidean_rot}\nSolution time: {t_sol}\nNum iter: {num_iter}."
    )
    print("------------------------------------")
    return err_euclidean_pos, err_euclidean_rot, t_sol, num_iter, fail


if __name__ == "__main__":

    np.random.seed(21)

    robot, graph = load_ur10()

    scale = 0.5
    radius = 0.45
    obstacles = [
        (scale * np.asarray([-1, 1, 1]), radius),
        (scale * np.asarray([-1, 1, -1]), radius),
        (scale * np.asarray([1, -1, 1]), radius),
        (scale * np.asarray([1, -1, -1]), radius),
        (scale * np.asarray([1, 1, 1]), radius),
        (scale * np.asarray([1, 1, -1]), radius),
        (scale * np.asarray([-1, -1, 1]), radius),
        (scale * np.asarray([-1, -1, -1]), radius),
    ]
    for idx, obs in enumerate(obstacles):
        graph.add_spherical_obstacle(f"o{idx}", obs[0], obs[1])

    # graph.distance_bounds_from_sampling()
    solver = EuclideanSolver(graph, {"method":"trust-krylov", "options": {"gtol":10e-9}})
    num_tests = 100
    e_pos = []
    e_rot = []
    t = []
    fails = []
    nit = []
    for _ in range(num_tests):
        e_r_pos, e_r_rot, t_sol, num_iter, fail = solve_random_problem(graph, solver)
        e_pos += [e_r_pos]
        e_rot += [e_r_rot]
        t += [t_sol]
        fails += [fail]
        nit += [num_iter]

    t = np.array(t)
    t = t[abs(t - np.mean(t)) < 2 * np.std(t)]
    print("Average solution time {:}".format(np.average(t)))
    print("Standard deviation of solution time {:}".format(np.std(np.array(t))))
    print("Average iterations {:}".format(np.average(nit)))
    print("Average pos error {:}".format(np.average(np.array(e_pos))))
    print("Average rot error {:}".format(np.average(np.array(e_rot))))
    print("Number of fails {:}".format(sum(fails)))
