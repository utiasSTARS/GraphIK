#!/usr/bin/env python3
import graphik
import numpy as np
from experiments.problem_generation import generate_revolute_problem
from graphik.graphs import RobotRevoluteGraph
from graphik.solvers.riemannian_solver import RiemannianSolver
from graphik.utils import *
from graphik.utils.roboturdf import load_ur10
from numpy import pi
from numpy.linalg import norm


def solve_random_problem(graph: RobotRevoluteGraph, solver: RiemannianSolver):

    G, T_goal, D_goal, X_goal = generate_revolute_problem(graph, obstacles=True)
    omega = adjacency_matrix_from_graph(G)
    lb, ub = bound_smoothing(G)

    # sol_info = solver.solve(D_goal, omega, use_limits=False, Y_init=X_init)
    sol_info = solver.solve(D_goal, omega, use_limits=True, bounds=(lb, ub))
    Y = sol_info["x"]
    t_sol = sol_info["time"]

    G_sol = graph_from_pos(Y, graph.node_ids)
    q_sol = robot.joint_variables(G_sol, {f"p{n}": T_goal})

    T_riemannian = robot.get_pose(q_sol, "p" + str(n))
    err_riemannian_pos = norm(T_goal.trans - T_riemannian.trans)

    z_goal = T_goal.as_matrix()[:3, 2]
    z_riemannian = T_riemannian.as_matrix()[:3, 2]
    err_riemannian_rot = abs(safe_arccos(z_riemannian.dot(z_goal)))

    fail = False
    if err_riemannian_pos > 0.01 or err_riemannian_rot > 0.01:
        fail = True

    # broken_limits = graph.check_distance_limits(graph.realization(q_sol))
    broken_limits = graph.check_distance_limits(G_sol)
    print(broken_limits)
    print(
        f"Pos. error: {err_riemannian_pos}\nRot. error: {err_riemannian_rot}\nCost: {sol_info['f(x)']}\nSolution time: {t_sol}."
    )
    print("------------------------------------")
    return err_riemannian_pos, err_riemannian_rot, t_sol, fail, sol_info


if __name__ == "__main__":

    np.random.seed(21)
    n = 6
    robot, graph = load_ur10()

    phi = (1 + np.sqrt(5)) / 2
    scale = 0.5
    radius = 0.5
    obstacles = [
        (scale * np.asarray([0, 1, phi]), radius),
        (scale * np.asarray([0, 1, -phi]), radius),
        (scale * np.asarray([0, -1, -phi]), radius),
        (scale * np.asarray([0, -1, phi]), radius),
        (scale * np.asarray([1, phi, 0]), radius),
        (scale * np.asarray([1, -phi, 0]), radius),
        (scale * np.asarray([-1, -phi, 0]), radius),
        (scale * np.asarray([-1, phi, 0]), radius),
        (scale * np.asarray([phi, 0, 1]), radius),
        (scale * np.asarray([-phi, 0, 1]), radius),
        (scale * np.asarray([-phi, 0, -1]), radius),
        (scale * np.asarray([phi, 0, -1]), radius),
    ]
    for idx, obs in enumerate(obstacles):
        graph.add_spherical_obstacle(f"o{idx}", obs[0], obs[1])
    # print(graph.directed.nodes(data="type"))
    # graph.distance_bounds_from_sampling()
    solver = RiemannianSolver(graph)
    num_tests = 20
    e_pos = []
    e_rot = []
    t = []
    fails = []
    for _ in range(num_tests):
        e_r_pos, e_r_rot, t_sol, fail, _ = solve_random_problem(graph, solver)
        e_pos += [e_r_pos]
        e_rot += [e_r_rot]
        t += [t_sol]
        fails += [fail]

    t = np.array(t)
    t = t[abs(t - np.mean(t)) < 2 * np.std(t)]
    print("Average solution time {:}".format(np.average(t)))
    print("Standard deviation of solution time {:}".format(np.std(np.array(t))))
    print("Average pos error {:}".format(np.average(np.array(e_pos))))
    print("Average rot error {:}".format(np.average(np.array(e_rot))))
    print("Number of fails {:}".format(sum(fails)))
