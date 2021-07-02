#!/usr/bin/env python3
import numpy as np
from experiments.problem_generation import generate_revolute_problem
from graphik.graphs import RobotRevoluteGraph

from graphik.solvers.euclidean_solver_un import EuclideanSolver
# from graphik.solvers.euclidean_solver import EuclideanSolver
from graphik.utils import *
from graphik.utils.roboturdf import RobotURDF, load_ur10
from numpy import pi
from numpy.linalg import norm


def solve_random_problem(graph: RobotRevoluteGraph, solver: EuclideanSolver):
    robot = graph.robot
    n = robot.n

    G, T_goal, D_goal, X_goal = generate_revolute_problem(graph, obstacles=False)
    Y_init = pos_from_graph(graph.realization(graph.robot.zero_configuration()))

    Y, t_sol, num_iter = solver.solve(D_goal, Y_init=Y_init)

    G_sol = graph_from_pos(Y, graph.node_ids)

    T_g = {f"p{n}": T_goal}
    q_sol = robot.joint_variables(G_sol, T_g)

    T_euclidean = robot.get_pose(q_sol, "p" + str(n))
    err_euclidean_pos = norm(T_goal.trans - T_euclidean.trans)

    z_goal = T_goal.as_matrix()[:3, 2]
    z_euclidean = T_euclidean.as_matrix()[:3, 2]
    err_euclidean_rot = abs(safe_arccos(z_euclidean.dot(z_goal)))

    broken_limits = graph.check_distance_limits(graph.realization(q_sol), tol=1e-6)

    fail = False
    if err_euclidean_pos > 0.01 or err_euclidean_rot > 0.01:
        fail = True

    if len(broken_limits) > 0:
        print(broken_limits)
        fail = True

    print(
        f"Pos. error: {err_euclidean_pos}\nRot. error: {err_euclidean_rot}\nSolution time: {t_sol}\nNum iter: {num_iter}."
    )
    print("------------------------------------")
    return err_euclidean_pos, err_euclidean_rot, t_sol, num_iter, fail


def main():
    np.random.seed(21)
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

    # graph.distance_bounds_from_sampling()
    # params = {"method": "trust-ncg", "options": {"gtol": 1e-10, "maxiter": 3000}}
    params = {"method": "trust-krylov", "options": {"gtol": 1e-10, "maxiter": 3000}}
    solver = EuclideanSolver(graph, params)
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
    print("Median solution time {:}".format(np.median(t)))
    print("Standard deviation of solution time {:}".format(np.std(np.array(t))))
    print("Average iterations {:}".format(np.average(nit)))
    print("Median iterations {:}".format(np.median(nit)))
    print("Average pos error {:}".format(np.average(np.array(e_pos))))
    print("Average rot error {:}".format(np.average(np.array(e_rot))))
    print("Number of fails {:}".format(sum(fails)))


if __name__ == "__main__":
    main()