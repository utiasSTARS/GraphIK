#!/usr/bin/env python3
import numpy as np
from experiments.problem_generation import generate_revolute_problem
from graphik.graphs import ProblemGraphRevolute
from graphik.solvers.riemannian_solver import RiemannianSolver
from graphik.utils import *
from graphik.utils.roboturdf import load_ur10
from numpy import pi
from numpy.linalg import norm


def solve_random_problem(graph: ProblemGraphRevolute, solver: RiemannianSolver):
    robot = graph.robot
    n = robot.n

    G, T_goal, D_goal, X_goal = generate_revolute_problem(graph, obstacles=True)
    omega = adjacency_matrix_from_graph(G)
    lb, ub = bound_smoothing(G)

    # Y_init = pos_from_graph(graph.realization(graph.robot.zero_configuration()))

    # sol_info = solver.solve(D_goal, omega, use_limits=False, Y_init=Y_init)
    sol_info = solver.solve(D_goal, omega, use_limits=True, bounds=(lb, ub),jit=True)
    Y = sol_info["x"]
    t_sol = sol_info["time"]

    G_sol = graph_from_pos(Y, graph.node_ids)
    q_sol = graph.joint_variables(G_sol, {f"p{n}": T_goal})

    T_riemannian = robot.pose(q_sol, "p" + str(n))
    err_riemannian_pos = norm(T_goal.trans - T_riemannian.trans)

    z_goal = T_goal.as_matrix()[:3, 2]
    z_riemannian = T_riemannian.as_matrix()[:3, 2]
    err_riemannian_rot = abs(safe_arccos(z_riemannian.dot(z_goal)))

    broken_limits = graph.check_distance_limits(graph.realization(q_sol), tol=1e-6)

    fail = False
    if err_riemannian_pos > 0.01 or err_riemannian_rot > 0.01:
        fail = True

    if len(broken_limits) > 0:
        print(broken_limits)
        fail = True

    print(
        f"Pos. error: {err_riemannian_pos}\nRot. error: {err_riemannian_rot}\nCost: {sol_info['f(x)']}\nSolution time: {t_sol}."
    )
    print("------------------------------------")
    return err_riemannian_pos, err_riemannian_rot, t_sol, fail, sol_info


def main():
    np.random.seed(21)
    robot, graph = load_ur10()

    phi = (1 + np.sqrt(5)) / 2
    scale = 0.5
    radius = 0.5
    obstacles = [
        # (scale * np.asarray([0, 1, phi]), radius),
        # (scale * np.asarray([0, 1, -phi]), radius),
        # (scale * np.asarray([0, -1, -phi]), radius),
        # (scale * np.asarray([0, -1, phi]), radius),
        (scale * np.asarray([1, phi, 0]), radius),
        (scale * np.asarray([1, -phi, 0]), radius),
        (scale * np.asarray([-1, -phi, 0]), radius),
        (scale * np.asarray([-1, phi, 0]), radius),
        # (scale * np.asarray([phi, 0, 1]), radius),
        # (scale * np.asarray([-phi, 0, 1]), radius),
        # (scale * np.asarray([-phi, 0, -1]), radius),
        # (scale * np.asarray([phi, 0, -1]), radius),
    ]
    for idx, obs in enumerate(obstacles):
        graph.add_spherical_obstacle(f"o{idx}", obs[0], obs[1])

    params = {
        "solver": "TrustRegions",
        "mingradnorm": 1e-10,
        "maxiter": 3000,
        "theta": 1,
        "logverbosity": 0,
    }
    solver = RiemannianSolver(graph, params)
    num_tests = 100
    e_pos = []
    e_rot = []
    t = []
    fails = []
    nit = []
    for _ in range(num_tests):
        e_r_pos, e_r_rot, t_sol, fail, sol_info = solve_random_problem(graph, solver)
        e_pos += [e_r_pos]
        e_rot += [e_r_rot]
        t += [t_sol]
        fails += [fail]
        nit += [sol_info["iterations"]]

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
