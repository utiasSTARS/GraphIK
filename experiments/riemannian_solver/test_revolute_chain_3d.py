#!/usr/bin/env python3
from experiments.problem_generation import generate_revolute_problem
import numpy as np
from graphik.graphs import ProblemGraphRevolute
from graphik.solvers.riemannian_solver import RiemannianSolver
from graphik.utils import *
from graphik.utils.roboturdf import load_kuka, load_ur10
from numpy import pi
from numpy.linalg import norm


def solve_random_problem(graph: ProblemGraphRevolute, solver: RiemannianSolver):
    n = graph.robot.n

    G, T_goal, D_goal, X_goal = generate_revolute_problem(graph)

    lb, ub = bound_smoothing(G)
    omega = adjacency_matrix_from_graph(G)

    # Y_init = pos_from_graph(graph.realization(graph.robot.zero_configuration()))
    # sol_info = solver.solve(D_goal, omega, use_limits=False, Y_init=Y_init)
    sol_info = solver.solve(D_goal, omega, use_limits=False, bounds=(lb, ub))
    Y = sol_info["x"]
    t_sol = sol_info["time"]

    G = graph_from_pos(Y, graph.node_ids)  # not really order-dependent

    q_sol = graph.joint_variables(G, {f"p{n}": T_goal})

    T_riemannian = graph.get_pose(list_to_variable_dict(q_sol), "p" + str(n))
    err_riemannian_pos = norm(T_goal.trans - T_riemannian.trans)

    z_goal = T_goal.as_matrix()[:3, 2]
    z_riemannian = T_riemannian.as_matrix()[:3, 2]
    err_riemannian_rot = abs(safe_arccos(z_riemannian.dot(z_goal)))

    fail = False
    if err_riemannian_pos > 0.01 or err_riemannian_rot > 0.01:
        fail = True

    print(
        f"Pos. error: {err_riemannian_pos}\nRot. error: {err_riemannian_rot}\nCost: {sol_info['f(x)']}\nSolution time: {t_sol}."
    )
    print("------------------------------------")
    return err_riemannian_pos, err_riemannian_rot, t_sol, fail, sol_info


if __name__ == "__main__":

    np.random.seed(21)

    robot, graph = load_ur10()
    params = {
        "solver": "TrustRegions",
        "mingradnorm": 1e-10,
        "maxiter": 1000,
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
    print("Standard deviation of solution time {:}".format(np.std(np.asarray(t))))
    print("Average iterations {:}".format(np.average(nit)))
    print("Median iterations {:}".format(np.median(nit)))
    print("Average pos error {:}".format(np.average(np.asarray(e_pos))))
    print("Average rot error {:}".format(np.average(np.asarray(e_rot))))
    print("Number of fails {:}".format(sum(fails)))
