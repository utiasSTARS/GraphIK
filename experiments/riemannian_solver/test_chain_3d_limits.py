#!/usr/bin/env python3
import numpy as np
from numpy import pi
from numpy.testing import assert_array_less
import networkx as nx
import time
from graphik.graphs import RobotSphericalGraph
from graphik.robots import RobotSpherical
from graphik.solvers.riemannian_solver import RiemannianSolver
from graphik.utils import *

# NOTE needs fixing
def random_problem_3d_chain():
    e_rot = []
    e_pos = []
    t_sol = []
    n = 10
    fails = 0

    a = list_to_variable_dict(0 * np.random.rand(n))
    d = list_to_variable_dict(np.random.rand(n))
    al = list_to_variable_dict(0 * np.random.rand(n))
    th = list_to_variable_dict(0 * np.random.rand(n))
    angular_limits = np.minimum(np.random.rand(n) * pi + 0.2, pi)
    ub = angular_limits
    lb = -angular_limits

    params = {
        "a": a,
        "alpha": al,
        "d": d,
        "theta": th,
        "joint_limits_lower": lb,
        "joint_limits_upper": ub,
    }

    robot = RobotSpherical(params)
    graph = RobotSphericalGraph(robot)
    solver = RiemannianSolver(graph)
    n_tests = 100

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
        # q_rand = graph.robot.random_configuration()
        # G_rand = graph.realization(q_rand)
        # X_rand = graph.vertex_positions(G_rand)

        goals = {f"p{n-1}": X_goal[-2, :], f"p{n}": X_goal[-1, :]}
        # goals = {f"p{n}": X_goal[-1, :]}
        G = graph.complete_from_pos(goals)
        # lb, ub = bound_smoothing(G)  # will take goals and jli
        F = adjacency_matrix_from_graph(G)

        # sol_info = solver.solve(D_goal, F, bounds=(lb, ub), use_limits = True)
        sol_info = solver.solve(D_goal, F, Y_init=X_init, use_limits=True)
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
            np.log(T_riemannian.as_matrix()[:3, 2] @ T_goal.as_matrix()[:3, 2])
        )

        e_rot += [err_riemannian_rot]
        e_pos += [err_riemannian_pos]
        # q_abs = np.abs(np.array(list(q_sol.values())))
        if err_riemannian_pos > 0.01 or err_riemannian_rot > 0.01:
            fails += 1
        else:
            q_abs = []
            for key in q_sol.keys():
                q_abs += [q_sol[key][1]]
            # broken_limits = {}
            for key in q_sol:
                if abs(q_sol[key][1]) > (graph.robot.ub[key] * 1.01):
                    fails += 1
                    break
        print(f"{idx}", end="\r")

    t_sol = np.array(t_sol)
    t_sol = t_sol[abs(t_sol - np.mean(t_sol)) < 2 * np.std(t_sol)]
    print("Average solution time {:}".format(np.average(t_sol)))
    print("Standard deviation of solution time {:}".format(np.std(np.array(t_sol))))
    print("Average pos error {:}".format(np.average(np.array(e_pos))))
    print("Average rot error {:}".format(np.average(np.array(e_rot))))
    print("Number of fails {:}".format(fails))


if __name__ == "__main__":
    np.random.seed(21)
    random_problem_3d_chain()
