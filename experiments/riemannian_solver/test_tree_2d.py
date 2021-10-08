#!/usr/bin/env python3
import numpy as np
from numpy.testing import assert_array_less
import networkx as nx
from graphik.graphs import ProblemGraphPlanar
from graphik.robots import RobotPlanar
from graphik.solvers.riemannian_solver import RiemannianSolver
from graphik.utils.utils import best_fit_transform, list_to_variable_dict
from graphik.utils.dgp import (
    adjacency_matrix_from_graph,
    pos_from_graph,
    graph_from_pos,
    bound_smoothing,
)


def random_problem_2d_tree():
    e_sol = []
    e_sol_D = []
    t_sol = []
    height = 3
    gen = nx.balanced_tree(2, height, create_using=nx.DiGraph)
    gen = nx.relabel_nodes(gen, {node: f"p{node}" for node in gen})
    n = gen.number_of_edges()
    parents = nx.to_dict_of_lists(gen)

    a = list_to_variable_dict(np.ones(n))
    th = list_to_variable_dict(np.zeros(n))
    lim_u = list_to_variable_dict(np.pi * np.ones(n))
    lim_l = list_to_variable_dict(-np.pi * np.ones(n))
    params = {
        "a": a,
        "theta": th,
        "parents": parents,
        "joint_limits_upper": lim_u,
        "joint_limits_lower": lim_l,
    }

    robot = RobotPlanar(params)
    graph = ProblemGraphPlanar(robot)
    solver = RiemannianSolver(graph)
    n_tests = 100
    for idx in range(n_tests):
        q_goal = graph.robot.random_configuration()
        G_goal = graph.realization(q_goal)
        X_goal = pos_from_graph(G_goal)
        D_goal = graph.distance_matrix_from_joints(q_goal)

        goals = {}
        for idx, ee_pair in enumerate(robot.end_effectors):
            goals[ee_pair[0]] = robot.get_pose(q_goal, ee_pair[0]).trans
            # goals[ee_pair[1]] = robot.get_pose(q_goal, ee_pair[1]).trans
            #
        align_ind = list(np.arange(graph.dim + 1))
        for name in goals.keys():
            align_ind.append(graph.node_ids.index(name))

        G = graph.from_pos(goals)
        lb, ub = bound_smoothing(G)
        F = adjacency_matrix_from_graph(G)

        q_init = list_to_variable_dict(n * [0])
        G_init = graph.realization(q_init)
        X_init = pos_from_graph(G_init)

        sol_info = solver.solve(
            D_goal, F, use_limits=False, bounds=(lb, ub), max_attempts=10
        )
        # Y = solver.solve(D_goal, F, X = X_init, use_limits=False, max_attempts=10)
        Y = sol_info["x"]
        t_sol = sol_info["time"]

        R, t = best_fit_transform(Y[align_ind, :], X_goal[align_ind, :])
        P_e = (R @ Y.T + t.reshape(2, 1)).T
        X_e = P_e @ P_e.T
        G_e = graph_from_pos(P_e, graph.node_ids)

        q_sol = robot.joint_variables(G_e)
        G_sol = graph.realization(q_sol)
        D_sol = graph.distance_matrix_from_joints(q_sol)

        e = 0
        for key, value in goals.items():
            e_pos = robot.get_pose(q_sol, key).trans - value
            e += e_pos.T @ e_pos
        e_sol += [e]
        e_D = F * (np.sqrt(D_sol) - np.sqrt(D_goal))
        e_sol_D += [abs(max(e_D.min(), e_D.max(), key=abs))]
    # print(F)
    print("Average solution time {:}".format(np.average(np.array(t_sol))))
    print("Standard deviation of solution time {:}".format(np.std(np.array(t_sol))))
    print("Average maximum error {:}".format(np.average(np.array(e_sol))))
    print("Standard deviation of maximum error {:}".format(np.std(np.array(e_sol))))
    print(t_sol)
    print(e_sol_D)
    assert_array_less(e_sol, 1e-4 * np.ones(n_tests))


if __name__ == "__main__":
    np.random.seed(21)
    random_problem_2d_tree()
