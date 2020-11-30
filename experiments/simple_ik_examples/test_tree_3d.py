#!/usr/bin/env python3
import numpy as np
from numpy.testing import assert_array_less
import networkx as nx
from graphik.graphs.graph_base import SphericalRobotGraph
from graphik.robots.robot_base import RobotSpherical
from graphik.solvers.riemannian_solver import RiemannianSolver
from graphik.utils.utils import best_fit_transform, list_to_variable_dict


def random_problem_3d_tree():
    e_sol = []
    e_sol_D = []
    t_sol = []
    height = 4
    gen = nx.balanced_tree(2, height, create_using=nx.DiGraph)
    gen = nx.relabel_nodes(gen, {node: f"p{node}" for node in gen})
    n = gen.number_of_edges()
    # Generate random DH parameters
    a = list_to_variable_dict(0 * np.random.rand(n))
    d = list_to_variable_dict(np.random.rand(n))
    al = list_to_variable_dict(0 * np.random.rand(n))
    th = list_to_variable_dict(0 * np.random.rand(n))
    parents = nx.to_dict_of_lists(gen)
    lim_u = list_to_variable_dict(np.pi * np.ones(n))
    lim_l = list_to_variable_dict(-np.pi * np.ones(n))

    params = {
        "a": a,
        "alpha": al,
        "d": d,
        "theta": th,
        "parents": parents,
        "joint_limits_lower": lim_l,
        "joint_limits_upper": lim_u,
    }
    robot = RobotSpherical(params)  # instantiate robot
    graph = SphericalRobotGraph(robot)  # instantiate graph
    solver = RiemannianSolver(graph)
    n_tests = 10
    for idx in range(n_tests):

        q_goal = graph.robot.random_configuration()
        G_goal = graph.realization(q_goal)
        X_goal = graph.pos_from_graph(G_goal)
        D_goal = graph.distance_matrix(q_goal)

        goals = {}
        for idx, ee_pair in enumerate(robot.end_effectors):
            goals[ee_pair[0]] = robot.get_pose(q_goal, ee_pair[0]).trans
            goals[ee_pair[1]] = robot.get_pose(q_goal, ee_pair[1]).trans

        G = graph.complete_from_pos(goals)
        lb, ub = graph.distance_bounds(G)
        F = graph.adjacency_matrix(G)

        sol_info = solver.solve(D_goal, F, use_limits=False, bounds=(lb, ub))
        Y = sol_info["x"]
        t_sol = sol_info["time"]

        R, t = best_fit_transform(Y[[0, 1, 2, 3], :], X_goal[[0, 1, 2, 3], :])
        P_e = (R @ Y.T + t.reshape(graph.dim, 1)).T
        X_e = P_e @ P_e.T
        G_e = graph.graph_from_pos(P_e)

        q_sol = robot.joint_variables(G_e)
        G_sol = graph.realization(q_sol)
        D_sol = graph.distance_matrix(q_sol)

        e = 0
        for key, value in goals.items():
            e_pos = robot.get_pose(q_sol, key).trans - value
            e += e_pos.T @ e_pos
        e_sol += [e]
        e_D = F * (np.sqrt(D_sol) - np.sqrt(D_goal))
        e_sol_D += [abs(max(e_D.min(), e_D.max(), key=abs))]

    print("Average solution time {:}".format(np.average(np.array(t_sol))))
    print("Standard deviation of solution time {:}".format(np.std(np.array(t_sol))))
    print("Average maximum error {:}".format(np.average(np.array(e_sol))))
    print("Standard deviation of maximum error {:}".format(np.std(np.array(e_sol))))

    assert_array_less(e_sol, 1e-4 * np.ones(n_tests))


if __name__ == "__main__":
    np.random.seed(21)
    random_problem_3d_tree()
