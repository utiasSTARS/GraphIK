#!/usr/bin/env python3
import numpy as np
from numpy.testing import assert_array_less
import networkx as nx
import time
from graphik.graphs.graph_planar import ProblemGraphPlanar
from graphik.robots.robot_planar import RobotPlanar
# from graphik.solvers.solver_rfr import RiemannianSolver
from graphik.utils import (
    pos_from_graph,
    list_to_variable_dict,
    distance_matrix_from_graph,
    adjacency_matrix_from_graph,
    graph_from_pos
    )
from graphik.solvers.riemannian_solver import RiemannianSolver
from graphik.utils.dgp import bound_smoothing


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
    robot_params = {
        "link_lengths": a,
        "theta": th,
        "joint_limits_upper": lim_u,
        "joint_limits_lower": lim_l,
        "num_joints": 10,
    }

    robot = RobotPlanar(robot_params)
    graph = ProblemGraphPlanar(robot)
    solver = RiemannianSolver(graph)

    n_tests = 100

    q_init = list_to_variable_dict(n * [0])
    G_init = graph.realization(q_init)
    Y_init = pos_from_graph(G_init)

    t_sol = []
    for idx in range(n_tests):

        q_goal = robot.random_configuration()
        T_goal = robot.pose(q_goal, f"p{robot.n}")

        G = graph.from_pose(T_goal)
        D_goal = distance_matrix_from_graph(G)
        omega = adjacency_matrix_from_graph(G)
        lb, ub = bound_smoothing(G)
        sol_info = solver.solve(D_goal, omega, Y_init =Y_init, bounds=(lb,ub), jit=False)
        G_sol = graph_from_pos(sol_info["x"], graph.node_ids)
        q_sol = graph.joint_variables(G_sol, {f"p{graph.robot.n}": T_goal})

        T_riemannian = robot.pose(q_sol, f"p{robot.n}")
        err_riemannian = (T_goal.dot(T_riemannian.inv())).log()
        err_riemannian_pos = np.linalg.norm(T_goal.trans - T_riemannian.trans)
        err_riemannian_rot = np.linalg.norm(err_riemannian[2])

        t_sol.append(sol_info['time'])
        e_rot.append(err_riemannian_rot)
        e_pos.append(err_riemannian_pos)
        if err_riemannian_pos > 0.01 or err_riemannian_rot > 0.01:
            fails += 1

    t_sol = np.array(t_sol[1:])
    t_sol = t_sol[abs(t_sol - np.mean(t_sol)) < 2 * np.std(t_sol)]
    print("Average solution time {:}".format(np.average(t_sol)))
    print("Standard deviation of solution time {:}".format(np.std(np.array(t_sol))))
    print("Average pos error {:}".format(np.average(np.array(e_pos))))
    print("Average rot error {:}".format(np.average(np.array(e_rot))))
    print("Number of fails {:}".format(fails))

    assert_array_less(e_pos, 1e-4 * np.ones(n_tests))


if __name__ == "__main__":
    np.random.seed(21)
    random_problem_2d_chain()
