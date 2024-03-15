#!/usr/bin/env python3
import numpy as np
from numpy.testing import assert_array_less
import time
from graphik.graphs.graph_planar import ProblemGraphPlanar
from graphik.robots.robot_planar import RobotPlanar
from graphik.utils.dgp import bound_smoothing
# from graphik.solvers.solver_rfr import RiemannianSolver
from graphik.utils import (
    pos_from_graph,
    list_to_variable_dict,
    distance_matrix_from_graph,
    adjacency_matrix_from_graph,
    graph_from_pos
    )
from graphik.solvers.riemannian_solver import RiemannianSolver

def random_problem_2d_chain():
    e_rot = []
    e_pos = []
    t_sol = []
    n = 10
    fails = 0

    a = list_to_variable_dict(np.random.rand(n) * 2.0 + 1.0)
    th = list_to_variable_dict(np.zeros(n))
    angular_limits = np.minimum(np.random.rand(n) * np.pi + 0.20, np.pi)
    upper_angular_limits = list_to_variable_dict(angular_limits)
    lower_angular_limits = list_to_variable_dict(-angular_limits)
    robot_params = {
        "link_lengths": a,
        "theta": th,
        "joint_limits_upper": upper_angular_limits,
        "joint_limits_lower": lower_angular_limits,
        "num_joints": 10,
    }

    robot = RobotPlanar(robot_params)
    graph = ProblemGraphPlanar(robot)
    solver = RiemannianSolver(graph)

    # solver = RiemannianSolver(solver_params)
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
        sol_info = solver.solve(D_goal, omega, Y_init =Y_init, use_limits=True, bounds=(lb, ub), jit=False)
        G_sol = graph_from_pos(sol_info["x"], graph.node_ids)
        q_sol = graph.joint_variables(G_sol, {f"p{graph.robot.n}": T_goal})

        T_riemannian = robot.pose(q_sol, f"p{robot.n}")
        err_riemannian = (T_goal.dot(T_riemannian.inv())).log()
        err_riemannian_pos = np.linalg.norm(T_goal.trans - T_riemannian.trans)
        err_riemannian_rot = np.linalg.norm(err_riemannian[2])

        t_sol.append(sol_info['time'])
        e_rot.append(err_riemannian_rot)
        e_pos.append(err_riemannian_pos)
        q_abs = np.abs(np.array(list(q_sol.values())))
        if err_riemannian_pos > 0.01 or err_riemannian_rot > 0.01:
            fails += 1
        elif sum(q_abs > (angular_limits + 0.01 * angular_limits)) > 0:
            print("FAIL")
            fails += 1
        print(f"{idx}", end="\r")

    t_sol = np.array(t_sol[1:])
    t_sol = t_sol[abs(t_sol - np.mean(t_sol)) < 2 * np.std(t_sol)]
    print("Average solution time {:}".format(np.average(t_sol)))
    print("Standard deviation of solution time {:}".format(np.std(np.array(t_sol))))
    print("Average pos error {:}".format(np.average(np.array(e_pos))))
    print("Average rot error {:}".format(np.average(np.array(e_rot))))
    print("Number of fails {:}".format(fails))


if __name__ == "__main__":
    np.random.seed(22)
    random_problem_2d_chain()
