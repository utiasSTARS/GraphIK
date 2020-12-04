#!/usr/bin/env python3
import graphik
from graphik.robots.robot_base import RobotRevolute
from graphik.utils.roboturdf import RobotURDF, plot_balls_from_points
import numpy as np
import networkx as nx
from numpy import pi
from numpy.linalg import norm
from liegroups import SE3
from graphik.graphs.graph_base import Graph, Revolute3dRobotGraph
from graphik.solvers.riemannian_solver import RiemannianSolver
from graphik.utils.dgp import (
    adjacency_matrix_from_graph,
    pos_from_graph,
    graph_from_pos,
    bound_smoothing,
)
from graphik.utils.geometry import trans_axis
from graphik.utils.utils import (
    best_fit_transform,
    list_to_variable_dict,
    safe_arccos,
)


def solve_random_problem(graph: Graph, solver: RiemannianSolver):
    n = graph.robot.n
    axis_len = graph.robot.axis_length
    q_goal = graph.robot.random_configuration()
    G_goal = graph.realization(q_goal)
    X_goal = pos_from_graph(G_goal)
    D_goal = graph.distance_matrix_from_joints(q_goal)
    T_goal = robot.get_pose(list_to_variable_dict(q_goal), f"p{n}")
    q_rand = list_to_variable_dict(graph.robot.n * [0])
    G_rand = graph.realization(q_rand)
    X_rand = pos_from_graph(G_rand)
    X_init = X_rand

    goals = {
        f"p{n}": T_goal.trans,
        f"q{n}": T_goal.dot(trans_axis(axis_len, "z")).trans,
    }
    G = graph.complete_from_pos(goals)
    lb, ub = bound_smoothing(G)
    # print(ajdacency_matrix_from_graph(G))

    omega = adjacency_matrix_from_graph(G)

    # sol_info = solver.solve(D_goal, omega, use_limits=False, Y_init=X_init)
    sol_info = solver.solve(D_goal, omega, use_limits=False, bounds=(lb, ub))
    Y = sol_info["x"]
    t_sol = sol_info["time"]

    align_ind = list(np.arange(graph.dim + 1))
    for name in goals.keys():
        align_ind.append(graph.node_ids.index(name))

    R, t = best_fit_transform(Y[align_ind, :], X_goal[align_ind, :])
    P_e = (R @ Y.T + t.reshape(3, 1)).T

    G_sol = graph_from_pos(P_e, graph.node_ids)

    T_g = {f"p{n}": T_goal}
    q_sol = robot.joint_angles_from_graph(G_sol, T_g)

    T_riemannian = robot.get_pose(list_to_variable_dict(q_sol), "p" + str(n))
    err_riemannian_pos = norm(T_goal.trans - T_riemannian.trans)

    z_goal = T_goal.as_matrix()[:3, 2]
    z_riemannian = T_riemannian.as_matrix()[:3, 2]
    err_riemannian_rot = abs(safe_arccos(z_riemannian.dot(z_goal)))
    # err_riemannian_rot = norm((T_goal.rot.dot(T_riemannian.rot.inv())).log())
    # print((T_goal.rot.dot(T_riemannian.rot.inv())).log())

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

    ### UR10 DH
    n = 6
    # a = [0, -0.612, -0.5723, 0, 0, 0]
    # d = [0.1273, 0, 0, 0.1639, 0.1157, 0.0922]
    # al = [pi / 2, 0, 0, pi / 2, -pi / 2, 0]
    # th = [0, pi, 0, 0, 0, 0]
    ub = (pi) * np.ones(n)
    lb = -ub
    # modified_dh = False

    ### Schunk Powerball
    # n = 6
    # L1 = 0.205
    # L2 = 0.350
    # L3 = 0.305
    # L4 = 0.075
    # a = [0, L2, 0, 0, 0, 0]
    # d = [L1, 0, 0, L3, 0, L4]
    # al = [-pi / 2, pi, -pi / 2, pi / 2, -pi / 2, 0]
    # th = [0, -pi/2, -pi/2, 0, 0, 0]
    # ub = (pi) * np.ones(n)
    # lb = -ub
    # modified_dh = False

    ## KUKA LWR-IW
    # n = 7
    # a = [0, 0, 0, 0, 0, 0, 0]
    # d = [0, 0, 0.4, 0, 0.39, 0, 0]
    # al = [pi / 2, -pi / 2, -pi / 2, pi / 2, pi / 2, -pi / 2, 0]
    # th = [0, 0, 0, 0, 0, 0, 0]
    # angular_limits = np.minimum(np.random.rand(n) * pi + pi / 4, pi)
    # ub = angular_limits
    # lb = -angular_limits
    # modified_dh = True

    ### Jaco Arm
    # n = 6
    # D1 = 0.2755
    # D2 = 0.2050
    # D3 = 0.2050
    # D4 = 0.2073
    # D5 = 0.1038
    # D6 = 0.1038
    # D7 = 0.1600
    # e2 = 0.0098
    # a = [0, D2, 0, 0, 0, 0]
    # d = [D1, 0, -e2, -(D3 + D4), 0, -(D5 + D6)]
    # al = [pi / 2, pi, pi / 2, pi / 2, pi / 2, 0]
    # th = [0, 0, 0, 0, 0, 0]
    # ub = pi * np.ones(n)
    # lb = -ub
    # modified_dh = True

    ### Franka Emika Panda
    # n = 7
    # a = [0, 0, 0, 0.0825, -0.0825, 0, 0.088]
    # d = [0.333, 0, 0.316, 0, 0.384, 0, 0]
    # al = [0, -pi / 2, pi / 2, pi / 2, -pi / 2, pi / 2, pi / 2]
    # th = [0, 0, 0, 0, 0, 0]
    # ub = (pi) * np.ones(n)
    # lb = -ub
    # modified_dh = True

    # params = {
    #     "a": a[:n],
    #     "alpha": al[:n],
    #     "d": d[:n],
    #     "theta": th[:n],
    #     "lb": lb[:n],
    #     "ub": ub[:n],
    #     "modified_dh": modified_dh,
    # }

    # robot = RobotRevolute(params)

    # n = 7
    # ub = (pi) * np.ones(n)
    # lb = -ub
    # fname = graphik.__path__[0] + "/robots/urdfs/ur10_mod.urdf"
    fname = graphik.__path__[0] + "/robots/urdfs/lwa4p.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/lwa4d.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/panda_arm.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/kuka_iiwr.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/kuka_lwr.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/jaco2arm6DOF_no_hand.urdf"
    urdf_robot = RobotURDF(fname)
    robot = urdf_robot.make_Revolute3d(ub, lb)  # make the Revolute class from a URDF

    graph = Revolute3dRobotGraph(robot)
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
