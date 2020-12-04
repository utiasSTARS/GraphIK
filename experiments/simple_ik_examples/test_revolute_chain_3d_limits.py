#!/usr/bin/env python3
import graphik
from graphik.utils.roboturdf import RobotURDF
import numpy as np

from numpy import pi
from numpy.linalg import norm

from graphik.graphs.graph_base import Graph, Revolute3dRobotGraph
from graphik.solvers.riemannian_solver import RiemannianSolver
from graphik.utils.geometry import trans_axis
from graphik.utils.utils import (
    best_fit_transform,
    list_to_variable_dict,
    safe_arccos,
)


def solve_random_problem(graph: Revolute3dRobotGraph, solver: RiemannianSolver):
    n = graph.robot.n
    axis_len = graph.robot.axis_length
    fail = False
    q_goal = graph.robot.random_configuration()
    G_goal = graph.realization(q_goal)
    X_goal = graph.pos_from_graph(G_goal)
    D_goal = graph.distance_matrix(q_goal)
    T_goal = robot.get_pose(list_to_variable_dict(q_goal), "p" + str(n))

    q_rand = list_to_variable_dict(graph.robot.n * [0])
    G_rand = graph.realization(q_rand)
    X_rand = graph.pos_from_graph(G_rand)
    X_init = X_rand

    G = graph.complete_from_pos(
        {f"p{n}": T_goal.trans, f"q{n}": T_goal.dot(trans_axis(axis_len, "z")).trans}
    )
    lb, ub = graph.distance_bounds(G)
    F = graph.adjacency_matrix(G)
    # print(D_goal - lb ** 2)

    sol_info = solver.solve(D_goal, F, use_limits=True, bounds=(lb, ub))
    # sol_info = solver.solve(D_goal, F, Y_init=X_init, use_limits=True)
    Y = sol_info["x"]
    t_sol = sol_info["time"]
    R, t = best_fit_transform(Y[[0, 1, 2, 3], :], X_goal[[0, 1, 2, 3], :])
    P_e = (R @ Y.T + t.reshape(3, 1)).T
    X_e = P_e @ P_e.T

    G_sol = graph.graph_from_pos(P_e)
    T_g = {f"p{n}": T_goal}
    q_sol = robot.joint_angles_from_graph(G_sol, T_g)
    T_riemannian = robot.get_pose(list_to_variable_dict(q_sol), "p" + str(n))
    err_riemannian_pos = norm(T_goal.trans - T_riemannian.trans)

    z_goal = T_goal.as_matrix()[:3, 2]
    z_riemannian = T_riemannian.as_matrix()[:3, 2]
    err_riemannian_rot = abs(safe_arccos(z_riemannian.dot(z_goal)))
    # err_riemannian_rot = norm((T_goal.rot.dot(T_riemannian.rot.inv())).log())
    # q_abs = np.abs(np.array(list(q_sol.values())))
    if err_riemannian_pos > 0.01 or err_riemannian_rot > 0.01:
        fail = True

    broken_limits = {}
    for key in robot.limited_joints:
        if abs(q_sol[key]) > (graph.robot.ub[key] * 1.01):
            fail = True
            broken_limits[key] = abs(q_sol[key]) - (graph.robot.ub[key])
            print(key, broken_limits[key])

    # e_sol = np.sqrt(norm_sq((T_goal.dot(T_riemannian.inv())).log()))
    print(
        f"Pos. error: {err_riemannian_pos}\nRot. error: {err_riemannian_rot}\nCost: {sol_info['f(x)']}\nSolution time: {t_sol}."
    )
    print("------------------------------------")
    return err_riemannian_pos, err_riemannian_rot, t_sol, fail


if __name__ == "__main__":

    np.random.seed(21)

    # UR10 DH params theta, d, alpha, a
    # n = 6
    # a = [0, -0.612, -0.5723, 0, 0, 0]
    # d = [0.1273, 0, 0, 0.1639, 0.1157, 0.0922]
    # al = [pi / 2, 0, 0, pi / 2, -pi / 2, 0]
    # th = [0, 0, 0, 0, 0, 0]
    # angular_limits = np.minimum(np.random.rand(n) * (pi/2) + pi / 2, pi)
    # ub = angular_limits
    # lb = -angular_limits
    # modified_dh = False
    # ub = (pi / 2) * np.ones(n)
    # lb = -ub

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
    # angular_limits = np.minimum(np.random.rand(n) * (pi/2) + pi / 2, pi)
    # ub = angular_limits
    # lb = -angular_limits
    # modified_dh = False
    # #
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

    n = 7
    angular_limits = np.minimum(np.random.rand(n) * (pi / 2) + pi / 2, pi)
    ub = angular_limits
    lb = -angular_limits

    # fname = graphik.__path__[0] + "/robots/urdfs/ur10_mod.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/lwa4d.urdf"
    fname = graphik.__path__[0] + "/robots/urdfs/lwa4p.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/kuka_iiwr.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/kuka_lwr.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/jaco2arm6DOF_no_hand.urdf"
    urdf_robot = RobotURDF(fname)
    robot = urdf_robot.make_Revolute3d(ub, lb)  # make the Revolute class from a URDF

    graph = Revolute3dRobotGraph(robot)
    print(graph.node_ids)
    print(robot.limit_edges)
    print(robot.limited_joints)
    # print(robot.get_pose(robot.random_configuration(), "p7"))
    solver = RiemannianSolver(graph)
    num_tests = 100
    e_pos = []
    e_rot = []
    t = []
    fails = []
    for _ in range(num_tests):
        e_r_pos, e_r_rot, t_sol, fail = solve_random_problem(graph, solver)
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
