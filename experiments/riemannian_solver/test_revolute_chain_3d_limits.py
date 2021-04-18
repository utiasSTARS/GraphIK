#!/usr/bin/env python3
from experiments.riemannian_solver.problem_generation import generate_revolute_problem
import graphik
from graphik.utils.roboturdf import RobotURDF
import numpy as np

from numpy import pi
from numpy.linalg import norm

from graphik.graphs import RobotRevoluteGraph
from graphik.solvers.riemannian_solver import RiemannianSolver
from graphik.utils import (
    adjacency_matrix_from_graph,
    graph_from_pos,
    bound_smoothing,
    list_to_variable_dict,
    safe_arccos,
    orthogonal_procrustes
)


def solve_random_problem(graph: RobotRevoluteGraph, solver: RiemannianSolver):
    n = graph.robot.n
    fail = False
    G, T_goal, D_goal, X_goal = generate_revolute_problem(graph)

    lb, ub = bound_smoothing(G)
    omega = adjacency_matrix_from_graph(G)

    sol_info = solver.solve(D_goal, omega, use_limits=True, bounds=(lb, ub))
    # sol_info = solver.solve(D_goal, F, Y_init=X_init, use_limits=True)
    Y = sol_info["x"]
    t_sol = sol_info["time"]

    G_raw = graph_from_pos(Y, graph.node_ids)  # not really order-dependent
    G_sol = orthogonal_procrustes(graph.base, G_raw)

    T_g = {f"p{n}": T_goal}
    q_sol = robot.joint_variables(G_sol, T_g)
    T_riemannian = robot.get_pose(list_to_variable_dict(q_sol), "p" + str(n))
    err_riemannian_pos = norm(T_goal.trans - T_riemannian.trans)

    z_goal = T_goal.as_matrix()[:3, 2]
    z_riemannian = T_riemannian.as_matrix()[:3, 2]
    err_riemannian_rot = abs(safe_arccos(z_riemannian.dot(z_goal)))


    if err_riemannian_pos > 0.01 or err_riemannian_rot > 0.01:
        fail = True

    broken_limits = graph.check_distance_limits(graph.realization(q_sol))
    if len(broken_limits)> 0:
        print(broken_limits)
        fail = True

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

    n = 6
    angular_limits = np.minimum(np.random.rand(n) * (pi / 2) + pi / 2, pi)
    ub = angular_limits
    lb = -angular_limits

    fname = graphik.__path__[0] + "/robots/urdfs/ur10_mod.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/lwa4d.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/lwa4p.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/kuka_iiwr.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/kuka_lwr.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/jaco2arm6DOF_no_hand.urdf"
    urdf_robot = RobotURDF(fname)
    robot = urdf_robot.make_Revolute3d(ub, lb)  # make the Revolute class from a URDF

    graph = RobotRevoluteGraph(robot)
    print(graph.node_ids)
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
