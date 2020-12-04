#!/usr/bin/env python3
import graphik
import numpy as np

from graphik.utils.roboturdf import RobotURDF, plot_balls_from_points
from matplotlib import pyplot as plt
from numpy import pi
from numpy.linalg import norm

from graphik.graphs.graph_base import Graph, Revolute3dRobotGraph

# from graphik.robots.revolute import Revolute3dChain
from graphik.robots.robot_base import RobotRevolute
from graphik.solvers.riemannian_solver import RiemannianSolver

from numpy.testing import assert_array_less
from graphik.utils.utils import (
    best_fit_transform,
    list_to_variable_dict,
    trans_axis,
    dZ,
    trans_axis,
    safe_arccos,
)

from graphik.utils.dgp_utils import sample_matrix, PCA, dist_to_gram

VERT_SAMPLE = 2
HOR_SAMPLE = 10


def ik_workspace_sample_riemannian(graph: Graph, solver: RiemannianSolver):
    robot = graph.robot
    n = robot.n
    q_zero = list_to_variable_dict(n * [0])
    height = norm(robot.get_pose(q_zero, f"p{n}").trans)
    width = 2 * height
    length = width

    goals = np.zeros([VERT_SAMPLE * HOR_SAMPLE * HOR_SAMPLE, graph.dim])
    for idx in range(VERT_SAMPLE):
        for jdx in range(HOR_SAMPLE):
            for kdx in range(HOR_SAMPLE):
                # TODO indexing is wrong
                goals[
                    idx * (HOR_SAMPLE * HOR_SAMPLE) + jdx * HOR_SAMPLE + kdx - 1,
                    :,
                ] = np.array(
                    [
                        2 * length * kdx / HOR_SAMPLE - length,
                        2 * width * jdx / HOR_SAMPLE - width,
                        height * idx / VERT_SAMPLE,
                    ]
                )

    results = []
    for goal in goals:
        fail = False
        G = graph.complete_from_pos({f"p{n}": goal})
        D_goal = graph.distance_matrix_from_graph(G)

        q_rand = list_to_variable_dict(graph.robot.n * [0])
        G_rand = graph.realization(q_rand)
        X_rand = graph.pos_from_graph(G_rand)
        X_init = X_rand

        lb, ub = graph.distance_bounds(G)
        F = graph.adjacency_matrix(G)
        # print(D_goal - lb ** 2)

        # sol_info = solver.solve(D_goal, F, use_limits=True, bounds=(lb, ub))
        sol_info = solver.solve(D_goal, F, use_limits=False, bounds=(lb, ub))
        # sol_info = solver.solve(D_goal, F, Y_init=X_init, use_limits=True)
        Y = sol_info["x"]
        t_sol = sol_info["time"]
        R, t = best_fit_transform(Y[[0, 1, 2, 3], :], X_rand[[0, 1, 2, 3], :])
        P_e = (R @ Y.T + t.reshape(3, 1)).T
        X_e = P_e @ P_e.T

        G_sol = graph.graph_from_pos(P_e)
        # T_g = {f"p{n}": T_goal}
        # q_sol = robot.joint_angles_from_graph(G_sol, T_g)
        q_sol = robot.joint_angles_from_graph(G_sol)
        T_riemannian = robot.get_pose(list_to_variable_dict(q_sol), "p" + str(n))
        err_riemannian_pos = norm(goal - T_riemannian.trans)

        if err_riemannian_pos > 0.01:
            fail = True

        # broken_limits = {}
        #  for key in robot.limited_joints:
        #      if abs(q_sol[key]) > (graph.robot.ub[key] * 1.01):
        #          fail = True
        #          broken_limits[key] = abs(q_sol[key]) - (graph.robot.ub[key])
        #          print(key, broken_limits[key])

        results += [[goal, fail]]
        print(
            f"Pos. error: {err_riemannian_pos}\nCost: {sol_info['f(x)']}\nSolution time: {t_sol}."
        )
        print("------------------------------------")
    return results


if __name__ == "__main__":

    np.random.seed(21)

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

    graph = Revolute3dRobotGraph(robot)
    print(graph.node_ids)
    print(robot.limit_edges)
    print(robot.limited_joints)
    solver = RiemannianSolver(graph)
    results = ik_workspace_sample_riemannian(graph, solver)

    q_zero = list_to_variable_dict(n * [0])
    urdf_robot.visualize(
        q=q_zero, with_frames=False, with_balls=False, with_robot=True, transparency=0.5
    )

    succ_points = np.empty([0, graph.dim])
    fail_points = np.empty([0, graph.dim])
    for res in results:
        if not res[1]:
            succ_points = np.vstack((succ_points, res[0]))
        else:
            fail_points = np.vstack((fail_points, res[0]))

    scene = plot_balls_from_points(
        succ_points, scene=urdf_robot.scene, return_scene_only=True, colour="blue"
    )
    scene = plot_balls_from_points(
        fail_points, scene=scene, return_scene_only=False, colour="red"
    )
