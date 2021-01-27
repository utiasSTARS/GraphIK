import numpy as np
import time

from numpy.linalg.linalg import norm
from graphik.solvers.convex_iteration import convex_iterate_sdp_snl
from graphik.solvers.sdp_formulations import SdpSolverParams
from graphik.graphs.graph_base import RobotRevoluteGraph
from graphik.utils.utils import safe_arccos
from graphik.utils.dgp import graph_from_pos_dict, pos_from_graph, graph_from_pos
from graphik.utils.constants import *
from graphik.solvers.sdp_snl import (
    solve_linear_cost_sdp,
    distance_constraints,
    extract_full_sdp_solution,
    extract_solution,
)
from graphik.solvers.constraints import get_full_revolute_nearest_point
from graphik.utils.roboturdf import load_ur10


def solve_random_problem(graph: RobotRevoluteGraph):
    robot = graph.robot
    n = robot.n
    t_sol = 0

    q_goal = robot.random_configuration()
    G_goal = graph.realization(q_goal)
    Y_goal = pos_from_graph(G_goal)
    T_goal = robot.get_pose(q_goal, f"p{n}")

    full_points = [f"p{idx}" for idx in range(0, n + 1)] + [
        f"q{idx}" for idx in range(0, n + 1)
    ]

    input_vals = get_full_revolute_nearest_point(graph, q_goal, full_points)
    end_effectors = {key: input_vals[key] for key in ["p0", "q0", f"p{n}", f"q{n}"]}

    canonical_point_order = [
        point for point in full_points if point not in end_effectors.keys()
    ]

    t_sol = time.perf_counter()
    (
        C,
        constraint_clique_dict,
        sdp_variable_map,
        canonical_point_order,
        eig_value_sum_vs_iterations,
        prob,
    ) = convex_iterate_sdp_snl(robot, end_effectors)
    t_sol = time.perf_counter() - t_sol

    solution = extract_solution(constraint_clique_dict, sdp_variable_map, robot.dim)

    # expand solution to include all the points
    for node in G_goal:
        if node not in solution.keys():
            solution[node] = G_goal.nodes[node][POS]

    G_sol = graph_from_pos_dict(solution)
    q_sol = robot.joint_variables(G_sol, {f"p{n}": T_goal})
    T_riemannian = robot.get_pose(q_sol, f"p{n}")

    err_riemannian_pos = norm(T_goal.trans - T_riemannian.trans)

    z_goal = T_goal.as_matrix()[:3, 2]
    z_riemannian = T_riemannian.as_matrix()[:3, 2]
    err_riemannian_rot = abs(safe_arccos(z_riemannian.dot(z_goal)))

    print(
        f"Pos. error: {err_riemannian_pos}\nRot. error: {err_riemannian_rot}\nSolution time: {t_sol}"
    )
    print("------------------------------------")
    return err_riemannian_pos, err_riemannian_rot, t_sol


if __name__ == "__main__":
    robot, graph = load_ur10()
    num_tests = 20
    e_pos = []
    e_rot = []
    t = []
    for _ in range(num_tests):
        e_r_pos, e_r_rot, t_sol = solve_random_problem(graph)
        e_pos += [e_r_pos]
        e_rot += [e_r_rot]
        t += [t_sol]

    t = np.array(t)
    t = t[abs(t - np.mean(t)) < 2 * np.std(t)]

    print("Average solution time {:}".format(np.average(t)))
    print("Average pos error {:}".format(np.average(np.array(e_pos))))
    print("Average rot error {:}".format(np.average(np.array(e_rot))))
