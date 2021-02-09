import numpy as np
import time

from numpy.linalg.linalg import norm
from numpy import pi
from graphik.solvers.convex_iteration import (
    convex_iterate_sdp_snl_graph,
)
from graphik.graphs.graph_base import RobotRevoluteGraph
from graphik.utils.utils import safe_arccos
from graphik.utils.dgp import graph_from_pos_dict
from graphik.utils.constants import *
from graphik.solvers.sdp_snl import (
    extract_solution,
)
from graphik.solvers.constraints import get_full_revolute_nearest_point
from graphik.utils.roboturdf import load_ur10


def check_collison(P: np.ndarray, obstacles: list):

    for obs in obstacles:
        center = obs[0]
        radius = obs[1]
        if any(
            np.diag((P - center) @ np.identity(len(center)) @ (P - center).T)
            <= radius ** 2
        ):
            return True
    return False


def solve_random_problem(graph: RobotRevoluteGraph):
    robot = graph.robot
    n = robot.n
    t_sol = 0

    q_goal = robot.random_configuration()
    G_goal = graph.realization(q_goal)
    T_goal = robot.get_pose(q_goal, f"p{n}")

    full_points = [f"p{idx}" for idx in range(0, n + 1)] + [
        f"q{idx}" for idx in range(0, n + 1)
    ]

    input_vals = get_full_revolute_nearest_point(graph, q_goal, full_points)
    anchors = {key: input_vals[key] for key in ["p0", "q0", f"p{n}", f"q{n}"]}

    t_sol = time.perf_counter()
    (
        _,
        constraint_clique_dict,
        sdp_variable_map,
        _,
        _,
        _,
    ) = convex_iterate_sdp_snl_graph(graph, anchors, ranges=True, sparse=True)
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

    fail = False
    if err_riemannian_pos > 0.01 or err_riemannian_rot > 0.01:
        fail = True

    broken_limits = {}
    for key in robot.limited_joints:
        if abs(q_sol[key]) > (graph.robot.ub[key] * 1.01):
            fail = True
            broken_limits[key] = abs(q_sol[key]) - (graph.robot.ub[key])
            print(key, broken_limits[key])
    print(
        f"Pos. error: {err_riemannian_pos}\nRot. error: {err_riemannian_rot}\nSolution time: {t_sol}"
    )
    print("------------------------------------")
    return err_riemannian_pos, err_riemannian_rot, t_sol, fail


if __name__ == "__main__":
    np.random.seed(21)
    ub = np.minimum(np.random.rand(6) * (pi / 2) + pi / 2, pi)
    lb = -ub
    limits = (lb, ub)

    robot, graph = load_ur10(limits)
    num_tests = 100
    e_pos = []
    e_rot = []
    t = []
    fails = []
    for _ in range(num_tests):
        e_r_pos, e_r_rot, t_sol, fail = solve_random_problem(graph)
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
