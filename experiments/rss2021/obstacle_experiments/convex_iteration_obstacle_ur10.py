import time
import os
import numpy as np
import pandas as pd
import pickle
from progress.bar import ShadyBar as Bar

from graphik.graphs.graph_base import RobotRevoluteGraph
from graphik.solvers.constraints import get_full_revolute_nearest_point
from graphik.solvers.convex_iteration import convex_iterate_sdp_snl_graph
from graphik.solvers.sdp_snl import extract_solution
from graphik.utils.constants import *
from graphik.utils.roboturdf import load_ur10
from graphik.utils.utils import safe_arccos
from numpy import pi
from numpy.linalg.linalg import norm


def solve_random_problem(graph: RobotRevoluteGraph, sparse=False, closed_form=True):
    robot = graph.robot
    n = robot.n

    # ensure the goal is feasible (i.e. no collisons)
    feasible = False
    while not feasible:
        q_goal = robot.random_configuration()
        G_goal = graph.realization(q_goal)
        T_goal = robot.get_pose(q_goal, f"p{n}")
        broken_limits = graph.check_distance_limits(G_goal)
        if len(broken_limits) == 0:
            feasible = True

    input_vals = get_full_revolute_nearest_point(graph, q_goal, list(robot.structure))
    anchors = {key: input_vals[key] for key in ["p0", "q0", f"p{n}", f"q{n}"]}

    t_sol = time.perf_counter()
    (
        _,
        constraint_clique_dict,
        sdp_variable_map,
        _,
        excess_eigvalue_sum,
        _,
        primal_runtime,
        fantope_runtime
    ) = convex_iterate_sdp_snl_graph(graph, anchors, ranges=True, sparse=sparse, closed_form=closed_form)
    t_sol = time.perf_counter() - t_sol

    solution = extract_solution(constraint_clique_dict, sdp_variable_map, robot.dim)

    # expand solution to include all the points
    for node in G_goal:
        if node not in solution.keys():
            solution[node] = G_goal.nodes[node][POS]

    G_sol = graph.complete_from_pos(solution)

    q_sol = robot.joint_variables(G_sol, {f"p{n}": T_goal})
    T_riemannian = robot.get_pose(q_sol, f"p{n}")
    z_goal = T_goal.as_matrix()[:3, 2]
    z_riemannian = T_riemannian.as_matrix()[:3, 2]
    err_pos = norm(T_goal.trans - T_riemannian.trans)
    err_rot = abs(safe_arccos(z_riemannian.dot(z_goal)))

    # check for all broken distance limits
    broken_limits = graph.check_distance_limits(G_sol)

    print(
        f"Pos. error: {err_pos}\nRot. error: {err_rot}\nViolations: {broken_limits}\nTotal time: {t_sol}" +
        f"\nSolvers Time: {primal_runtime+fantope_runtime}\nExcess Eig. Sum: {excess_eigvalue_sum[-1]}"
    )
    print("------------------------------------")

    sol_data = {
        "Goal Pose": T_goal,
        "Prob. Graph": graph.directed,
        "Sol. Graph": G_sol,
        "Sol. Config": q_sol,
        "Sol. Time": t_sol,
        "Primal Time": primal_runtime,
        "Fantope Time": fantope_runtime,
        "Pos. Error": err_pos,
        "Rot. Error": err_rot,
        "Excess Eig. Sum": excess_eigvalue_sum
    }

    return sol_data, broken_limits


if __name__ == "__main__":
    ub = np.minimum(np.random.rand(6) * (pi / 2) + pi / 2, pi)
    lb = -ub
    limits = (lb, ub)

    # Convex iteration parameters
    sparse = False
    closed_form = True

    robot, graph = load_ur10(limits=None)
    obstacles = [
        (np.array([0, 1, 1]), 0.5),
        (np.array([0, 1, -1]), 0.5),
        (np.array([0, -1, 1]), 0.5),
        (np.array([1, 0, 1]), 0.5),
    ]

    for idx, obs in enumerate(obstacles):
        graph.add_spherical_obstacle(f"o{idx}", obs[0], obs[1])

    sol_data = []
    viol_data = []
    num_tests = 10  # 4000
    bar = Bar("", max=num_tests, check_tty=False, hide_cursor=False)
    for idx in range(num_tests):
        sol, viol = solve_random_problem(graph, sparse=sparse, closed_form=closed_form)
        sol_data += [sol]
        viol_data += [pd.DataFrame(viol, index=len(viol) * [idx])]
        bar.next()
    bar.finish()
    prob_cols = ["Goal Pose", "Prob. Graph"]
    sol_cols = ["Sol. Graph", "Sol. Config", "Sol. Time", "Pos. Error", "Rot. Error"]
    data = {
        "Problem": pd.DataFrame(sol_data, columns=prob_cols),
        "Solution": pd.DataFrame(sol_data, columns=sol_cols),
        "Constraint Violations": pd.concat(viol_data),
    }

    pickle.dump(
        data,
        open(
            os.path.dirname(os.path.realpath(__file__))
            + "/results/ur10_"
            + time.strftime("%Y%m%d-%H%M%S")
            + ".p",
            "wb",
        ),
    )
