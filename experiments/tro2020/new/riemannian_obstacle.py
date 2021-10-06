import time
import sys
import os
import argparse
import numpy as np
import pandas as pd
import pickle
from progress.bar import ShadyBar as Bar

from graphik.graphs import ProblemGraphRevolute
from graphik.utils import *
from graphik.utils.roboturdf import load_ur10, load_kuka, load_schunk_lwa4d
from graphik.solvers.riemannian_solver import RiemannianSolver
from numpy import pi
from numpy.linalg.linalg import norm


def solve_problem(
        graph: ProblemGraphRevolute, solver, sparse=False, closed_form=True, prob_data=None
):
    robot = graph.robot
    n = robot.n
    # q_rand = list_to_variable_dict(n * [0])
    # G_rand = graph.realization(q_rand)
    # Y_init = pos_from_graph(G_rand)

    G = prob_data["G_partial"]
    T_goal = prob_data["T_goal"]
    true_feasibility = prob_data["Feasibility"]

    omega = adjacency_matrix_from_graph(G)
    lb, ub = bound_smoothing(G)
    D_goal = distance_matrix_from_graph(G)
    # sol_info = solver.solve(D_goal, omega, use_limits=True, Y_init = Y_init)
    sol_info = solver.solve(D_goal, omega, use_limits=True, bounds = (lb,ub))
    Y_sol = sol_info["x"]
    t_sol = sol_info["time"]
    G_sol = graph_from_pos(Y_sol, graph.node_ids)  # not really order-dependent

    q_sol = robot.joint_variables(G_sol, {f"p{n}": T_goal})

    broken_limits = graph.check_distance_limits(graph.realization(q_sol))
    feasible = len(broken_limits) == 0

    T_riemannian = robot.get_pose(q_sol, f"p{n}")
    z_goal = T_goal.as_matrix()[:3, 2]
    z_riemannian = T_riemannian.as_matrix()[:3, 2]
    err_pos = norm(T_goal.trans - T_riemannian.trans)
    err_rot = abs(safe_arccos(z_riemannian.dot(z_goal)))

    # print(
    #     f"\nPos. error: {err_pos}\nRot. error: {err_rot}\nViolations: {broken_limits}\nTotal time: {t_sol}"
    #     + f"\nSolvers Time: {primal_runtime+fantope_runtime}\nExcess Eig. Sum: {excess_eigvalue_sum[-1]}",
    # )
    # print("------------------------------------")

    sol_data = {
        "Goal Pose": T_goal,
        "Prob. Graph": graph.directed,
        "True Feasible": true_feasibility,
        "Sol. Graph": G_sol,
        "Sol. Config": q_sol,
        "Sol. Time": t_sol,
        "Sol. Feasible": feasible,
        "Pos. Error": err_pos,
        "Rot. Error": err_rot,
    }
    return sol_data, broken_limits

def main(args):
    robot_name = args.robot[0]
    env_name = args.env[0]
    num_prob = args.num_prob[0]

    # np.random.seed(21)

    if robot_name == "ur10":
        robot, graph = load_ur10(limits=None)
    elif robot_name == "kuka":
        robot, graph = load_kuka(limits=None)
    elif robot_name == "schunk":
        robot, graph = load_schunk_lwa4d(limits=None)
    else:
        raise NotImplementedError

    # robot, graph = load_truncated_ur10(6)
    obstacles = []
    if env_name == "icosahedron":
        phi = (1 + np.sqrt(5)) / 2
        scale = 0.5
        radius = 0.5
        obstacles = [
            (scale * np.asarray([0, 1, phi]), radius),
            (scale * np.asarray([0, 1, -phi]), radius),
            (scale * np.asarray([0, -1, -phi]), radius),
            (scale * np.asarray([0, -1, phi]), radius),
            (scale * np.asarray([1, phi, 0]), radius),
            (scale * np.asarray([1, -phi, 0]), radius),
            (scale * np.asarray([-1, -phi, 0]), radius),
            (scale * np.asarray([-1, phi, 0]), radius),
            (scale * np.asarray([phi, 0, 1]), radius),
            (scale * np.asarray([-phi, 0, 1]), radius),
            (scale * np.asarray([-phi, 0, -1]), radius),
            (scale * np.asarray([phi, 0, -1]), radius),
        ]

    if env_name == "cube":
        scale = 0.5
        radius = 0.45
        obstacles = [
            (scale * np.asarray([-1, 1, 1]), radius),
            (scale * np.asarray([-1, 1, -1]), radius),
            (scale * np.asarray([1, -1, 1]), radius),
            (scale * np.asarray([1, -1, -1]), radius),
            (scale * np.asarray([1, 1, 1]), radius),
            (scale * np.asarray([1, 1, -1]), radius),
            (scale * np.asarray([-1, -1, 1]), radius),
            (scale * np.asarray([-1, -1, -1]), radius),
        ]
    if env_name == "octahedron":
        scale = 0.75
        radius = 0.4
        obstacles = [
            (scale * np.asarray([1, 1, 0]), radius),
            (scale * np.asarray([1, -1, 0]), radius),
            (scale * np.asarray([-1, 1, 0]), radius),
            (scale * np.asarray([-1, -1, 0]), radius),
            (scale * np.asarray([0, 0, 1]), radius),
            (scale * np.asarray([0, 0, -1]), radius),
        ]
    for idx, obs in enumerate(obstacles):
        graph.add_spherical_obstacle(f"o{idx}", obs[0], obs[1])

    sol_data = []
    viol_data = []
    bar = Bar(
        "Riemannian " + robot_name + ", " + str(num_prob) + ", " + env_name,
        max=num_prob,
        check_tty=False,
        hide_cursor=False,
    )

    # Convex iteration parameters
    sparse = False
    closed_form = True

    problems_base_path = os.path.dirname(os.path.realpath(__file__)) + "/generated_problems/"
    problems_path = problems_base_path + robot_name + "_" + env_name + ".p"

    with open(problems_path, "rb") as f:
        try:
            problem_data = pickle.load(f)
        except (OSError, IOError) as e:
            problem_data = None

    problem_data = problem_data[
        (problem_data["Robot"] == robot_name)
        & (problem_data["Environment"] == env_name)
    ]

    for idx in range(num_prob):
        params = {
            "solver": "TrustRegions",
            "mingradnorm": 1e-10,
            "maxiter": 3000,
            "theta": 1,
            "logverbosity": 0,
        }
        solver = RiemannianSolver(graph, params)
        sol, viol = solve_problem(
            graph,
            solver,
            sparse=sparse,
            closed_form=closed_form,
            prob_data=problem_data.iloc[idx],
        )
        sol_data += [sol]
        viol_data += [pd.DataFrame(viol, index=len(viol) * [idx])]
        bar.next()
    bar.finish()
    prob_cols = ["Goal Pose", "Prob. Graph", "True Feasible"]
    sol_cols = [
        "Sol. Graph",
        "Sol. Config",
        "Sol. Time",
        "Sol. Feasible",
        "Pos. Error",
        "Rot. Error",
    ]
    data = {
        "Problem": pd.DataFrame(sol_data, columns=prob_cols),
        "Solution": pd.DataFrame(sol_data, columns=sol_cols),
        "Constraint Violations": pd.concat(viol_data),
    }

    storage_base_path = os.path.dirname(os.path.realpath(__file__)) + "/results/"
    path = storage_base_path + robot_name + "_RI_" + env_name + ".p"
    if not os.path.exists(storage_base_path):
        os.makedirs(storage_base_path)
    with open(path, "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":

    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--robot",
        nargs="*",
        type=str,
        default=["ur10"],
    )
    CLI.add_argument(
        "--env",
        nargs="*",
        type=str,
        default=[""],
    )
    CLI.add_argument(
        "--num_prob",
        nargs="*",
        type=int,
        default=[100],
    )

    args = CLI.parse_args()

    main(args)
