"""Preconditioner study for the Riemannian solver's truncated CG.

Compares no preconditioner vs block-Jacobi vs the full Gauss-Newton
Laplacian on UR10 with joint limits (the production configuration).
Counts HVPs (the profile's hottest operation), outer iterations, wall
time, and solution quality (final pose error after angle decoding).
"""
import time

import numpy as np

from graphik.solvers import rtr
from graphik.solvers.riemannian_solver import RiemannianSolver
from graphik.utils.dgp import (
    adjacency_matrix_from_graph,
    bound_smoothing,
    distance_matrix_from_graph,
    graph_from_pos,
)
from graphik.utils.roboturdf import load_ur10


def run_case(graph, T_goal, precon):
    solver = RiemannianSolver(graph)
    G = graph.from_pose(T_goal)
    D_goal = distance_matrix_from_graph(G)
    omega = adjacency_matrix_from_graph(G)
    lb, ub = bound_smoothing(G)

    n_hvp = [0]
    orig_for_riemannian = __import__(
        "graphik.solvers.loss", fromlist=["for_riemannian"]
    ).for_riemannian

    import graphik.solvers.loss as loss_mod

    def counting_for_riemannian(*args, **kwargs):
        cost, egrad, ehvp = orig_for_riemannian(*args, **kwargs)

        def counted_ehvp(Y, Z):
            n_hvp[0] += 1
            return ehvp(Y, Z)

        return cost, egrad, counted_ehvp

    loss_mod.for_riemannian, saved = counting_for_riemannian, loss_mod.for_riemannian
    try:
        t0 = time.perf_counter()
        out = solver.solve(
            D_goal, omega, use_limits=True, bounds=(lb, ub), precon=precon
        )
        wall = time.perf_counter() - t0
    finally:
        loss_mod.for_riemannian = saved

    G_sol = graph_from_pos(out["x"], graph.node_ids)
    q_sol = graph.joint_variables(G_sol, {f"p{graph.robot.n}": T_goal})
    T_sol = graph.robot.pose(q_sol, f"p{graph.robot.n}")
    trans_err = np.linalg.norm(T_sol[:3, 3] - T_goal[:3, 3])
    rot_err = np.linalg.norm(T_sol[:3, :3] - T_goal[:3, :3])
    limits_ok = (
        len(graph.check_distance_limits(graph.realization(q_sol), tol=1e-6)) == 0
    )
    return {
        "iters": out["iterations"],
        "hvps": n_hvp[0],
        "time": wall,
        "trans_err": trans_err,
        "rot_err": rot_err,
        "success": trans_err < 1e-2 and rot_err < 1e-2 and limits_ok,
    }


def main(n_goals=20, seed=0):
    np.random.seed(seed)
    robot, graph = load_ur10()
    goals = []
    for _ in range(n_goals):
        q = robot.random_configuration()
        goals.append(robot.pose(q, f"p{robot.n}"))

    for precon in (None, "jacobi", "gn"):
        rows = [run_case(graph, T, precon) for T in goals]
        succ = sum(r["success"] for r in rows)
        med = lambda k: np.median([r[k] for r in rows])
        tot = lambda k: np.sum([r[k] for r in rows])
        print(
            f"{str(precon):>8}: success {succ}/{n_goals} | "
            f"median outer {med('iters'):.0f} | median HVPs {med('hvps'):.0f} | "
            f"HVPs/outer {tot('hvps')/tot('iters'):.1f} | "
            f"median time {med('time')*1e3:.0f} ms | "
            f"median trans_err {med('trans_err'):.2e}"
        )


if __name__ == "__main__":
    main()
