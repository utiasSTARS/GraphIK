"""Preconditioner study for the Riemannian solver's truncated CG."""
import time

import numpy as np

from graphik.solvers import RiemannianSolver
import graphik.solvers.costs as costs_mod
from graphik.utils.roboturdf import load_ur10


def run_case(graph, T_goal, precon):
    solver = RiemannianSolver(graph, precon=precon)

    n_hvp = [0]
    saved = costs_mod.for_riemannian

    def counting_for_riemannian(*args, **kwargs):
        cost, egrad, ehvp = saved(*args, **kwargs)

        def counted_ehvp(Y, Z):
            n_hvp[0] += 1
            return ehvp(Y, Z)

        return cost, egrad, counted_ehvp

    costs_mod.for_riemannian = counting_for_riemannian
    try:
        t0 = time.perf_counter()
        res = solver.solve(T_goal)
        wall = time.perf_counter() - t0
    finally:
        costs_mod.for_riemannian = saved

    T_sol = graph.robot.pose(res.q, f"p{graph.robot.n}")
    trans_err = np.linalg.norm(T_sol[:3, 3] - T_goal[:3, 3])
    rot_err = np.linalg.norm(T_sol[:3, :3] - T_goal[:3, :3])
    return {
        "iters": res.iterations,
        "hvps": n_hvp[0],
        "time": wall,
        "trans_err": trans_err,
        "rot_err": rot_err,
        "success": trans_err < 1e-2 and rot_err < 1e-2 and res.feasible,
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
