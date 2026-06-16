"""Profile the in-house RTR solver on a single robot.

Runs N IK problems, reports per-pose wall-clock/convergence stats, and dumps
a cProfile of the whole run sorted by cumulative time.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time

import numpy as np

from graphik.solvers import RiemannianSolver
from graphik.solvers.initializations import INIT_STRATEGIES
from graphik.utils.roboturdf import (
    load_kuka,
    load_panda,
    load_schunk_lwa4d,
    load_schunk_lwa4p,
    load_ur10,
)


ROBOTS = {
    "ur10": load_ur10,
    "kuka": load_kuka,
    "panda": load_panda,
    "lwa4d": load_schunk_lwa4d,
    "lwa4p": load_schunk_lwa4p,
}


def _run_one(graph, T_goal, init: str, params: dict) -> dict:
    solver = RiemannianSolver(graph, init=init, rtr_params=params)
    t0 = time.perf_counter()
    try:
        res = solver.solve(T_goal)
    except Exception:
        return {
            "wall_s": time.perf_counter() - t0,
            "iterations": 0,
            "stopping_criterion": "solve/decode raised",
            "fx": float("nan"),
            "q_sol": None,
            "solver_feasible": False,
        }
    return {
        "wall_s": time.perf_counter() - t0,
        "iterations": res.iterations,
        "stopping_criterion": res.status,
        "fx": res.cost,
        "q_sol": res.q,
        "solver_feasible": res.feasible,
    }


def _pose_error(robot, ee: str, T_goal: np.ndarray, q_sol) -> tuple[bool, float, float]:
    if q_sol is None:
        return False, float("nan"), float("nan")
    T_got = np.asarray(robot.pose(q_sol, ee))
    pos = float(np.linalg.norm(T_goal[:3, 3] - T_got[:3, 3]))
    R_delta = T_goal[:3, :3] @ T_got[:3, :3].T
    cos_th = np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0)
    rot = float(np.arccos(cos_th))
    return pos < 1e-2 and rot < 1e-2, pos, rot


def _print_summary(results: list[dict], total_wall: float, args) -> None:
    walls = np.array([r["wall_s"] for r in results])
    iters = np.array([r["iterations"] for r in results])
    feas = np.array([r["feasible"] for r in results], dtype=bool)

    print(f"=== RTR profile on {args.robot} ===")
    print(f"  init:                    {args.init}")
    print(f"  n_poses:                 {args.n_poses}")
    print(f"  seed:                    {args.seed}")
    print(f"  total wall (incl prof):  {total_wall*1000:.0f}ms")
    print(f"  feasible:                {feas.sum()}/{args.n_poses}  ({feas.mean()*100:.1f}%)")
    print(
        f"  iterations:              mean {iters.mean():.1f}  "
        f"median {np.median(iters):.0f}  max {iters.max()}"
    )
    print(
        f"  per-pose wall:           mean {walls.mean()*1000:.1f}ms  "
        f"median {np.median(walls)*1000:.1f}ms  max {walls.max()*1000:.1f}ms"
    )
    if feas.any():
        pos = np.array([r["pos_err"] for r in results])[feas]
        rot = np.array([r["rot_err"] for r in results])[feas]
        print(f"  pose error (feasible):   pos mean {pos.mean():.2e}  rot mean {rot.mean():.2e}")


def _print_profile(profiler: cProfile.Profile, top: int) -> None:
    print(f"\n=== cProfile top-{top} by cumulative time (all functions) ===")
    s = io.StringIO()
    pstats.Stats(profiler, stream=s).sort_stats("cumulative").print_stats(top)
    print(s.getvalue())

    print(f"=== cProfile top-{top} by cumulative time (graphik only) ===")
    s = io.StringIO()
    pstats.Stats(profiler, stream=s).sort_stats("cumulative").print_stats("graphik", top)
    print(s.getvalue())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-poses", type=int, default=20, help="number of IK problems")
    p.add_argument("--robot", choices=tuple(ROBOTS), default="ur10")
    p.add_argument("--init", choices=INIT_STRATEGIES, default="spectral")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--mingradnorm", type=float, default=None)
    p.add_argument("--maxiter", type=int, default=None)
    p.add_argument("--theta", type=float, default=None)
    p.add_argument("--kappa", type=float, default=None)
    p.add_argument("--maxinner", type=int, default=None)
    p.add_argument("--mininner", type=int, default=None)
    p.add_argument("--rho-prime", type=float, default=None)
    p.add_argument("--delta-bar", type=float, default=None)
    p.add_argument("--delta0", type=float, default=None)
    args = p.parse_args()

    params: dict = {}
    for cli_name, key in [
        ("mingradnorm", "mingradnorm"),
        ("maxiter", "maxiter"),
        ("theta", "theta"),
        ("kappa", "kappa"),
        ("maxinner", "maxinner"),
        ("mininner", "mininner"),
        ("rho_prime", "rho_prime"),
        ("delta_bar", "Delta_bar"),
        ("delta0", "Delta0"),
    ]:
        value = getattr(args, cli_name)
        if value is not None:
            params[key] = value

    np.random.seed(args.seed)
    robot, graph = ROBOTS[args.robot]()
    ee = f"p{robot.n}"
    poses = [
        np.asarray(robot.pose(robot.random_configuration(), ee))
        for _ in range(args.n_poses)
    ]

    profiler = cProfile.Profile()
    results: list[dict] = []
    t0 = time.perf_counter()
    profiler.enable()
    try:
        for T_goal in poses:
            results.append(_run_one(graph, T_goal, args.init, params))
    finally:
        profiler.disable()
    total_wall = time.perf_counter() - t0

    # Decode is inside solve; this loop only computes pose-FK error.
    for r, T_goal in zip(results, poses):
        ok, pos_err, rot_err = _pose_error(robot, ee, T_goal, r["q_sol"])
        r["feasible"] = r["solver_feasible"] and ok
        r["pos_err"] = pos_err
        r["rot_err"] = rot_err

    _print_summary(results, total_wall, args)
    _print_profile(profiler, args.top)


if __name__ == "__main__":
    main()
