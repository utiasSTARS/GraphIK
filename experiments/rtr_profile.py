"""Profile the in-house RTR solver on a single robot.

Runs N IK problems on ur10 (or another robot via --robot), reports per-
pose wall-clock and convergence stats, and dumps a cProfile of the
whole run sorted by cumulative time.

Usage:
    python experiments/rtr_profile.py [--n-poses 20] [--init spectral|bsmooth]
                                       [--robot ur10] [--seed 0] [--top 20]
                                       [--jit]
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time

import numpy as np

from graphik.utils import (
    adjacency_matrix_from_graph,
    bound_smoothing,
    distance_matrix_from_graph,
    graph_from_pos,
)
from graphik.utils.roboturdf import (
    load_kuka,
    load_panda,
    load_schunk_lwa4d,
    load_schunk_lwa4p,
    load_ur10,
)
from graphik.solvers.riemannian_solver import RiemannianSolver


ROBOTS = {
    "ur10": load_ur10,
    "kuka": load_kuka,
    "panda": load_panda,
    "lwa4d": load_schunk_lwa4d,
    "lwa4p": load_schunk_lwa4p,
}


def _run_one(graph, T_goal, init: str, jit: bool, params: dict) -> dict:
    G_partial = graph.from_pose(T_goal)
    D_goal = distance_matrix_from_graph(G_partial)
    omega = adjacency_matrix_from_graph(G_partial)
    bounds = bound_smoothing(G_partial) if init in ("bsmooth", "bspectral") else None

    t0 = time.perf_counter()
    solver = RiemannianSolver(graph, jit=jit, init=init)
    solver.params.update(params)
    sol = solver.solve(D_goal, omega, use_limits=True, bounds=bounds)
    wall = time.perf_counter() - t0
    return {
        "wall_s": wall,
        "rtr_time_s": sol["time"],
        "iterations": sol["iterations"],
        "stopping_criterion": sol["stopping_criterion"],
        "gradnorm": sol["gradnorm"],
        "fx": sol["f(x)"],
        "Y_sol": sol["x"],
    }


def _pose_error(graph, robot, ee: str, T_goal: np.ndarray, Y_sol: np.ndarray) -> tuple[bool, float, float]:
    G_sol = graph_from_pos(Y_sol, graph.node_ids)
    try:
        q_sol = graph.joint_variables(G_sol, {ee: T_goal})
    except Exception:
        return False, float("nan"), float("nan")
    T_got = np.asarray(robot.pose(q_sol, ee))
    pos = float(np.linalg.norm(T_goal[:3, 3] - T_got[:3, 3]))
    R_delta = T_goal[:3, :3] @ T_got[:3, :3].T
    cos_th = np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0)
    rot = float(np.arccos(cos_th))
    return pos < 1e-2 and rot < 1e-2, pos, rot


def _print_summary(results: list[dict], total_wall: float, args) -> None:
    walls = np.array([r["wall_s"] for r in results])
    rtr_times = np.array([r["rtr_time_s"] for r in results])
    iters = np.array([r["iterations"] for r in results])
    feas = np.array([r["feasible"] for r in results], dtype=bool)

    print(f"=== RTR profile on {args.robot} ===")
    print(f"  init:                    {args.init}")
    print(f"  jit:                     {args.jit}")
    print(f"  n_poses:                 {args.n_poses}")
    print(f"  seed:                    {args.seed}")
    print(f"  total wall (incl prof):  {total_wall*1000:.0f}ms")
    print(f"  feasible:                {feas.sum()}/{args.n_poses}  ({feas.mean()*100:.1f}%)")
    print(f"  iterations:              mean {iters.mean():.1f}  median {np.median(iters):.0f}  max {iters.max()}")
    print(f"  per-pose wall:           mean {walls.mean()*1000:.1f}ms  median {np.median(walls)*1000:.1f}ms  max {walls.max()*1000:.1f}ms")
    print(f"  rtr-internal time:       mean {rtr_times.mean()*1000:.1f}ms  (vs {walls.mean()*1000:.1f}ms wall — overhead {(walls.mean()-rtr_times.mean())*1000:.1f}ms)")
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
    p.add_argument("--n-poses", type=int, default=20, help="number of IK problems (default: 20)")
    p.add_argument("--robot", choices=tuple(ROBOTS), default="ur10")
    p.add_argument("--init", choices=("spectral", "bsmooth"), default="spectral")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top", type=int, default=20, help="cProfile rows to print (default: 20)")
    p.add_argument("--jit", action="store_true", help="use AOT-compiled costgrd kernels")
    # Solver hyperparameter overrides (forwarded to RiemannianSolver.params).
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
        v = getattr(args, cli_name)
        if v is not None:
            params[key] = v

    np.random.seed(args.seed)
    robot, graph = ROBOTS[args.robot]()
    ee = f"p{robot.n}"
    poses = [np.asarray(robot.pose(robot.random_configuration(), ee)) for _ in range(args.n_poses)]

    profiler = cProfile.Profile()
    results: list[dict] = []
    t0 = time.perf_counter()
    profiler.enable()
    try:
        for T_goal in poses:
            results.append(_run_one(graph, T_goal, args.init, args.jit, params))
    finally:
        profiler.disable()
    total_wall = time.perf_counter() - t0

    # Feasibility check is outside the profile so it doesn't pollute solver
    # function timings; it's a tiny pose-FK roundtrip per pose.
    for r, T_goal in zip(results, poses):
        ok, pos_err, rot_err = _pose_error(graph, robot, ee, T_goal, r["Y_sol"])
        r["feasible"] = ok
        r["pos_err"] = pos_err
        r["rot_err"] = rot_err

    _print_summary(results, total_wall, args)
    _print_profile(profiler, args.top)


if __name__ == "__main__":
    main()
