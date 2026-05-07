"""Benchmark the non-SDP IK solvers across multiple robot arms.

Compares wall time and pose-recovery accuracy across:
- Riemannian (RTR) with spectral init from the partial-Gram block
- Riemannian (RTR) with the legacy bound-smoothing + MDS init
- BFGS                   — scipy.optimize.minimize, method='BFGS'
- L-BFGS-B               — scipy.optimize.minimize, method='L-BFGS-B' with anchor pinning

Wall time includes any per-pose initialization the solver performs (e.g.
bound-smoothing for the legacy init or spectral eigendecomposition for the
new one).

Usage:
    python experiments/solver_benchmark.py [--robots ur10,kuka,...] [--n-poses N]
                                            [--obstacles] [--seed S]
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from graphik.utils.utils import table_environment
from graphik.utils import (
    distance_matrix_from_graph,
    adjacency_matrix_from_graph,
    bound_smoothing,
    graph_from_pos,
)
from graphik.utils.roboturdf import (
    load_ur10,
    load_kuka,
    load_panda,
    load_schunk_lwa4d,
    load_schunk_lwa4p,
)
from graphik.solvers.riemannian_solver import RiemannianSolver
from graphik.solvers.nonlinear_solver import NonlinearSolver


ROBOTS = {
    "ur10":  load_ur10,
    "kuka":  load_kuka,
    "panda": load_panda,
    "lwa4d": load_schunk_lwa4d,
    "lwa4p": load_schunk_lwa4p,
}


@dataclass
class Result:
    wall_s: float
    pos_err_m: float
    rot_err_rad: float
    feasible: bool


def pose_error(T_goal: np.ndarray, T_got: np.ndarray) -> tuple[float, float]:
    pos = float(np.linalg.norm(T_goal[:3, 3] - T_got[:3, 3]))
    R_delta = T_goal[:3, :3] @ T_got[:3, :3].T
    cos_theta = np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0)
    return pos, float(np.arccos(cos_theta))


# Each runner takes the problem and returns the recovered Y (n_nodes × dim).
# Wall time is measured around each runner so pre-processing (init) is included.

def run_rtr_spectral(graph, D_goal, omega, G_partial):
    solver = RiemannianSolver(graph, jit=False, init="spectral")
    return solver.solve(D_goal, omega, use_limits=True)["x"]


def run_rtr_bsmooth(graph, D_goal, omega, G_partial):
    lb, ub = bound_smoothing(G_partial)
    solver = RiemannianSolver(graph, jit=False, init="bsmooth")
    return solver.solve(D_goal, omega, use_limits=True, bounds=(lb, ub))["x"]


def run_bfgs(graph, D_goal, omega, G_partial):
    lb, ub = bound_smoothing(G_partial)
    solver = NonlinearSolver(graph)
    # gtol=1e-8 to match RTR's stopping tolerance. scipy's BFGS does not
    # accept ftol, so we don't pass one.
    return solver.solve(
        D_goal, omega, use_limits=True, bounds=(lb, ub), method="BFGS",
        options={"gtol": 1e-8},
    )["x"]


def run_lbfgsb(graph, D_goal, omega, G_partial):
    lb, ub = bound_smoothing(G_partial)
    solver = NonlinearSolver(graph)
    return solver.solve(
        D_goal, omega, use_limits=True, bounds=(lb, ub), method="L-BFGS-B",
        options={"gtol": 1e-8, "ftol": 0},
    )["x"]


CONFIGS = [
    ("rtr-spectral", run_rtr_spectral),
    ("rtr-bsmooth",  run_rtr_bsmooth),
    ("bfgs",         run_bfgs),
    ("l-bfgs-b",     run_lbfgsb),
]


def _eval(robot, graph, T_goal, ee, runner) -> Result:
    G_partial = graph.from_pose(T_goal)
    D_goal = distance_matrix_from_graph(G_partial)
    omega = adjacency_matrix_from_graph(G_partial)
    t0 = time.perf_counter()
    try:
        Y_sol = runner(graph, D_goal, omega, G_partial)
    except Exception:
        return Result(time.perf_counter() - t0, np.nan, np.nan, feasible=False)
    wall = time.perf_counter() - t0
    G_sol = graph_from_pos(Y_sol, graph.node_ids)
    try:
        q_sol = graph.joint_variables(G_sol, {ee: T_goal})
    except Exception:
        return Result(wall, np.nan, np.nan, feasible=False)
    T_got = robot.pose(q_sol, ee)
    pos_err, rot_err = pose_error(T_goal, np.asarray(T_got))
    feasible = pos_err < 1e-2 and rot_err < 1e-2
    return Result(wall, pos_err, rot_err, feasible)


def run_robot(robot_name, n_poses, obstacles, seed):
    np.random.seed(seed)
    robot, graph = ROBOTS[robot_name]()
    if obstacles:
        for idx, obs in enumerate(table_environment()):
            graph.add_spherical_obstacle(f"o{idx}", obs[0], obs[1])
    ee = f"p{robot.n}"

    poses = [np.asarray(robot.pose(robot.random_configuration(), ee)) for _ in range(n_poses)]

    results: dict[str, list[Result]] = {label: [] for label, _ in CONFIGS}
    for T_goal in poses:
        for label, runner in CONFIGS:
            results[label].append(_eval(robot, graph, T_goal, ee, runner))
    return results


def summarize(robot_name, results):
    header = (
        f"  {'solver':14s}  {'mean wall':>10s}  {'median':>8s}  "
        f"{'pos err':>10s}  {'rot err':>10s}  {'feas %':>7s}"
    )
    print(f"\n=== {robot_name} ===")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, _ in CONFIGS:
        rs = results[label]
        walls = np.array([r.wall_s for r in rs])
        feas = np.array([r.feasible for r in rs], dtype=bool)
        pos = np.array([r.pos_err_m for r in rs])
        rot = np.array([r.rot_err_rad for r in rs])
        if feas.any():
            pos_mean = float(np.nanmean(pos[feas]))
            rot_mean = float(np.nanmean(rot[feas]))
        else:
            pos_mean = rot_mean = float("nan")
        print(
            f"  {label:14s}  {walls.mean():>10.4f}  {np.median(walls):>8.4f}  "
            f"{pos_mean:>10.2e}  {rot_mean:>10.2e}  {feas.mean()*100:>6.1f}%"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robots", type=str, default=",".join(ROBOTS.keys()),
                   help="comma-separated subset of: " + ", ".join(ROBOTS.keys()))
    p.add_argument("--n-poses", type=int, default=20)
    p.add_argument("--obstacles", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    selected = [r.strip() for r in args.robots.split(",") if r.strip()]
    bad = [r for r in selected if r not in ROBOTS]
    if bad:
        raise SystemExit(f"unknown robots: {bad}; choices are {list(ROBOTS)}")

    print(f"n_poses={args.n_poses}  obstacles={args.obstacles}  seed={args.seed}")
    for robot_name in selected:
        print(f"\nrunning {robot_name}...", flush=True)
        results = run_robot(robot_name, args.n_poses, args.obstacles, args.seed)
        summarize(robot_name, results)


if __name__ == "__main__":
    main()
