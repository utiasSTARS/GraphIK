"""Run a baseline case and measure pose error.

`run_case(case)` is the only public entrypoint; it returns a dict that goes
straight into the baseline JSON. All randomness is reseeded per-case so cases
are independent regardless of order.
"""
from __future__ import annotations
import time
import numpy as np

from graphik.utils.roboturdf import load_schunk_lwa4d, load_ur10
from graphik.utils.utils import table_environment
from graphik.solvers.riemannian_solver import solve_with_riemannian

from tests.baselines.cases import Case

ROBOT_LOADERS = {
    "schunk_lwa4d": load_schunk_lwa4d,
    "ur10": load_ur10,
}


def _setup(case: Case):
    np.random.seed(case["seed"])
    robot, graph = ROBOT_LOADERS[case["robot"]]()
    if case["obstacles"]:
        for idx, obs in enumerate(table_environment()):
            graph.add_spherical_obstacle(f"o{idx}", obs[0], obs[1])
    q_goal = robot.random_configuration()
    T_goal = robot.pose(q_goal, f"p{robot.n}")
    return robot, graph, T_goal


def _solve_riemannian(graph, T_goal):
    q_sol, _ = solve_with_riemannian(graph, T_goal, use_jit=False)
    return q_sol


def _measure(robot, T_goal, q_sol) -> dict:
    if q_sol is None:
        return {"solved": False, "trans_err": None, "rot_err": None, "q_sol": None}
    T_sol = robot.pose(q_sol, f"p{robot.n}")
    t_err = float(np.linalg.norm(T_goal.trans - T_sol.trans))
    R_rel = T_goal.rot.as_matrix() @ T_sol.rot.as_matrix().T
    cos_theta = (np.trace(R_rel) - 1.0) / 2.0
    rot_err = float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    # q_sol stored as a list sorted by joint id for forensic comparison only;
    # the gate compares pose error, not joint values (multiple IK solutions exist).
    q_serialised = [(k, float(v)) for k, v in sorted(q_sol.items())]
    return {"solved": True, "trans_err": t_err, "rot_err": rot_err, "q_sol": q_serialised}


def run_case(case: Case) -> dict:
    """Execute a case end to end and return a JSON-serialisable result dict."""
    robot, graph, T_goal = _setup(case)
    t0 = time.perf_counter()
    if case["solver"] == "riemannian":
        q_sol = _solve_riemannian(graph, T_goal)
    else:
        raise ValueError(f"unknown solver: {case['solver']}")
    elapsed = time.perf_counter() - t0
    measured = _measure(robot, T_goal, q_sol)
    return {
        "name": case["name"],
        "robot": case["robot"],
        "obstacles": case["obstacles"],
        "solver": case["solver"],
        "seed": case["seed"],
        "elapsed_sec": elapsed,
        **measured,
    }
