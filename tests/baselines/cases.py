"""Solver-correctness baseline case list.

A case is a deterministic IK problem + solver invocation. Cases are kept
small and CPU-only so the baseline can be regenerated in a few minutes.
Adding/removing cases is a baseline regeneration; do not silently mutate
this file once a baseline is committed.
"""
from typing import TypedDict, Literal


class Case(TypedDict):
    name: str
    robot: Literal["schunk_lwa4d", "ur10"]
    obstacles: bool
    solver: Literal["riemannian", "nonlinear_bfgs", "nonlinear_lbfgsb"]
    seed: int


CASES: list[Case] = [
    {"name": "riemann_schunk_free",      "robot": "schunk_lwa4d", "obstacles": False, "solver": "riemannian",       "seed": 42},
    {"name": "riemann_schunk_obstacles", "robot": "schunk_lwa4d", "obstacles": True,  "solver": "riemannian",       "seed": 42},
    {"name": "riemann_ur10_free",        "robot": "ur10",         "obstacles": False, "solver": "riemannian",       "seed": 42},
    {"name": "riemann_ur10_obstacles",   "robot": "ur10",         "obstacles": True,  "solver": "riemannian",       "seed": 42},
    {"name": "nonlin_bfgs_schunk",       "robot": "schunk_lwa4d", "obstacles": False, "solver": "nonlinear_bfgs",   "seed": 42},
    {"name": "nonlin_lbfgsb_schunk",     "robot": "schunk_lwa4d", "obstacles": False, "solver": "nonlinear_lbfgsb", "seed": 42},
]
