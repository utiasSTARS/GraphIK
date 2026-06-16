"""Shared contract for the local (non-SDP) IK solvers.

Every solver constructs with a ProblemGraph and exposes
``solve(T_goal, **per_problem) -> IKResult``. ``T_goal`` is a single
homogeneous transform for the primary end effector, or a
``{node: transform}`` dict accepted by ``ProblemGraph.from_pose``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from graphik.graphs.graph import ProblemGraph

LIMIT_TOL = 1e-6


@dataclass
class IKResult:
    q: dict
    cost: float
    iterations: int
    time: float
    status: str
    limit_violations: list
    Y: np.ndarray | None = None

    @property
    def feasible(self) -> bool:
        return not self.limit_violations


class IKSolver(ABC):
    """Problem-level IK solver: construct once per graph, solve per goal."""

    def __init__(self, graph: ProblemGraph):
        self.graph = graph
        self.robot = graph.robot
        self.dim = graph.dim
        self.N = graph.number_of_nodes()

    def goals_from(self, T_goal) -> dict:
        """Normalize ``T_goal`` to a ``{node: transform}`` mapping."""
        if isinstance(T_goal, dict):
            return dict(T_goal)
        return {self.robot.end_effectors[0]: T_goal}

    def check_limits(self, q: dict) -> list:
        return self.graph.check_distance_limits(
            self.graph.realization(q), tol=LIMIT_TOL
        )

    @abstractmethod
    def solve(self, T_goal, **kwargs) -> IKResult:
        ...
