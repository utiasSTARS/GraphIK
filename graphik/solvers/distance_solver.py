"""Shared problem-level plumbing for the EDM-based solvers."""
from __future__ import annotations

import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import networkx as nx
import numpy as np

from graphik.graphs.graph import ProblemGraph
from graphik.solvers.base import IKResult, IKSolver
from graphik.solvers.initializations import (
    INIT_STRATEGIES,
    bsmooth_init,
    spectral_init,
    zero_init,
)
from graphik.utils import graph_from_pos
from graphik.utils.dgp import adjacency_matrix_from_graph, distance_matrix_from_graph


@dataclass
class DistanceProblem:
    """Per-solve state handed to ``_minimize``."""

    G: nx.DiGraph
    D_goal: np.ndarray
    omega: np.ndarray
    psi_L: Optional[np.ndarray]
    psi_U: Optional[np.ndarray]
    Y0: np.ndarray


@dataclass
class MinimizeInfo:
    """What ``_minimize`` reports back for ``IKResult``."""

    cost: float
    iterations: int
    status: str


class DistanceSolver(IKSolver):
    def __init__(
        self,
        graph: ProblemGraph,
        *,
        init: str = "bsmooth",
        use_limits: bool = True,
        cache: bool = True,
    ):
        super().__init__(graph)
        if init not in INIT_STRATEGIES:
            raise ValueError(
                "init must be one of "
                + ", ".join(repr(s) for s in INIT_STRATEGIES)
            )
        self.init = init
        self.use_limits = use_limits
        self.cache = cache

    def generate_initialization(self, G, D_goal, omega) -> np.ndarray:
        if self.init == "spectral":
            return spectral_init(D_goal, omega, self.dim)
        if self.init == "zero":
            return zero_init(self.graph)
        return bsmooth_init(G, omega, self.dim)

    @abstractmethod
    def _minimize(self, problem: DistanceProblem) -> Tuple[np.ndarray, MinimizeInfo]:
        ...

    def solve(self, T_goal, *, Y_init=None) -> IKResult:
        t0 = time.perf_counter()
        goals = self.goals_from(T_goal)
        G = self.graph.from_pose(goals)
        D_goal = distance_matrix_from_graph(G)
        omega = adjacency_matrix_from_graph(G)

        psi_L = psi_U = None
        if self.use_limits:
            psi_L, psi_U = self.graph.distance_bound_matrices()

        Y0 = (
            Y_init
            if Y_init is not None
            else self.generate_initialization(G, D_goal, omega)
        )

        Y, info = self._minimize(
            DistanceProblem(
                G=G,
                D_goal=D_goal,
                omega=omega,
                psi_L=psi_L,
                psi_U=psi_U,
                Y0=Y0,
            )
        )

        G_sol = graph_from_pos(Y, self.graph.node_ids)
        q = self.graph.joint_variables(G_sol, goals)
        return IKResult(
            q=q,
            cost=info.cost,
            iterations=info.iterations,
            time=time.perf_counter() - t0,
            status=info.status,
            limit_violations=self.check_limits(q),
            Y=Y,
        )
