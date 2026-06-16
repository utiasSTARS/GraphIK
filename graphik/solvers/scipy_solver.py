"""EDM solver backed by scipy.optimize.minimize."""
from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, minimize

from graphik.graphs.graph import ProblemGraph
from graphik.solvers import costs
from graphik.solvers.distance_solver import (
    DistanceProblem,
    DistanceSolver,
    MinimizeInfo,
)
from graphik.utils.constants import POS

_HESSP_METHODS = frozenset(
    {"Newton-CG", "trust-ncg", "trust-krylov", "trust-constr"}
)

_METHOD_DEFAULTS = {
    "L-BFGS-B": {"ftol": 1e-16, "gtol": 1e-16},
    "BFGS": {"xrtol": 0.75e-6, "gtol": 0.25e-6, "norm": np.inf},
}


class ScipySolver(DistanceSolver):
    def __init__(
        self,
        graph: ProblemGraph,
        *,
        init: str = "bsmooth",
        use_limits: bool = True,
        cache: bool = True,
        method: str = "BFGS",
        options: dict | None = None,
    ):
        super().__init__(graph, init=init, use_limits=use_limits, cache=cache)
        self.method = method
        self.options = dict(options or {})

    def position_constraints(self, G) -> Bounds:
        """Pin POS-tagged nodes of ``G`` to their stored positions.

        Called with the per-solve goal graph, so the end-effector goal nodes
        (POS-tagged there) are pinned alongside the base anchors.
        """
        N = G.number_of_nodes()
        ub = np.full((N, self.dim), np.inf)
        lb = np.full((N, self.dim), -np.inf)
        for idx, (node, data) in enumerate(G.nodes(data=True)):
            if POS in data:
                ub[idx] = data[POS]
                lb[idx] = data[POS]
        return Bounds(lb=lb.flatten(), ub=ub.flatten())

    def _minimize(self, problem: DistanceProblem):
        cost_and_grad, hessp = costs.for_minimize(
            problem.D_goal,
            problem.omega,
            dim=self.dim,
            psi_L=problem.psi_L,
            psi_U=problem.psi_U,
            cache=self.cache,
        )

        bounds = None
        if self.method == "L-BFGS-B":
            bounds = self.position_constraints(problem.G)
        options = {**_METHOD_DEFAULTS.get(self.method, {}), **self.options}

        minimize_kwargs = dict(
            jac=True, method=self.method, bounds=bounds, options=options
        )
        if hessp is not None and self.method in _HESSP_METHODS:
            minimize_kwargs["hessp"] = hessp

        sol = minimize(
            cost_and_grad,
            np.ascontiguousarray(problem.Y0.flatten()),
            **minimize_kwargs,
        )
        return sol.x.reshape(-1, self.dim), MinimizeInfo(
            cost=float(sol.fun),
            iterations=int(sol.nit),
            status=str(sol.message),
        )
