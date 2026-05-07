#!/usr/bin/env python3
import time
import numpy as np

from graphik.utils import (
    MDS,
    linear_projection,
    gram_from_distance_matrix,
)
from graphik.graphs.graph import ProblemGraph
from scipy.optimize import Bounds, minimize
from graphik.utils.constants import POS
from graphik.solvers import loss

# scipy.optimize.minimize methods that consume hessp. Passing the kwarg to
# other methods just provokes a UserWarning per call.
_HESSP_METHODS = frozenset({
    "Newton-CG", "trust-ncg", "trust-krylov", "trust-constr",
})


class NonlinearSolver:
    def __init__(self, graph: ProblemGraph, jit=False, *args, **kwargs):
        """Distance-based IK solver wrapping ``scipy.optimize.minimize``.

        Cost / gradient / HVP come from ``graphik.solvers.loss.for_minimize``,
        which dispatches between a NumPy-dense backend and the AOT-compiled
        ``costgrd`` kernels based on ``jit``.
        """
        if "cost_type" in kwargs:
            raise TypeError(
                "cost_type was removed in the loss-module consolidation. "
                "The single dense backend is selected by default; pass "
                "jit=True for the AOT-compiled kernels."
            )
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.graph = graph
        self.dim = graph.dim
        self.N = graph.number_of_nodes()
        self.jit = jit

    def generate_initialization(self, bounds, dim, omega):
        """Sample an EDM within the supplied (lb, ub) bounds, then MDS + project."""
        lb = np.sqrt(bounds[0])
        ub = np.sqrt(bounds[1])
        D_rand = (lb + 0.9 * (ub - lb)) ** 2
        X_rand = MDS(gram_from_distance_matrix(D_rand), eps=1e-8)
        return linear_projection(X_rand, omega, dim)

    def create_cost(self, D_goal, omega):
        return loss.for_minimize(
            D_goal, omega, dim=self.dim, jit=self.jit,
        )

    def create_cost_limits(self, D_goal, omega, psi_L, psi_U):
        return loss.for_minimize(
            D_goal, omega, psi_L=psi_L, psi_U=psi_U,
            dim=self.dim, jit=self.jit,
        )

    def position_constraints(self):
        """Pin POS-tagged nodes to their stored goal positions.

        Used by L-BFGS-B to enforce known positions (end-effector goal, base
        anchor, axis-frame nodes) as hard equality constraints rather than as
        soft residuals. Reads positions from the graph; do not pass a random
        initialization here.
        """
        ub_ = np.ones((self.N, self.dim)) * np.inf
        lb_ = -np.ones((self.N, self.dim)) * np.inf
        for idx, (node, data) in enumerate(self.graph.nodes(data=True)):
            if POS in data:
                ub_[idx] = data[POS]
                lb_[idx] = data[POS]
        return Bounds(lb=lb_.flatten(), ub=ub_.flatten())

    def solve(
        self,
        D_goal,
        omega,
        use_limits=False,
        bounds=None,
        Y_init=None,
        output_log=True,
        method='BFGS',
        options=None,
    ):
        """Run scipy.optimize.minimize on the EDM loss."""
        if use_limits:
            psi_L, psi_U = self.graph.distance_bound_matrices()
            cost_and_grad, hessp = self.create_cost_limits(D_goal, omega, psi_L, psi_U)
        else:
            cost_and_grad, hessp = self.create_cost(D_goal, omega)

        if Y_init is None:
            Y_init = self.generate_initialization(bounds, self.dim, omega)
        Yi = np.ascontiguousarray(Y_init.flatten())

        bnds = None
        if method == 'L-BFGS-B':
            bnds = self.position_constraints()
            defaults = {"ftol": 1e-16, "gtol": 1e-16, "iprint": -1}
        elif method == 'BFGS':
            defaults = {"xrtol": 0.75e-6, "gtol": 0.25e-6, "norm": np.inf}
        else:
            defaults = {}
        options = {**defaults, **(options or {})}

        minimize_kwargs = dict(
            jac=True, method=method, bounds=bnds, options=options,
        )
        if hessp is not None and method in _HESSP_METHODS:
            minimize_kwargs["hessp"] = hessp

        start_time = time.time()
        sol = minimize(cost_and_grad, Yi, **minimize_kwargs)
        end_time = time.time()

        if output_log:
            return {
                "x": sol.x.reshape(omega.shape[0], self.dim),
                "time": end_time - start_time,
                "iterations": sol.nit,
                "f(x)": sol.fun,
            }
        return sol.x
