#!/usr/bin/env python3
from graphik.utils.dgp import (
    adjacency_matrix_from_graph, bound_smoothing,
    distance_matrix_from_graph, graph_from_pos,
)

import numpy as np
from graphik.utils import (
    MDS,
    linear_projection,
    gram_from_distance_matrix,
)
from graphik.utils.manifolds.fixed_rank_psd_sym import PSDFixedRank
from graphik.graphs.graph import ProblemGraph
from graphik.solvers import rtr, loss


class RiemannianSolver:
    def __init__(
        self,
        graph: ProblemGraph,
        jit=False,
        cache=True,
        init="bsmooth",
        *args,
        **kwargs,
    ):
        if "cost_type" in kwargs:
            raise TypeError(
                "cost_type was removed in the loss-module consolidation. "
                "The single dense backend is selected by default; pass "
                "jit=True for the AOT-compiled kernels."
            )
        self.params = {}
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.graph = graph
        self.dim = graph.dim
        self.N = graph.number_of_nodes()
        self.jit = jit
        # Single switch that enables both per-Y memoization caches:
        # the solver's cost-state builder and the manifold's projection
        # operator. Both are perf-only — same algorithmic trajectory.
        self.cache = cache
        if init not in ("spectral", "bsmooth"):
            raise ValueError("init must be 'spectral' or 'bsmooth'")
        self.init = init

    def generate_initialization(self, D_goal, omega, bounds=None):
        """Build a starting point for RTR. Dispatches on `self.init`:

        - 'spectral' (default): Smith–Cai–Tasissa style. Center the partially-
          observed distance matrix (zeros for unknowns), eigendecompose, take
          the top-d eigenvectors weighted by sqrt(eigvals). One N×N symmetric
          eigendecomposition.
        - 'bsmooth': legacy. Triangle-inequality smoothing of distance bounds
          (networkx all-pairs Bellman-Ford), sample an EDM at 0.9 of (lb, ub),
          classical MDS, then project to dim along the known-edge structure.
          Requires a ``bounds=(lb, ub)`` tuple (e.g. from ``bound_smoothing``).
        """
        if self.init == "spectral":
            n = D_goal.shape[0]
            D_partial = omega * D_goal
            J = np.eye(n) - np.full((n, n), 1.0 / n)
            G = -0.5 * J @ D_partial @ J
            eigvals, eigvecs = np.linalg.eigh(G)            # ascending
            top = np.argsort(-eigvals)[:self.dim]
            # Floor eigenvalues to keep Y full-rank: PSDFixedRank requires
            # rank == dim, and degenerate problems can yield fewer than dim
            # positive eigvals here, which would otherwise produce zero
            # columns and break the manifold projection.
            lam = np.maximum(eigvals[top], 1e-12)
            return eigvecs[:, top] * np.sqrt(lam)

        # bsmooth
        if bounds is None:
            raise ValueError("init='bsmooth' requires bounds=(lb, ub)")
        lb, ub = bounds
        lb_sqrt = np.sqrt(lb)
        ub_sqrt = np.sqrt(ub)
        D_rand = (lb_sqrt + 0.9 * (ub_sqrt - lb_sqrt)) ** 2
        X_rand = MDS(gram_from_distance_matrix(D_rand), eps=1e-8)
        return linear_projection(X_rand, omega, self.dim)

    def create_cost(self, D_goal, omega):
        return loss.for_riemannian(
            D_goal, omega, jit=self.jit, cache=self.cache,
        )

    def create_cost_limits(self, D_goal, omega, psi_L, psi_U):
        return loss.for_riemannian(
            D_goal, omega, psi_L=psi_L, psi_U=psi_U,
            jit=self.jit, cache=self.cache,
        )

    def solve(
        self,
        D_goal,
        omega,
        use_limits=False,
        bounds=None,
        Y_init=None,
        method=None,
        output_log=True,
    ):
        manifold = PSDFixedRank(
            self.N, self.dim,
            jit=self.jit,
            cache_projection=self.cache,
        )

        if not use_limits:
            cost, egrad, ehess = self.create_cost(D_goal, omega)
        else:
            psi_L, psi_U = self.graph.distance_bound_matrices()
            cost, egrad, ehess = self.create_cost_limits(
                D_goal, omega, psi_L, psi_U
            )

        if Y_init is None:
            Y_init = self.generate_initialization(D_goal, omega, bounds=bounds)

        def rgrad(Y):
            return manifold.projection(Y, egrad(Y))

        def rhess(Y, U):
            return manifold.projection(Y, ehess(Y, U))

        rtr_kwargs = dict(
            max_iterations=self.params.get("maxiter", 3000),
            min_gradient_norm=self.params.get("mingradnorm", 1e-8),
            theta=self.params.get("theta", 1.0),
            kappa=self.params.get("kappa", 0.1),
        )
        for k in ("rho_prime", "rho_regularization",
                  "Delta_bar", "Delta0", "mininner", "maxinner"):
            if k in self.params:
                rtr_kwargs[k] = self.params[k]
        res = rtr.trust_regions(manifold, cost, rgrad, rhess, Y_init, **rtr_kwargs)

        if output_log:
            return {
                "x": res.point,
                "f(x)": res.cost,
                "iterations": res.iterations,
                "stopping_criterion": res.stopping_criterion,
                "time": res.time,
                "gradnorm": res.gradient_norm,
            }
        return res.point

def solve_with_riemannian(graph, T_goal, use_jit=False, cache=True):
    """One-shot wrapper: build the IK problem from ``T_goal``, solve via
    ``RiemannianSolver``, decode joint angles from the recovered point cloud.

    Parameters
    ----------
    use_jit : bool, default False
        Use the AOT-compiled cost kernels when True. False keeps the pure
        numpy path; build the kernels first via ``python -m graphik.solvers.costs``
        before enabling this.
    cache : bool, default True
        Memoize per-Y cost state and projection ops across cost/grad/HVP
        calls within a single iteration. Pure perf, identical trajectory.

    Uses the bsmooth initialization (the ``RiemannianSolver`` default), so the
    triangle-inequality bounds from ``bound_smoothing(G)`` are computed here
    and forwarded as the ``bounds`` argument to ``solve``.
    """
    G = graph.from_pose(T_goal)
    solver = RiemannianSolver(graph, jit=use_jit, cache=cache)
    D_goal = distance_matrix_from_graph(G)
    omega = adjacency_matrix_from_graph(G)
    lb, ub = bound_smoothing(G)
    sol_info = solver.solve(D_goal, omega, use_limits=True, bounds=(lb, ub))
    G_sol = graph_from_pos(sol_info["x"], graph.node_ids)
    q_sol = graph.joint_variables(G_sol, {f"p{graph.robot.n}": T_goal})

    broken_limits = graph.check_distance_limits(graph.realization(q_sol), tol=1e-6)
    if len(broken_limits) > 0:
        return None, None
    else:
        return q_sol, sol_info["x"]
