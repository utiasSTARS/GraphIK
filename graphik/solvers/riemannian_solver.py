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
    distance_matrix_from_pos,
    memoize_last,
)
from graphik.utils.manifolds.fixed_rank_psd_sym import PSDFixedRank
from graphik.graphs.graph import ProblemGraph
from graphik.solvers import rtr, loss


def _gn_weights(omega, psi_L, psi_U):
    """Active-edge weight matrix for the Gauss-Newton model: equality edges
    plus whichever bound edges are active at the current distances."""
    if psi_L is None:
        return lambda D: omega
    diff = psi_L != psi_U
    LL = diff * (psi_L > 0)
    UU = diff * (psi_U > 0)

    def weights(D):
        return omega + LL * (D < psi_L) + UU * (D > psi_U)

    return weights


def _gn_blocks(Y, W):
    """Per-edge Gauss-Newton blocks of the EDM loss at Y.

    The cost is sum over (undirected, active) edges of
    (w_ij * (d_goal² - ||y_i - y_j||²))², whose GN Hessian has graph-Laplacian
    block structure with edge blocks 8 w_ij u u^T, u = y_i - y_j. Returns the
    (N, N, d, d) edge-block tensor E (E[i, i] = 0).
    """
    Y_diff = Y[:, None, :] - Y[None, :, :]
    return (8.0 * W)[:, :, None, None] * (
        Y_diff[..., :, None] * Y_diff[..., None, :]
    )


def make_gn_preconditioner(manifold, omega, psi_L=None, psi_U=None,
                           floor_rel=1e-3):
    """Preconditioner for RTR's truncated CG: the inverse of the (eigenvalue-
    floored) Gauss-Newton Hessian of the EDM loss.

    The GN matrix is N*d x N*d — small for IK-sized graphs — so one eigh per
    outer iteration (memoized per Y) buys a near-exact Newton preconditioner.
    The flooring keeps it positive definite: the GN matrix is PSD with a
    nullspace containing the translation directions. Output is projected back
    to the tangent (horizontal) space.
    """
    weights = _gn_weights(omega, psi_L, psi_U)

    @memoize_last
    def factor(Y):
        N, d = Y.shape
        D = distance_matrix_from_pos(Y)
        E = _gn_blocks(Y, weights(D))
        B = -E
        idx = np.arange(N)
        B[idx, idx] = E.sum(axis=1)
        H = B.transpose(0, 2, 1, 3).reshape(N * d, N * d)
        lam, V = np.linalg.eigh(H)
        lam = np.maximum(lam, max(floor_rel * lam[-1], 1e-12))
        return V, lam

    def precondition(Y, r):
        V, lam = factor(Y)
        z = (V @ ((V.T @ r.ravel()) / lam)).reshape(r.shape)
        return manifold.projection(Y, z)

    return precondition


def make_jacobi_preconditioner(manifold, omega, psi_L=None, psi_U=None,
                               floor_rel=1e-3):
    """Block-Jacobi variant of `make_gn_preconditioner`: keeps only the N
    diagonal d x d blocks of the GN Hessian. Cheaper apply, weaker model."""
    weights = _gn_weights(omega, psi_L, psi_U)

    @memoize_last
    def factor(Y):
        D = distance_matrix_from_pos(Y)
        E = _gn_blocks(Y, weights(D))
        Bd = E.sum(axis=1)                       # (N, d, d) diagonal blocks
        lam, V = np.linalg.eigh(Bd)              # batched
        lam = np.maximum(lam, np.maximum(floor_rel * lam[:, -1:], 1e-12))
        return V, lam

    def precondition(Y, r):
        V, lam = factor(Y)
        rv = np.einsum("nij,ni->nj", V, r)
        z = np.einsum("nij,nj->ni", V, rv / lam)
        return manifold.projection(Y, z)

    return precondition


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
        precon=None,
    ):
        manifold = PSDFixedRank(
            self.N, self.dim,
            jit=self.jit,
            cache_projection=self.cache,
        )

        psi_L = psi_U = None
        if not use_limits:
            cost, egrad, ehess = self.create_cost(D_goal, omega)
        else:
            psi_L, psi_U = self.graph.distance_bound_matrices()
            cost, egrad, ehess = self.create_cost_limits(
                D_goal, omega, psi_L, psi_U
            )

        if precon is None:
            preconditioner = None
        elif precon == "gn":
            preconditioner = make_gn_preconditioner(
                manifold, omega, psi_L=psi_L, psi_U=psi_U
            )
        elif precon == "jacobi":
            preconditioner = make_jacobi_preconditioner(
                manifold, omega, psi_L=psi_L, psi_U=psi_U
            )
        elif callable(precon):
            preconditioner = precon
        else:
            raise ValueError("precon must be None, 'gn', 'jacobi', or callable")

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
            preconditioner=preconditioner,
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

def solve_with_riemannian(graph, T_goal, use_jit=False, cache=True, precon=None):
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
    precon : None | "gn" | "jacobi" | callable, default None
        tCG preconditioner. "gn" (inverse Gauss-Newton Hessian, one small
        eigh per outer iteration) cuts inner HVPs ~10x and wall time 2-5x
        at equal-or-better success rate on the UR10 benchmark — see
        experiments/rtr_preconditioner_study.py. Off by default because it
        changes the optimization trajectory (baselines would shift).

    Uses the bsmooth initialization (the ``RiemannianSolver`` default), so the
    triangle-inequality bounds from ``bound_smoothing(G)`` are computed here
    and forwarded as the ``bounds`` argument to ``solve``.
    """
    G = graph.from_pose(T_goal)
    solver = RiemannianSolver(graph, jit=use_jit, cache=cache)
    D_goal = distance_matrix_from_graph(G)
    omega = adjacency_matrix_from_graph(G)
    lb, ub = bound_smoothing(G)
    sol_info = solver.solve(
        D_goal, omega, use_limits=True, bounds=(lb, ub), precon=precon
    )
    G_sol = graph_from_pos(sol_info["x"], graph.node_ids)
    q_sol = graph.joint_variables(G_sol, {f"p{graph.robot.n}": T_goal})

    broken_limits = graph.check_distance_limits(graph.realization(q_sol), tol=1e-6)
    if len(broken_limits) > 0:
        return None, None
    else:
        return q_sol, sol_info["x"]
