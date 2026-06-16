"""Riemannian trust-region EDM solver and tCG preconditioners."""
from __future__ import annotations

import numpy as np

from graphik.graphs.graph import ProblemGraph
from graphik.solvers import costs, rtr
from graphik.solvers.distance_solver import (
    DistanceProblem,
    DistanceSolver,
    MinimizeInfo,
)
from graphik.utils import distance_matrix_from_pos, memoize_last
from graphik.utils.manifolds.fixed_rank_psd_sym import PSDFixedRank


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
    (w_ij * (d_goal^2 - ||y_i - y_j||^2))^2, whose GN Hessian has
    graph-Laplacian block structure with edge blocks 8 w_ij u u^T,
    u = y_i - y_j. Returns the (N, N, d, d) edge-block tensor E.
    """
    Y_diff = Y[:, None, :] - Y[None, :, :]
    return (8.0 * W)[:, :, None, None] * (
        Y_diff[..., :, None] * Y_diff[..., None, :]
    )


def make_gn_preconditioner(manifold, omega, psi_L=None, psi_U=None, floor_rel=1e-3):
    """Inverse eigenvalue-floored Gauss-Newton Hessian preconditioner."""
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


def make_jacobi_preconditioner(
    manifold, omega, psi_L=None, psi_U=None, floor_rel=1e-3
):
    """Block-Jacobi variant of ``make_gn_preconditioner``."""
    weights = _gn_weights(omega, psi_L, psi_U)

    @memoize_last
    def factor(Y):
        D = distance_matrix_from_pos(Y)
        E = _gn_blocks(Y, weights(D))
        Bd = E.sum(axis=1)
        lam, V = np.linalg.eigh(Bd)
        lam = np.maximum(lam, np.maximum(floor_rel * lam[:, -1:], 1e-12))
        return V, lam

    def precondition(Y, r):
        V, lam = factor(Y)
        rv = np.einsum("nij,ni->nj", V, r)
        z = np.einsum("nij,nj->ni", V, rv / lam)
        return manifold.projection(Y, z)

    return precondition


_RTR_PASSTHROUGH = (
    "rho_prime",
    "rho_regularization",
    "Delta_bar",
    "Delta0",
    "mininner",
    "maxinner",
)


class RiemannianSolver(DistanceSolver):
    """EDM solver on the PSD fixed-rank manifold via in-house RTR."""

    def __init__(
        self,
        graph: ProblemGraph,
        *,
        init: str = "bsmooth",
        use_limits: bool = True,
        cache: bool = True,
        precon=None,
        rtr_params: dict | None = None,
    ):
        super().__init__(graph, init=init, use_limits=use_limits, cache=cache)
        if not (precon in (None, "gn", "jacobi") or callable(precon)):
            raise ValueError("precon must be None, 'gn', 'jacobi', or callable")
        self.precon = precon
        self.rtr_params = dict(rtr_params or {})

    def _minimize(self, problem: DistanceProblem):
        manifold = PSDFixedRank(self.N, self.dim, cache_projection=self.cache)
        cost, egrad, ehvp = costs.for_riemannian(
            problem.D_goal,
            problem.omega,
            psi_L=problem.psi_L,
            psi_U=problem.psi_U,
            cache=self.cache,
        )

        if self.precon is None:
            preconditioner = None
        elif self.precon == "gn":
            preconditioner = make_gn_preconditioner(
                manifold, problem.omega, psi_L=problem.psi_L, psi_U=problem.psi_U
            )
        elif self.precon == "jacobi":
            preconditioner = make_jacobi_preconditioner(
                manifold, problem.omega, psi_L=problem.psi_L, psi_U=problem.psi_U
            )
        else:
            preconditioner = self.precon

        def rgrad(Y):
            return manifold.projection(Y, egrad(Y))

        def rhess(Y, U):
            return manifold.projection(Y, ehvp(Y, U))

        rtr_kwargs = dict(
            max_iterations=self.rtr_params.get("maxiter", 3000),
            min_gradient_norm=self.rtr_params.get("mingradnorm", 1e-8),
            theta=self.rtr_params.get("theta", 1.0),
            kappa=self.rtr_params.get("kappa", 0.1),
            preconditioner=preconditioner,
        )
        for key in _RTR_PASSTHROUGH:
            if key in self.rtr_params:
                rtr_kwargs[key] = self.rtr_params[key]

        res = rtr.trust_regions(
            manifold, cost, rgrad, rhess, problem.Y0, **rtr_kwargs
        )
        return res.point, MinimizeInfo(
            cost=res.cost,
            iterations=res.iterations,
            status=res.stopping_criterion,
        )
