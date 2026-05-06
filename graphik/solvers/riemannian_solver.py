#!/usr/bin/env python3
from graphik.utils.dgp import (
    adjacency_matrix_from_graph, bound_smoothing,
    distance_matrix_from_graph, graph_from_pos, sample_matrix,
)

import numpy as np
from graphik.utils import (
    distance_matrix_from_gram,
    distance_matrix_from_pos,
    MDS,
    linear_projection,
    gram_from_distance_matrix,
    memoize_last,
)
from graphik.utils.manifolds.fixed_rank_psd_sym import PSDFixedRank
from graphik.graphs.graph import ProblemGraph
from graphik.solvers import rtr


def _import_costgrd():
    """Lazy-import the AOT-compiled cost kernels. Only invoked when jit=True."""
    try:
        from graphik.solvers.costgrd import (
            jcost, jgrad, jhess, lcost, lgrad, lhess,
        )
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "jit=True requires the AOT-compiled costgrd module. "
            "Build it via 'python -m graphik.solvers.costs'."
        ) from e
    return jcost, jgrad, jhess, lcost, lgrad, lhess


class RiemannianSolver:
    def __init__(
        self,
        graph: ProblemGraph,
        cost_type="dense",
        jit=False,
        cache=True,
        init="bsmooth",
        *args,
        **kwargs,
    ):
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
        self._memo = memoize_last if cache else (lambda f: f)
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
        if self.jit:
            inds = np.nonzero(np.triu(omega))
            jcost, jgrad, jhess, _, _, _ = _import_costgrd()

            def cost(Y):
                return jcost(Y, D_goal, inds)

            def egrad(Y):
                return jgrad(Y, D_goal, inds)

            def ehessp(Y, Z):
                return jhess(Y, Z, D_goal, inds)

            return cost, egrad, ehessp

        # Per-Y state shared across cost / egrad / ehess at the same Y.
        # ``_state(Y)`` is memoized when cache_cost=True; RTR's inner CG
        # then pays the build once per outer iteration.
        def _build_state(Y):
            S = omega * (D_goal - distance_matrix_from_pos(Y))
            S_diag = S.copy()
            np.fill_diagonal(S_diag, S_diag.diagonal() - np.sum(S_diag, axis=1))
            return S, S_diag

        _state = self._memo(_build_state)

        def cost(Y):
            S, _ = _state(Y)
            return np.linalg.norm(S) ** 2 / 2

        def egrad(Y):
            _, S_diag = _state(Y)
            return 2 * S_diag.dot(Y)

        def ehessp(Y, Z):
            _, S_diag = _state(Y)
            YZT = Y.dot(Z.T)
            YZT += YZT.T
            dSdZ = -omega * distance_matrix_from_gram(YZT)
            np.fill_diagonal(dSdZ, dSdZ.diagonal() - np.sum(dSdZ, axis=1))
            return 2 * (dSdZ.dot(Y) + S_diag.dot(Z))

        return cost, egrad, ehessp

    def create_cost_limits(self, D_goal, omega, psi_L, psi_U):
        diff = psi_L != psi_U
        inds = np.nonzero(np.triu(omega) + np.triu(diff * (psi_L > 0)) + np.triu(diff * (psi_U > 0)))
        LL = diff*(psi_L>0)
        UU = diff*(psi_U>0)

        if self.jit:
            _, _, _, lcost, lgrad, lhess = _import_costgrd()

            def cost(Y):
                return lcost(Y, D_goal, omega, psi_L, psi_U, inds)

            def egrad(Y):
                return lgrad(Y, D_goal, omega, psi_L, psi_U, inds)

            def ehess(Y, v):
                return lhess(Y, v, D_goal, omega, psi_L, psi_U, inds)

            return cost, egrad, ehess

        # Per-Y state shared across cost / egrad / ehess at the same Y.
        # The three constraint slices (A0/A1/A2) enter the gradient and
        # Hessian only through the linear adjoint operator, which commutes
        # with summation — so we fold the (3, N, N) stack into single
        # (N, N) arrays A_adj and m_total. Y_diff is the pairwise-difference
        # tensor that lets ehess express
        #   adjoint(M).dot(Y) == sum_k M[i,k] * (Y[k,:] - Y[i,:])
        # as a batched matmul.
        def _build_state(Y):
            D = distance_matrix_from_pos(Y)
            A0 = omega * (D_goal - D)
            A1 = np.maximum(psi_L - LL * D, 0)
            A2 = -np.maximum(-psi_U + UU * D, 0)
            A_sum = A0 + A1 + A2
            A_adj = A_sum.copy()
            np.fill_diagonal(
                A_adj, A_adj.diagonal() - np.sum(A_sum, axis=1),
            )
            m4 = -np.where(A1 > 0, 1, 0) * LL
            m5 = -np.where(-A2 > 0, 1, 0) * UU
            m_total = -omega + m4 + m5
            Y_diff = Y[None, :, :] - Y[:, None, :]  # (N, N, d)
            return {
                "A0": A0, "A1": A1, "A2": A2,
                "A_adj": A_adj, "m_total": m_total, "Y_diff": Y_diff,
            }

        _state = self._memo(_build_state)

        def cost(Y):
            s = _state(Y)
            return (np.linalg.norm(s["A0"])**2
                    + np.linalg.norm(s["A1"])**2
                    + np.linalg.norm(s["A2"])**2) / 2

        def egrad(Y):
            s = _state(Y)
            return 2 * s["A_adj"].dot(Y)

        def ehess(Y, Z):
            s = _state(Y)
            d_yz = distance_matrix_from_gram(Y.dot(Z.T) + Z.dot(Y.T))
            M = s["m_total"] * d_yz
            # adjoint(M).dot(Y) reformulated as a batched matmul:
            #   row i is M[i, :] @ Y_diff[i, :, :].
            adj_M_Y = np.matmul(M[:, None, :], s["Y_diff"]).squeeze(1)
            return 2 * (adj_M_Y + s["A_adj"].dot(Z))

        return cost, egrad, ehess

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
