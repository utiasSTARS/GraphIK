#!/usr/bin/env python3
import time
import numpy as np

from graphik.utils import (
    distance_matrix_from_gram,
    distance_matrix_from_pos,
    MDS,
    linear_projection,
    gram_from_distance_matrix,
    adjacency_matrix_from_graph,
)
from graphik.graphs.graph_base import ProblemGraph
from scipy.optimize import Bounds, minimize
from graphik.utils.constants import END_EFFECTOR, POS, TYPE
from graphik.utils.operators import sum_square_op_batched

# scipy.optimize.minimize methods that consume hess / hessp. Passing the
# kwargs to other methods just provokes a UserWarning per call.
_HESS_METHODS = frozenset({
    "Newton-CG", "dogleg", "trust-ncg", "trust-exact",
    "trust-krylov", "trust-constr",
})
_HESSP_METHODS = frozenset({
    "Newton-CG", "trust-ncg", "trust-krylov", "trust-constr",
})

try:
    from numba import njit
except ImportError:
    print("Numba not installed, JIT compilation disabled.")
    jitted_f = None
else:
    jitted_f = lambda f: njit(fastmath=True)(f)


def cost_and_grad_sparse_(x, A, d):
    prod = A.dot(x).reshape(d.size, -1)
    res = prod.dot(x) - d
    cost = res.dot(res)
    grad = 4 * res.dot(prod)
    return cost, grad


def hessp_sparse_(x, z, A, d):
    prod = A.dot(x).reshape(d.size, -1)
    prod_z = A.dot(z).reshape(d.size, -1)  # A
    sqd = prod.dot(x)
    res = sqd - d
    hessvec = 8 * (prod_z.dot(x)).dot(prod) + 4 * res.dot(prod_z)
    return hessvec


def hess_sparse_(x, A, d):
    n = A.shape[1]
    prod = A.dot(x).reshape(d.size, -1)
    res = prod.dot(x) - d
    H1 = 4 * A.reshape(d.size, -1).T.dot(res)
    H2 = 8 * prod.T.dot(prod)
    return H1.reshape(n, n) + H2.reshape(n, n)


def cost_and_grad_limits_sparse_(x, A, U, L, d, u, l):
    prod = A.dot(x).reshape(d.size, -1)
    res = prod.dot(x)
    e0 = res - d
    cost = e0.dot(e0)
    grad = e0.dot(prod)

    prod = U.dot(x).reshape(u.size, -1)
    res = prod.dot(x)
    e1 = np.maximum(res - u, 0)
    cost += e1.dot(e1)
    grad += e1.dot(prod)

    prod = L.dot(x).reshape(l.size, -1)
    res = prod.dot(x)
    e2 = np.minimum(res - l, 0)
    cost += e2.dot(e2)
    grad += e2.dot(prod)
    return cost, 4 * grad


def hessp_limits_sparse_(x, z, A, U, L, d, u, l):
    prod = A.dot(x).reshape(d.size, -1)
    prod_z = A.dot(z).reshape(d.size, -1)
    sqd = prod.dot(x)
    e0 = sqd - d
    hessvec = 8 * (prod_z.dot(x)).dot(prod) + 4 * e0.dot(prod_z)

    prod = U.dot(x).reshape(u.size, -1)
    prod_z = U.dot(z).reshape(u.size, -1)
    sqd = prod.dot(x)
    e1 = np.maximum(sqd - u, 0)
    active_u = (e1 > 0).astype(np.float64)
    hessvec += 8 * (active_u * prod_z.dot(x)).dot(prod) + 4 * e1.dot(prod_z)

    prod = L.dot(x).reshape(l.size, -1)
    prod_z = L.dot(z).reshape(l.size, -1)
    sqd = prod.dot(x)
    e2 = np.minimum(sqd - l, 0)
    active_l = (e2 < 0).astype(np.float64)
    hessvec += 8 * (active_l * prod_z.dot(x)).dot(prod) + 4 * e2.dot(prod_z)
    return hessvec


def cost_and_grad_dense_(Y, D_goal, omega):
    N = omega.shape[0]
    Y = Y.reshape(N, -1)
    D = distance_matrix_from_pos(Y)
    S = omega * (D - D_goal)
    f = np.linalg.norm(S) ** 2
    np.fill_diagonal(S, S.diagonal() - np.sum(S, axis=1))
    dfdY = -8 * S.dot(Y)
    return f, dfdY.ravel()


def hessp_dense_(Y, Z, D_goal, omega):
    N = omega.shape[0]
    Y = Y.reshape(N, -1)
    Z = Z.reshape(N, -1)
    D = distance_matrix_from_pos(Y)
    YZT = Y.dot(Z.T)
    YZT += YZT.T
    dSdZ = omega * distance_matrix_from_gram(YZT)
    np.fill_diagonal(dSdZ, dSdZ.diagonal() - np.sum(dSdZ, axis=1))
    S = omega * (D - D_goal)
    np.fill_diagonal(S, S.diagonal() - np.sum(S, axis=1))
    H = dSdZ.dot(Y) + S.dot(Z)
    return -8 * H.ravel()


def cost_and_grad_dense_einsum_(x, A, d, dim=3):
    # einsum-only solution with dense matrix
    x = x.reshape(-1, dim)
    prod = np.einsum("i j k, k l -> i j l", A, x, optimize=True)
    res = np.einsum("i j, k j l -> k i l", x.T, prod)
    res = np.einsum("k i i -> k", res) - d  # dx3x3 -> d
    cost = res.dot(res)
    grad = 4 * np.einsum("i, i j k -> j k", res, prod).flatten()
    return cost, grad


def cost_and_grad_loop_(Y, D_goal, inds, dim):
    cost = 0
    grad = np.zeros(Y.shape)
    diff = np.zeros(dim)
    for idx, jdx in zip(*inds):
        nrm = 0
        for kdx in range(dim):
            diff[kdx] = Y[idx * dim + kdx] - Y[jdx * dim + kdx]
            nrm += diff[kdx] ** 2
        for kdx in range(dim):
            update = -4 * (D_goal[idx, jdx] - nrm) * diff[kdx]
            grad[idx * dim + kdx] += update
            grad[jdx * dim + kdx] -= update
        cost += (D_goal[idx, jdx] - nrm) ** 2
    return cost, grad


def hessp_loop_(Y, Z, D_goal, inds, dim):
    hess = np.zeros(Y.shape)
    diff_Y = np.zeros(dim)
    diff_Z = np.zeros(dim)
    for idx, jdx in zip(*inds):
        nrm = 0
        sc = 0
        for kdx in range(dim):
            diff_Y[kdx] = Y[idx * dim + kdx] - Y[jdx * dim + kdx]
            diff_Z[kdx] = Z[idx * dim + kdx] - Z[jdx * dim + kdx]
            sc += diff_Y[kdx] * diff_Z[kdx]
            nrm += diff_Y[kdx] ** 2
        for kdx in range(dim):
            update = 4 * (2 * sc * diff_Y[kdx] + (nrm - D_goal[idx, jdx]) * diff_Z[kdx])
            hess[idx * dim + kdx] += update
            hess[jdx * dim + kdx] -= update
    return hess


def hess_loop_(Y, D_goal, inds, dim):
    n = Y.shape[0] // dim  # Total number of points assuming Y is flat
    H = np.zeros((n * dim, n * dim))  # The full Hessian matrix

    for idx, jdx in zip(*inds):
        diff = np.zeros(dim)
        nrm = 0
        for kdx in range(dim):
            diff[kdx] = Y[idx * dim + kdx] - Y[jdx * dim + kdx]
            nrm += diff[kdx] ** 2

        # Compute the Hessian block for the (idx, jdx) pair
        for kdx in range(dim):
            for ldx in range(dim):
                H_entry = 4 * (
                    2 * diff[kdx] * diff[ldx] + (nrm - D_goal[idx, jdx]) * (kdx == ldx)
                )
                H[idx * dim + kdx, idx * dim + ldx] += H_entry
                H[jdx * dim + kdx, jdx * dim + ldx] += H_entry
                H[idx * dim + kdx, jdx * dim + ldx] -= H_entry
                H[jdx * dim + kdx, idx * dim + ldx] -= H_entry

    return H


class NonlinearSolver:
    def __init__(
        self, graph: ProblemGraph, cost_type="dense", jit=False, *args, **kwargs
    ):
        """
        Implementation of distance-based solution using the standard scipy nonlinear solver interface (minimize).
        The cost_type parameter chooses specific implementation of cost function:
        - 'dense' -> closest to the original paper [Maric & Giamou, 2021.]
        - 'sparse' -> lighter formulation that uses scipy sparse matrices (fast)
        - 'loop' -> implementation based on for-loops, primarily exists for JIT compilation (fastest with JIT, else slowest)
        - 'einsum' -> purely einsum-based implementation
        The JIT parameter performs JIT compilation when set to True, works with 'loop' and 'sparse'.
        """

        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.graph = graph
        self.dim = graph.dim
        self.N = graph.number_of_nodes()
        self.cost_type = cost_type
        self.jit = jit

        if self.cost_type == "loop":
            self.cost_and_grad_ = cost_and_grad_loop_
            self.hessp_ = hessp_loop_
            self.hess_ = hess_loop_
            self.cost_and_grad_limits_ = None
            self.hessp_limits_ = None
            self.hess_limits_ = None
        elif self.cost_type == "sparse":
            self.cost_and_grad_ = cost_and_grad_sparse_
            self.hessp_ = hessp_sparse_
            self.hess_ = hess_sparse_
            self.cost_and_grad_limits_ = cost_and_grad_limits_sparse_
            self.hessp_limits_ = hessp_limits_sparse_
            self.hess_limits_ = None
        elif self.cost_type == "dense":
            self.cost_and_grad_ = cost_and_grad_dense_
            self.hessp_ = hessp_dense_
            self.hess_ = None
            self.cost_and_grad_limits_ = None
            self.hessp_limits_ = None
            self.hess_limits_ = None
        elif self.cost_type == "einsum":
            self.cost_and_grad_ = cost_and_grad_dense_einsum_
            self.hessp_ = None
            self.hess_ = None
            self.cost_and_grad_limits_ = None
            self.hessp_limits_ = None
            self.hess_limits_ = None
        else:
            raise NotImplementedError(f"Cost {cost_type} not implemented.")

        if jit and jitted_f:
            if self.cost_and_grad_:
                self.cost_and_grad_ = jitted_f(self.cost_and_grad_)
            if self.cost_and_grad_limits_:
                self.cost_and_grad_limits_ = jitted_f(self.cost_and_grad_limits_)
            if self.hessp_:
                self.hessp_ = jitted_f(self.hessp_)
            if self.hessp_limits_:
                self.hessp_limits_ = jitted_f(self.hessp_limits_)
            if self.hess_:
                self.hess_ = jitted_f(self.hess_)
            if self.hess_limits_:
                self.hess_limits_ = jitted_f(self.hess_limits_)

    def generate_initialization(self, bounds, dim, omega):
        """Sample an EDM within the supplied (lb, ub) bounds, then MDS + project."""
        lb = np.sqrt(bounds[0])
        ub = np.sqrt(bounds[1])
        D_rand = (lb + 0.9 * (ub - lb)) ** 2
        X_rand = MDS(gram_from_distance_matrix(D_rand), eps=1e-8)
        return linear_projection(X_rand, omega, dim)

    def create_cost(self, D_goal, omega):
        inds = np.nonzero(np.triu(omega))
        N = omega.shape[0]
        dist = D_goal[inds]
        num_dist = len(inds[0])
        dim = self.dim

        if self.cost_type == "loop":
            cost_and_grad = lambda Y: self.cost_and_grad_(Y, D_goal, inds, dim)
            hessp = lambda Y, Z: self.hessp_(Y, Z, D_goal, inds, dim)
            hess = lambda Y: self.hess_(Y, D_goal, inds, dim)
        elif self.cost_type == "sparse":
            res_sq_vec_batch = sum_square_op_batched(
                omega, dim, vectorized=True, flat=True, sparse=not self.jit
            )
            cost_and_grad = lambda Y: self.cost_and_grad_(Y, res_sq_vec_batch, dist)
            hessp = lambda Y, Z: self.hessp_(Y, Z, res_sq_vec_batch, dist)
            hess = lambda Y: self.hess_(Y, res_sq_vec_batch, dist)

        elif self.cost_type == "einsum":
            res_sq_batch = sum_square_op_batched(
                omega, dim, vectorized=False, flat=False, sparse=False
            )  # (d N*dim N*dim)
            res_sq_batch = np.ascontiguousarray(res_sq_batch)
            cost_and_grad = lambda Y: self.cost_and_grad_(
                Y, res_sq_batch, dist, dim=dim
            )
            hessp = None
            hess = None
        else:
            cost_and_grad = lambda Y: self.cost_and_grad_(Y, D_goal, omega)
            hessp = lambda Y, Z: self.hessp_(Y, Z, D_goal, omega)
            hess = None

        return cost_and_grad, hessp, hess

    def create_cost_limits(self, D_goal, omega, psi_L, psi_U):
        diff = psi_L != psi_U
        LL = diff * (psi_L > 0) * (~omega.astype(bool))
        UU = diff * (psi_U > 0) * (~omega.astype(bool))

        L = sum_square_op_batched(
            LL, self.dim, vectorized=True, sparse=not self.jit, flat=True
        )
        U = sum_square_op_batched(
            UU, self.dim, vectorized=True, sparse=not self.jit, flat=True
        )
        A = sum_square_op_batched(
            omega, self.dim, vectorized=True, sparse=not self.jit, flat=True
        )

        dist = D_goal[np.nonzero(np.triu(omega))]
        dist_L = psi_L[np.nonzero(np.triu(LL))]
        dist_U = psi_U[np.nonzero(np.triu(UU))]

        cost_and_grad = lambda Y: self.cost_and_grad_limits_(
            Y, A, U, L, dist, dist_U, dist_L
        )
        hessp = lambda Y, Z: self.hessp_limits_(Y, Z, A, U, L, dist, dist_U, dist_L)
        return cost_and_grad, hessp, None

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
        """
        Implementation of solver using base Scipy unconstrained optimization algorithms.
        """
        if use_limits:
            if self.cost_and_grad_limits_ is None:
                raise NotImplementedError(
                    f"cost_type={self.cost_type!r} does not support use_limits=True"
                )
            psi_L, psi_U = self.graph.distance_bound_matrices()
            cost_and_grad, hessp, hess = self.create_cost_limits(
                D_goal, omega, psi_L, psi_U
            )
        else:
            cost_and_grad, hessp, hess = self.create_cost(D_goal, omega)

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
        if hess is not None and method in _HESS_METHODS:
            minimize_kwargs["hess"] = hess
        if hessp is not None and method in _HESSP_METHODS:
            minimize_kwargs["hessp"] = hessp

        start_time = time.time()
        sol = minimize(cost_and_grad, Yi, **minimize_kwargs)
        end_time = time.time()

        if output_log:
            optlog = {
                "x": sol.x.reshape(omega.shape[0], self.dim),
                "time": end_time - start_time,
                "iterations": sol.nit,
                "f(x)": sol.fun,
            }
            return optlog
        else:
            return sol.x

