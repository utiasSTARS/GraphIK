import time
import sys
import timeit

from scipy.sparse import csc_array

from graphik.utils.dgp import (
    adjacency_matrix_from_graph,
    incidence_matrix_from_adjacency,
)

import numpy as np
from graphik.utils import (
    MDS,
    linear_projection,
    gram_from_distance_matrix,
)
from graphik.graphs.graph_base import ProblemGraph
from scipy.optimize import least_squares
from graphik.utils.operators import sum_square_op_batched

try:
    from numba import njit
except ImportError:
    print("Numba not installed, JIT compilation disabled.")
    jitted_f = None
else:
    jitted_f = lambda f: njit(fastmath=True)(f)


def residual_(x, A, d):
    return A.dot(x).reshape(-1, x.size).dot(x) - d


def jacobian_(x, A):
    return A.dot(x).reshape(-1, x.size)


def residual_limits_(x, A, U, L, d, u, l):
    e0 = A.dot(x).reshape(-1, x.size).dot(x) - d
    e1 = np.maximum(U.dot(x).reshape(-1, x.size).dot(x) - u, 0)
    e2 = np.minimum(L.dot(x).reshape(-1, x.size).dot(x) - l, 0)
    return np.hstack((e0, e1, e2))


def jacobian_limits_(x, A, U, L, u, l):
    j0 = A.dot(x).reshape(-1, x.size)
    j1 = U.dot(x).reshape(-1, x.size)
    j2 = L.dot(x).reshape(-1, x.size)
    m1 = j1.dot(x) - u < 0
    m2 = j2.dot(x) - l > 0
    j1[m1] = 0
    j2[m2] = 0
    return 2 * np.vstack((j0, j1, j2))


def residual_dense_(x, A, d, dim=3):
    x = x.reshape(-1, dim)
    prod = np.einsum("i j k, k l -> i j l", A, x, optimize=True)
    res = np.einsum("i j, k j l -> k i l", x.T, prod)
    res = np.einsum("k i i -> k", res) - d  # dx3x3 -> d
    return res


def jacobian_dense_(x, A, dim=3):
    x = x.reshape(-1, dim)
    prod = np.einsum("i j k, k l -> i j l", A, x, optimize=True)
    return prod.reshape(-1, x.size)


def residual_limits_dense_(x, A, U, L, d, u, l, dim=3):
    x = x.reshape(-1, dim)

    prod = np.einsum("i j k, k l -> i j l", A, x, optimize=True)
    res = np.einsum("i j, k j l -> k i l", x.T, prod)
    e0 = np.einsum("k i i -> k", res) - d  # dx3x3 -> d

    prod = np.einsum("i j k, k l -> i j l", U, x, optimize=True)
    res = np.einsum("i j, k j l -> k i l", x.T, prod)
    e1 = np.maximum(np.einsum("k i i -> k", res) - u, 0)

    prod = np.einsum("i j k, k l -> i j l", L, x, optimize=True)
    res = np.einsum("i j, k j l -> k i l", x.T, prod)
    e2 = np.minimum(np.einsum("k i i -> k", res) - l, 0)

    return np.hstack((e0, e1, e2))


def jacobian_limits_dense_(x, A, U, L, u, l, dim=3):
    x = x.reshape(-1, dim)
    j0 = np.einsum("i j k, k l -> i j l", A, x, optimize=True).reshape(-1, x.size)

    j1 = np.einsum("i j k, k l -> i j l", U, x, optimize=True)
    res = np.einsum("i j, k j l -> k i l", x.T, j1)
    e1 = np.einsum("k i i -> k", res) - u

    j2 = np.einsum("i j k, k l -> i j l", L, x, optimize=True)
    res = np.einsum("i j, k j l -> k i l", x.T, j2)
    e2 = np.einsum("k i i -> k", res) - l

    j1 = j1.reshape(-1, x.size)
    j1[e1<0] = 0
    j2 = j2.reshape(-1, x.size)
    j2[e2>0] = 0

    return 2 * np.vstack((j0, j1, j2))


def residual_loop_(Y, D_goal, inds, dim):
    num_dist = inds[0].size
    res = np.zeros(num_dist)
    diff = np.zeros(dim)
    for i in range(num_dist):
        idx, jdx = inds[0][i], inds[1][i]
        nrm = 0
        for kdx in range(dim):
            diff[kdx] = Y[idx * dim + kdx] - Y[jdx * dim + kdx]
            nrm += diff[kdx] ** 2
        res[i] = nrm - D_goal[idx, jdx]
    return res


def jacobian_loop_(Y, inds, dim):
    num_dist = inds[0].size
    jac = np.zeros((num_dist, Y.size))
    diff = np.zeros(dim)
    for i in range(num_dist):
        idx, jdx = inds[0][i], inds[1][i]
        nrm = 0
        for kdx in range(dim):
            diff[kdx] = Y[idx * dim + kdx] - Y[jdx * dim + kdx]
            nrm += diff[kdx] ** 2
        for kdx in range(dim):
            jac[i, idx * dim + kdx] = 2 * diff[kdx]
            jac[i, jdx * dim + kdx] = -2 * diff[kdx]
    return jac


np.set_printoptions(threshold=sys.maxsize)


class LeastSquaresSolver:
    def __init__(
        self, graph: ProblemGraph, cost_type="sparse", jit=False, *args, **kwargs
    ):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.graph = graph
        self.dim = graph.dim
        self.N = graph.number_of_nodes()
        self.cost_type = cost_type
        self.jit = jit

        if self.cost_type == "loop":
            self.residual_ = residual_loop_
            self.jacobian_ = jacobian_loop_
            self.residual_limits_ = None
            self.jacobian_limits_ = None
        elif self.cost_type == "sparse":
            self.residual_ = residual_
            self.jacobian_ = jacobian_
            self.residual_limits_ = residual_limits_
            self.jacobian_limits_ = jacobian_limits_
        elif self.cost_type == "dense":
            self.residual_ = residual_dense_
            self.jacobian_ = jacobian_dense_
            self.residual_limits_ = residual_limits_dense_
            self.jacobian_limits_ = jacobian_limits_dense_
        else:
            raise NotImplementedError(f"Cost {self.cost_type} not implemented.")

        if jitted_f:
            if jit and self.cost_type != "dense":
                self.residual_ = jitted_f(self.residual_)
                self.jacobian_ = jitted_f(self.jacobian_)
                if self.residual_limits_:
                    self.residual_limits_ = jitted_f(self.residual_limits_)
                if self.jacobian_limits_:
                    self.jacobian_limits_ = jitted_f(self.jacobian_limits_)

    def generate_initialization(self, bounds, dim, omega, psi_L, psi_U):
        # Generates a random EDM within the set bounds
        lb = np.sqrt(bounds[0])
        ub = np.sqrt(bounds[1])
        D_rand = (lb + 0.9 * (ub - lb)) ** 2
        X_rand = MDS(gram_from_distance_matrix(D_rand), eps=1e-8)
        Y_rand = linear_projection(X_rand, omega, dim)
        return Y_rand

    def create_cost(self, D_goal, omega):
        inds = np.nonzero(np.triu(omega))
        N = self.N
        num_dist = len(inds[0])
        M = D_goal.shape[0] * self.dim
        dim = self.dim
        d = D_goal[inds]

        if self.cost_type == "loop":
            residual = lambda x: self.residual_(x, D_goal, inds, dim)
            jacobian = lambda x: self.jacobian_(x, inds, dim)
        elif self.cost_type == "sparse":
            use_sparse = not self.jit
            res_sq_vec_batch = sum_square_op_batched(
                omega, dim, vectorized=True, flat=True, sparse=use_sparse
            )
            residual = lambda Y: self.residual_(Y, res_sq_vec_batch, d)
            diff_sq_vec_batch = 2 * res_sq_vec_batch
            jacobian = lambda Y: self.jacobian_(Y, diff_sq_vec_batch)
        elif self.cost_type == "dense":
            res_sq_batch = sum_square_op_batched(
                omega, dim, vectorized=False, flat=False, sparse=False
            )  # (d N*dim N*dim)
            res_sq_batch = np.ascontiguousarray(res_sq_batch)
            residual = lambda Y: self.residual_(Y, res_sq_batch, d)
            jacobian = lambda Y: self.jacobian_(Y, res_sq_vec_batch)

        return residual, jacobian

    def create_cost_limits(self, D_goal, omega, psi_L, psi_U):
        diff = psi_L != psi_U
        LL = diff * (psi_L > 0) * (~omega.astype(bool))
        UU = diff * (psi_U > 0) * (~omega.astype(bool))
        dist = D_goal[np.nonzero(np.triu(omega))]
        dist_L = psi_L[np.nonzero(np.triu(LL))]
        dist_U = psi_U[np.nonzero(np.triu(UU))]

        use_sparse = not self.jit
        if self.cost_type == "sparse":
            L = sum_square_op_batched(
                LL, self.dim, vectorized=True, flat=True, sparse=use_sparse
            )
            U = sum_square_op_batched(
                UU, self.dim, vectorized=True, flat=True, sparse=use_sparse
            )
            A = sum_square_op_batched(
                omega, self.dim, vectorized=True, flat=True, sparse=use_sparse
            )

        elif self.cost_type == "dense":
            L = sum_square_op_batched(
                LL, self.dim, vectorized=False, flat=False, sparse=False
            )  # (d N N)
            L = np.ascontiguousarray(L)
            U = sum_square_op_batched(
                UU, self.dim, vectorized=False, flat=False, sparse=False
            )  # (d N N)
            U = np.ascontiguousarray(U)
            A = sum_square_op_batched(
                omega, self.dim, vectorized=False, flat=False, sparse=False
            )  # (d N N)
            A = np.ascontiguousarray(A)

        residual = lambda Y: self.residual_limits_(Y, A, U, L, dist, dist_U, dist_L)
        if use_sparse:
            jacobian = lambda Y: csc_array(
                self.jacobian_limits_(Y, A, U, L, dist_U, dist_L)
            )
        else:
            jacobian = lambda Y: self.jacobian_limits_(Y, A, U, L, dist_U, dist_L)
        return residual, jacobian

    def solve(
        self,
        D_goal,
        omega,
        use_limits=False,
        bounds=None,
        Y_init=None,
        output_log=True,
        method="trf",
        tol=1e-4,
    ):
        """
        Implementation of solver using base Scipy unconstrained optimization algorithms.
        Currently does not support limits.
        """
        psi_L, psi_U = self.graph.distance_bound_matrices()
        # Generate cost, gradient and hessian-vector product
        if use_limits:
            residual, jacobian = self.create_cost_limits(D_goal, omega, psi_L, psi_U)
        else:
            residual, jacobian = self.create_cost(D_goal, omega)

        # Generate initialization
        Y_init = self.generate_initialization(bounds, self.dim, omega, psi_L, psi_U)
        Y_init = np.ascontiguousarray(Y_init.flatten())
        start_time = time.time()
        sol = least_squares(
            residual, Y_init, jac=jacobian, method=method, verbose=0, xtol=tol
        )
        end_time = time.time()

        # Solve problem
        if output_log:
            optlog = {
                "x": sol.x.reshape(-1, self.dim),
                "time": end_time - start_time,
                "iterations": sol.nfev,
                "f(x)": sum(sol.fun),
            }
            return optlog
        else:
            return sol.x
