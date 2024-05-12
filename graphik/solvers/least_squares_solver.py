import time
import sys
import timeit

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
        jac = np.zeros((num_dist,Y.size))
        diff = np.zeros(dim)
        for i in range(num_dist):
            idx, jdx = inds[0][i], inds[1][i]
            nrm = 0
            for kdx in range(dim):
                diff[kdx] = Y[idx * dim + kdx] - Y[jdx * dim + kdx]
                nrm += diff[kdx] ** 2
            for kdx in range(dim):
                jac[i, idx*dim + kdx] = 2*diff[kdx]
                jac[i, jdx*dim + kdx] = -2*diff[kdx]
        return jac

np.set_printoptions(threshold=sys.maxsize)

class LeastSquaresSolver:
    def __init__(self, graph: ProblemGraph, cost_type='dense', jit=False, *args, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.graph = graph
        self.dim = graph.dim
        self.N = graph.number_of_nodes()
        self.cost_type = cost_type

        if not hasattr(self, 'cost_type'):
            self.cost_type = 'loop'
        if not hasattr(self, 'jit'):
            self.cost_type = 'loop'

        if self.cost_type == "loop":
            self.residual_ = residual_loop_
            self.jacobian_ = jacobian_loop_
        elif self.cost_type == "sparse" or "dense":
            self.residual_ = residual_
            self.jacobian_ = jacobian_
        else:
            raise NotImplementedError(f"Cost {self.cost_type} not implemented.")

        if jit and jitted_f:
            self.residual_ = jitted_f(self.residual_)
            self.jacobian_ = jitted_f(self.jacobian_)

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

        else:
            sparse = self.cost_type == "sparse"
            res_sq_vec_batch = sum_square_op_batched(
                omega, dim, vectorized=True, flat=True, sparse=sparse
            )
            residual = lambda Y: residual_(Y, res_sq_vec_batch, d)
            diff_sq_vec_batch = 2*res_sq_vec_batch
            jacobian = lambda Y: jacobian_(Y, diff_sq_vec_batch)

        return residual, jacobian

    def solve(
        self,
        D_goal,
        omega,
        use_limits=False,
        bounds=None,
        Y_init=None,
        output_log=True,
        method='lm',
        tol=1e-4,
    ):
        """
        Implementation of solver using base Scipy unconstrained optimization algorithms.
        Currently does not support limits.
        """
        # Generate cost, gradient and hessian-vector product
        residual, jacobian = self.create_cost(D_goal, omega)

        # Generate initialization
        # if Y_init is None:
        psi_L, psi_U = self.graph.distance_bound_matrices()
        Y_init = self.generate_initialization(bounds, self.dim, omega, psi_L, psi_U)
        Y_init = np.ascontiguousarray(Y_init.flatten())
        start_time = time.time()
        sol = least_squares(
            residual,
            Y_init,
            jac=jacobian,
            method=method,
            verbose=0,
            xtol = tol
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


if __name__ == "__main__":
    from scipy.sparse import csc_array
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    from scipy.linalg import norm
    from graphik.utils.roboturdf import load_kuka, load_ur10
    from experiments.problem_generation import generate_revolute_problem

    robot, graph = load_ur10()
    G, T_goal, D_goal, X_goal = generate_revolute_problem(graph)
    omega = adjacency_matrix_from_graph(G)
    inds = np.nonzero(np.triu(omega))
    N = omega.shape[0]
    dim = robot.dim
    num_dist = len(inds[0])
    M = num_dist * dim
    idx, jdx = inds

    # Incidence matrix, implements p_i - p_j for p -> (N,dim)
    B = incidence_matrix_from_adjacency(omega)

    # Implements p_i - p_j for p -> N*dim
    B_vec = np.kron(B, np.eye(dim))

    # Implements sum((p_i - p_j)**2) as p.T @ A @ p
    res_sum_sq = B.T.dot(B)
    res_sum_sq_vec = B_vec.T @ B_vec

    # Implements d[i] = (p[i] - p[j])**2 = p @ A[i] @ p
    res_sq_vec_all = [
        np.kron(np.outer(B[i], B[i]), np.eye(dim)) for i in range(num_dist)
    ]

    # Implements batched d = (p[all_i] - p[all_j])**2
    res_sq_vec_batch = np.stack(res_sq_vec_all)
    res_sq_vec_batch_inds = np.nonzero(res_sq_vec_batch)
    res_sq_vec_batch_data = res_sq_vec_batch[res_sq_vec_batch_inds]
    res_sq_vec_batch_sparse = csc_array(
        np.ascontiguousarray(res_sq_vec_batch.swapaxes(0, 1).reshape(N * dim, -1).T)
    )

    # Implements the gradient for batched d w.r.t p
    diff_sq_vec_batch = -2 * res_sq_vec_batch
    diff_sq_vec_batch_inds = np.nonzero(diff_sq_vec_batch)
    diff_sq_vec_batch_data = diff_sq_vec_batch[diff_sq_vec_batch_inds]
    diff_sq_vec_batch_sparse = csc_array(
        np.ascontiguousarray(diff_sq_vec_batch.swapaxes(0, 1).reshape(N * dim, -1).T)
    )

    d = D_goal[inds]
    sel = np.arange(0, num_dist * dim, dim)
    Y = np.ones(N * 3)

    num_runs = 10000  # Number of times to run the timing test
    setup_code = """
from __main__ import f, Y
"""
    print("RESIDUAL TIMING")

    def f(Y):
        Y = Y.reshape(N, 3)
        return d - np.diag(np.linalg.multi_dot((B, Y, Y.T, B.T)))

    time_taken = timeit.timeit("f(Y)", setup=setup_code, number=num_runs)
    print(f"Average time per run: {time_taken / num_runs:.6f} seconds")

    def f(Y):
        return d - np.sum(B_vec.dot(Y).reshape(num_dist, dim) ** 2, axis=1)

    time_taken = timeit.timeit("f(Y)", setup=setup_code, number=num_runs)
    print(f"Average time per run: {time_taken / num_runs:.6f} seconds")

    def f(Y):
        res = np.diag(np.linalg.multi_dot((B_vec, Y[:, None], Y[None, :], B_vec.T)))
        return d - np.add.reduceat(res, sel)

    time_taken = timeit.timeit("f(Y)", setup=setup_code, number=num_runs)
    print(f"Average time per run: {time_taken / num_runs:.6f} seconds")

    def f(Y):
        return d - Y.dot(res_sq_vec_batch.dot(Y[:, None]))

    time_taken = timeit.timeit("f(Y)", setup=setup_code, number=num_runs)
    print(f"Average time per run: {time_taken / num_runs:.6f} seconds")

    print("JACOBIAN TIMING")

    diff_sq_vec_batch_flat = np.ascontiguousarray(
        diff_sq_vec_batch.swapaxes(0, 1).reshape(N * dim, -1).T
    )

    def f(Y):
        return diff_sq_vec_batch_flat.dot(Y).reshape(num_dist, dim * N)

    time_taken = timeit.timeit("f(Y)", setup=setup_code, number=num_runs)
    print(f"Average time per run: {time_taken / num_runs:.6f} seconds")

    def f(Y):
        return diff_sq_vec_batch_sparse.dot(Y).reshape(num_dist, dim * N)

    time_taken = timeit.timeit("f(Y)", setup=setup_code, number=num_runs)

    print(f"Average time per run: {time_taken / num_runs:.6f} seconds")
