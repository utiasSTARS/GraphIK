#!/usr/bin/env python3
import time
import numpy as np
import warnings

warnings.filterwarnings("ignore")

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
from graphik.utils.dgp import pos_from_graph

from graphik.utils.operators import sum_square_op_batched

try:
    from numba import njit
except ImportError:
    print("Numba not installed, JIT compilation disabled.")
    jitted_f = None
else:
    jitted_f = lambda f: njit(fastmath=True)(f)


def adjoint(X: np.ndarray) -> np.ndarray:
    D = np.zeros_like(X)
    np.einsum("ijj->ij", D)[...] = np.sum(X, axis=-1)
    return X - D


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
    # FIXME not correct
    n = A.shape[1]
    prod = A.dot(x).reshape(d.size, -1)  # (d*N*dim) - > (d N*dim)
    res = prod.dot(x) - d
    H1 = 4 * A.reshape(d.size, -1).T.dot(res)  # (d N*dim*N*dim).T (d)
    H2 = prod.T.dot(prod)
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
    prod_z = A.dot(z).reshape(d.size, -1)  # A
    sqd = prod.dot(x)
    e0 = sqd - d
    hessvec = 8 * (prod_z.dot(x)).dot(prod) + 4 * e0.dot(prod_z)

    prod = U.dot(x).reshape(u.size, -1)
    prod_z = U.dot(z).reshape(u.size, -1)  # A
    sqd = prod.dot(x)
    e1 = np.maximum(sqd - u, 0)
    hessvec += 8 * (prod_z.dot(x)).dot(prod) + 4 * e1.dot(prod_z)

    prod = L.dot(x).reshape(l.size, -1)
    prod_z = L.dot(z).reshape(l.size, -1)  # A
    sqd = prod.dot(x)
    e2 = -np.minimum(sqd - l, 0)
    hessvec += 8 * (prod_z.dot(x)).dot(prod) + 4 * e2.dot(prod_z)
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
    # prod = rearrange(res_sq_batch.dot(Y), '(N d) dim -> N d dim', N=N, d=num_dist, dim=dim) # (pxd)xp * pxn -> (pxd)x3
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

    def position_constraints(self, x):
        ub_ = np.ones((self.N, self.dim)) * np.inf
        lb_ = -np.ones((self.N, self.dim)) * np.inf
        for idx, (node, data) in enumerate(self.graph.nodes(data=True)):
            if POS in data or END_EFFECTOR in data[TYPE]:
                ub_[idx] = x[idx]
                lb_[idx] = x[idx]
        return Bounds(lb=lb_.flatten(), ub=ub_.flatten())

    def solve(
        self,
        D_goal,
        omega,
        use_limits=False,
        bounds=None,
        Y_init=None,
        output_log=True,
        method="BFGS",
        tol=1e-8,
    ):
        """
        Implementation of solver using base Scipy unconstrained optimization algorithms.
        Currently does not support limits.
        """
        psi_L, psi_U = self.graph.distance_bound_matrices()

        # Generate cost, gradient and hessian-vector product
        if use_limits:
            # psi_L, psi_U = bounds[0]**2, bounds[1]**2
            cost_and_grad, hessp, hess = self.create_cost_limits(
                D_goal, omega, psi_L, psi_U
            )
        else:
            cost_and_grad, hessp, hess = self.create_cost(D_goal, omega)

        bnds, options = None, None
        if method in ["L-BFGS-B"]:
            bnds = self.position_constraints(Y_init)
            options = {"xtol": 1e-16, "ftol": 1e-16, "gtol": 1e-16, "iprint": -1}

        # Generate initialization
        Y_init = self.generate_initialization(bounds, self.dim, omega, psi_L, psi_U)
        Y_init = Y_init.flatten()

        start_time = time.time()
        sol = minimize(
            cost_and_grad,
            Y_init,
            jac=True,
            hessp=hessp,
            hess=hess,
            method=method,
            tol=tol,
            bounds=bnds,
            options=options,
        )
        end_time = time.time()

        # Solve problem
        if output_log:
            optlog = {
                "x": sol.x.reshape(omega.shape[0], 3),
                "time": end_time - start_time,
                "iterations": sol.nit,
                "f(x)": sol.fun,
            }
            return optlog
        else:
            return sol.x


if __name__ == "__main__":
    from graphik.utils.roboturdf import load_kuka, load_ur10
    from experiments.problem_generation import generate_revolute_problem

    robot, graph = load_ur10()
    G, T_goal, D_goal, X_goal = generate_revolute_problem(graph)
    omega = adjacency_matrix_from_graph(G)
    inds = np.nonzero(np.triu(omega))
    n_points = omega.shape[0]
    dim = robot.dim

    cost_type = "sparse"
    solver = NonlinearSolver(graph, cost_type=cost_type)

    def numerical_gradient(Y, f, eps=1e-6):
        num_grad = np.zeros(Y.shape)
        perturb = np.zeros(Y.shape)
        for i in range(len(Y)):
            perturb[i] = eps
            loss1, _ = f(Y - perturb)
            loss2, _ = f(Y + perturb)
            num_grad[i] = (loss2 - loss1) / (2 * eps)
            perturb[i] = 0
        return num_grad

    def numerical_hessian(Y, f, eps=1e-5):
        H_approx = np.zeros((len(Y), len(Y)))
        _, grad_base = f(Y)

        for i in range(len(Y)):
            Y[i] += eps
            _, grad_plus = f(Y)
            Y[i] -= eps
            H_approx[:, i] = (grad_plus - grad_base) / eps

        return H_approx.ravel()

    cost_and_grad, hessp, hess = solver.create_cost(D_goal, omega)
    cost_and_grad_func = lambda Y: cost_and_grad(Y.flatten())

    for _ in range(10):
        # Test your gradient calculation
        Y_test = np.random.randn(n_points, dim)  # example Y vector
        _, grad_analytical = cost_and_grad_func(Y_test)
        grad_numerical = numerical_gradient(Y_test.flatten(), cost_and_grad_func)

        tolerance = 1e-6
        if np.allclose(grad_analytical, grad_numerical, atol=tolerance):
            print(
                "The analytical and numerical gradients are close within a tolerance of",
                tolerance,
            )
        else:
            print(
                "There are discrepancies between the analytical and numerical gradients beyond the tolerance of",
                tolerance,
            )
            print("-----------")

        if cost_type in ["loop", "sparse"]:
            # Test the Hessian approximation
            Y_test = np.random.randn(n_points, dim)  # example Y vector
            H_analytical = hess(Y_test.flatten()).flatten()
            H_numerical = numerical_hessian(Y_test.flatten(), cost_and_grad_func)

            tolerance = 1e-3
            if np.allclose(H_analytical, H_numerical, atol=tolerance):
                print(
                    "The analytical and numerical Hessians are close within a tolerance of",
                    tolerance,
                )
            else:
                print(
                    "There are discrepancies between the analytical and numerical Hessians beyond the tolerance of",
                    tolerance,
                )
                print("Analytical Hessian:\n", H_analytical)
                print("Numerical Hessian:\n", H_numerical)

        # res_batch_left_dense = res_left_op_batched(omega, dim, vectorized=True, sparse=False) # (d dim N dim)
        # res_batch_left = res_batch_left_dense.reshape(num_dist*dim, N*dim) # (d*dim N*dim)
        # res_batch_left = csc_array(np.ascontiguousarray(res_batch_left))
        # def cost_and_grad_sparse(Y):
        #     prod = res_batch_left.dot(Y).reshape(num_dist, dim) # Y1 - Y2 in (d dim)
        #     res = np.sum(prod*prod, axis=1) - dist # (Y1 - Y2)**2 - dist (d)
        #     cost = res.dot(res)
        #     grad = res_sq_vec_batch.dot(Y).reshape(num_dist,N*dim) # (res_sq_batch + res_sq_batch.T) = 2*res_sq_batch
        #     grad = 4*res.dot(grad)
        #     return cost, grad
