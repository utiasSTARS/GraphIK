#!/usr/bin/env python3
import pymanopt

from numba import njit
import numpy as np
from pymanopt import tools
from graphik.utils.manifolds.fixed_rank_psd_sym import PSDFixedRank
from graphik.utils.dgp import (
    pos_from_graph,
    adjacency_matrix_from_graph,
    distance_matrix_from_gram,
    distance_matrix_from_pos,
    MDS,
    linear_projection,
    gram_from_distance_matrix,
)
from pymanopt.solvers import ConjugateGradient
from graphik.solvers.trust_region import TrustRegions
from graphik.graphs.graph_base import RobotGraph

BetaTypes = tools.make_enum(
    "BetaTypes", "FletcherReeves PolakRibiere HestenesStiefel HagerZhang".split()
)


# @njit(cache=True, fastmath=True)
def add_to_diagonal_fast(X: np.ndarray):
    # num_atoms = X.shape[0]
    # for i in range(num_atoms):
    #     rowsum = 0
    #     for j in range(num_atoms):
    #         rowsum += X[i, j]
    #     X[i, i] += -rowsum
    # return X
    # Non-numba
    X.ravel()[:: X.shape[1] + 1] += -np.sum(X, axis=0)
    return X


@njit(cache=True, fastmath=True)
def distmat(Y: np.ndarray, F: np.ndarray = None):
    num_atoms = Y.shape[0]
    dim = Y.shape[1]
    locs = Y
    dmat = np.zeros((num_atoms, num_atoms))
    if F is None:
        F = np.ones((num_atoms, num_atoms))
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            if F[i, j]:
                d = 0
                for k in range(dim):
                    d += (locs[i, k] - locs[j, k]) ** 2
                dmat[i, j] = d
                dmat[j, i] = d
    return dmat

    # Non-numba
    # X = Y.dot(Y.T)
    # return (np.diagonal(X)[:, np.newaxis] + np.diagonal(X)) - 2 * X


@njit(cache=True, fastmath=True)
def distmat_ind(Y: np.ndarray, inds):
    num_atoms = Y.shape[0]
    dim = Y.shape[1]
    locs = Y
    dmat = np.zeros((num_atoms, num_atoms))
    for (i, j) in zip(*inds):
        d = 0
        for k in range(dim):
            d += (locs[i, k] - locs[j, k]) ** 2
        dmat[i, j] = d
    return dmat


@njit(cache=True, fastmath=True)
def distmat_gram(X: np.ndarray):
    num_atoms = X.shape[0]
    dmat = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            d = X[i, i] + X[j, j] - 2 * X[i, j]
            dmat[i, j] = d
            dmat[j, i] = d
    return dmat
    # Non-numba
    # return (np.diagonal(X)[:, np.newaxis] + np.diagonal(X)) - 2 * X


@njit(cache=True, fastmath=True)
def frobenius_norm_sq(X: np.ndarray):
    num_atoms = X.shape[0]
    nrm = 0
    for i in range(num_atoms):
        for j in range(num_atoms):
            nrm += X[i, j] ** 2
    return nrm


class RiemannianSolver:
    def __init__(self, graph: RobotGraph, params={}):

        self.params = params
        self.graph = graph
        self.dim = graph.dim
        self.N = graph.n_nodes

        solver_type = params.get("solver", "TrustRegions")

        if solver_type == "TrustRegions":
            self.solver = TrustRegions(
                mingradnorm=params.get("mingradnorm", 0.5*1e-9),
                logverbosity=params.get("logverbosity", 0),
                maxiter=params.get("maxiter", 3000),
                theta=params.get("theta", 1.0),
                kappa=params.get("kappa", 0.1),
            )
        elif solver_type == "ConjugateGradient":
            self.solver = ConjugateGradient(
                mingradnorm=params.get("mingradnorm",1e-9),
                logverbosity=params.get("logverbosity", 0),
                maxiter=params.get("maxiter", 10e4),
                minstepsize=params.get("minstepsize", 1e-10),
                orth_value=params.get("orth_value", 10e10),
                beta_type=params.get("beta_type", BetaTypes[3]),
            )
        else:
            raise (
                ValueError,
                "params[\"solver\"] must be one of 'ConjugateGradient', 'TrustRegions'",
            )

    @staticmethod
    @njit(cache=True, fastmath=True)
    def jcost(Y, D_goal, inds):
        cost = 0
        dim = Y.shape[1]
        for (idx, jdx) in zip(*inds):
            nrm = 0
            for kdx in range(dim):
                nrm += (Y[idx, kdx] - Y[jdx, kdx]) ** 2
            cost += (D_goal[idx, jdx] - nrm) ** 2
        return 0.5 * cost

    @staticmethod
    @njit(cache=True, fastmath=True)
    def jgrad(Y, D_goal, inds):
        num_el = Y.shape[0]
        dim = Y.shape[1]
        grad = np.zeros((num_el, dim))
        for (idx, jdx) in zip(*inds):
            nrm = 0
            for kdx in range(dim):
                nrm += (Y[idx, kdx] - Y[jdx, kdx]) ** 2
            for kdx in range(dim):
                grad[idx, kdx] += (
                    -4 * (D_goal[idx, jdx] - nrm) * (Y[idx, kdx] - Y[jdx, kdx])
                )
        return 0.5 * grad

    @staticmethod
    @njit(cache=True, fastmath=True)
    def jhess(Y, w, D_goal, inds):
        num_el = Y.shape[0]
        dim = Y.shape[1]
        hess = np.zeros((num_el, dim))
        for (idx, jdx) in zip(*inds):
            nrm = 0
            sc = 0
            for kdx in range(dim):
                sc += (Y[idx, kdx] - Y[jdx, kdx]) * (w[idx, kdx] - w[jdx, kdx])
                nrm += (Y[idx, kdx] - Y[jdx, kdx]) ** 2
            for kdx in range(dim):
                hess[idx, kdx] += 4 * (
                    2 * sc * (Y[idx, kdx] - Y[jdx, kdx])
                    + (nrm - D_goal[idx, jdx]) * (w[idx, kdx] - w[jdx, kdx])
                )
        return 0.5 * hess

    @staticmethod
    @njit(cache=True, fastmath=True)
    def lcost(Y, D_goal, omega, psi_L, psi_U):
        cost = 0
        num_el = Y.shape[0]
        dim = Y.shape[1]
        for idx in range(num_el):
            for jdx in range(idx + 1, num_el):
                if omega[idx, jdx]:
                    nrm = 0
                    for kdx in range(dim):
                        nrm += (Y[idx, kdx] - Y[jdx, kdx]) ** 2
                    cost += 2 * (D_goal[idx, jdx] - nrm) ** 2
                if psi_L[idx, jdx]:
                    nrm = 0
                    for kdx in range(dim):
                        nrm += (Y[idx, kdx] - Y[jdx, kdx]) ** 2
                    cost += 2 * max((psi_L[idx, jdx] - nrm), 0) ** 2
                if psi_U[idx, jdx]:
                    nrm = 0
                    for kdx in range(dim):
                        nrm += (Y[idx, kdx] - Y[jdx, kdx]) ** 2
                    cost += 2 * max((-psi_U[idx, jdx] + nrm), 0) ** 2

        return 0.5 * cost

    # def lcost(self, Y, D_goal, F, Bl, L):
    #     D = distmat(Y)
    #     E1 = F * (D_goal - D)
    #     E2 = np.maximum(Bl - L * D, 0)
    #     return frobenius_norm_sq(E1) + frobenius_norm_sq(E2)

    @staticmethod
    @njit(cache=True, fastmath=True)
    def lgrad(Y, D_goal, omega, psi_L, psi_U):
        num_el = Y.shape[0]
        dim = Y.shape[1]
        grad = np.zeros((num_el, dim))
        D = distmat(Y, omega + psi_L + psi_U)
        for idx in range(num_el):
            for jdx in range(num_el):
                if omega[idx, jdx] and idx != jdx:
                    for kdx in range(dim):
                        grad[idx, kdx] += (
                            4
                            * (D[idx, jdx] - D_goal[idx, jdx])
                            * (Y[idx, kdx] - Y[jdx, kdx])
                        )
                if psi_L[idx, jdx] and idx != jdx:
                    if max(psi_L[idx, jdx] - D[idx, jdx], 0) > 0:
                        for kdx in range(dim):
                            grad[idx, kdx] += (
                                4
                                * (D[idx, jdx] - psi_L[idx, jdx])
                                * (Y[idx, kdx] - Y[jdx, kdx])
                            )
                if psi_U[idx, jdx] and idx != jdx:
                    if max(-psi_U[idx, jdx] + D[idx, jdx], 0) > 0:
                        for kdx in range(dim):
                            grad[idx, kdx] += (
                                4
                                * (D[idx, jdx] - psi_U[idx, jdx])
                                * (Y[idx, kdx] - Y[jdx, kdx])  # might be wrong
                            )
        return grad

    # def lgrad(self, Y, D_goal, F, Bl, L):
    #     D = distmat(Y)
    #     R = F * (D_goal - D)
    #     add_to_diagonal_fast(R)
    #     dfdY = R.dot(Y)
    #     Rb = np.maximum(Bl - L * D, 0)
    #     add_to_diagonal_fast(Rb)
    #     dfdYb = Rb.dot(Y)
    #     return 4 * (dfdY + dfdYb)

    @staticmethod
    @njit(cache=True, fastmath=True)
    def lhess(Y, w, D_goal, omega, psi_L, psi_U):
        num_el = Y.shape[0]
        dim = Y.shape[1]
        hess = np.zeros((num_el, dim))
        D = distmat(Y, omega + psi_L + psi_U)
        for idx in range(num_el):  # first hess
            for jdx in range(num_el):  # second hess
                if omega[idx, jdx] and idx != jdx:
                    sc = 0
                    for kdx in range(dim):
                        sc += (Y[idx, kdx] - Y[jdx, kdx]) * (w[idx, kdx] - w[jdx, kdx])
                    for kdx in range(dim):
                        hess[idx, kdx] += 4 * (
                            2 * sc * (Y[idx, kdx] - Y[jdx, kdx])
                            + (D[idx, jdx] - D_goal[idx, jdx])
                            * (w[idx, kdx] - w[jdx, kdx])
                        )
                if psi_L[idx, jdx] and idx != jdx:
                    if max(psi_L[idx, jdx] - D[idx, jdx], 0) > 0:
                        sc = 0
                        for kdx in range(dim):
                            sc += (Y[idx, kdx] - Y[jdx, kdx]) * (
                                w[idx, kdx] - w[jdx, kdx]
                            )
                        for kdx in range(dim):
                            hess[idx, kdx] += 4 * (
                                2 * sc * (Y[idx, kdx] - Y[jdx, kdx])
                                + (D[idx, jdx] - psi_L[idx, jdx])
                                * (w[idx, kdx] - w[jdx, kdx])
                            )
                if psi_U[idx, jdx] and idx != jdx:
                    if max(-psi_U[idx, jdx] + D[idx, jdx], 0) > 0:
                        sc = 0
                        for kdx in range(dim):
                            sc += (Y[idx, kdx] - Y[jdx, kdx]) * (
                                w[idx, kdx] - w[jdx, kdx]
                            )
                        for kdx in range(dim):
                            hess[idx, kdx] += 4 * (
                                2 * sc * (Y[idx, kdx] - Y[jdx, kdx])
                                + (D[idx, jdx] - psi_U[idx, jdx])
                                * (w[idx, kdx] - w[jdx, kdx])
                            )
        return hess

    # def lhess(self, Y, w, D_goal, F, Bl, L):
    #     D = distmat(Y)
    #     R = F * (D_goal - D)
    #     dDdZ = distmat_gram(Y @ w.T + w @ Y.T)  # directional der of dist matrix
    #     FdDdZ = F * dDdZ
    #     add_to_diagonal_fast(FdDdZ)
    #     add_to_diagonal_fast(R)
    #     Hw = 4 * (-FdDdZ.dot(Y) + R.dot(w))

    #     Rb = np.maximum(Bl - L * D, 0)
    #     dDdZb = np.where(Rb > 0, 1, 0) * (-L * dDdZ)
    #     add_to_diagonal_fast(dDdZb)
    #     add_to_diagonal_fast(Rb)
    #     Hwb = 4 * (dDdZb.dot(Y) + Rb.dot(w))
    #     return Hw + Hwb

    @staticmethod
    def generate_initialization(bounds, dim, omega, psi_L, psi_U):
        # Generates a random EDM within the set bounds
        lb = bounds[0]
        ub = bounds[1]
        # D_rand = ub**2
        # D_rand = sample_matrix(lb, ub) ** 2
        D_rand = (lb + 0.9 * (ub - lb)) ** 2
        # Y_rand = PCA(dist_to_gram(D_rand), dim)
        X_rand = MDS(gram_from_distance_matrix(D_rand), eps=1e-8)
        Y_rand = linear_projection(X_rand, omega, dim)
        return Y_rand

    def create_cost(self, D_goal, omega, jit=True):
        inds = np.nonzero(omega)

        if jit:

            def cost(Y):
                return self.jcost(Y, D_goal, inds)

            def egrad(Y):
                return self.jgrad(Y, D_goal, inds)

            def ehess(Y, v):
                return self.jhess(Y, v, D_goal, inds)

            return cost, egrad, ehess

        else:

            def cost(Y):
                D = distance_matrix_from_pos(Y)
                S = omega * (D_goal - D)
                f = np.linalg.norm(S) ** 2
                return 0.5 * f

            def egrad(Y):
                D = distance_matrix_from_pos(Y)
                S = omega * (D_goal - D)
                dfdY = 4 * (S - np.diag(np.sum(S, axis=1))).dot(Y)
                return 0.5 * dfdY

            def ehess(Y, Z):
                D = distance_matrix_from_pos(Y)
                S = omega * (D_goal - D)
                dDdZ = distance_matrix_from_gram(Y.dot(Z.T) + Z.dot(Y.T))
                dSdZ = -omega * dDdZ
                d1 = 4 * (dSdZ - np.diag(np.sum(dSdZ, axis=1))).dot(Y)
                d2 = 4 * (S - np.diag(np.sum(S, axis=1))).dot(Z)
                HZ = d1 + d2
                return 0.5 * HZ

            # def cost_and_egrad(Y):
            #     D = distance_matrix_from_pos(Y)
            #     S = omega * (D_goal - D)
            #     f = np.linalg.norm(S) ** 2
            #     dfdY = 4 * (S - np.diag(np.sum(S, axis=1))).dot(Y)
            #     return f, dfdY

            return cost, egrad, ehess

    def create_cost_limits(self, D_goal, omega, psi_L, psi_U):
        def cost(Y):
            return self.lcost(Y, D_goal, omega, psi_L, psi_U)

        def egrad(Y):
            return self.lgrad(Y, D_goal, omega, psi_L, psi_U)

        def ehess(Y, v):
            return self.lhess(Y, v, D_goal, omega, psi_L, psi_U)

        return cost, egrad, ehess

    def solve(
        self,
        D_goal,
        omega,
        use_limits=False,
        bounds=None,
        Y_init=None,
        output_log=True,
    ):

        # Generate cost, gradient and hessian-vector product
        if not use_limits:
            [psi_L, psi_U] = [0 * omega, 0 * omega]
            cost, egrad, ehess = self.create_cost(D_goal, omega, jit=True)
        else:
            psi_L, psi_U = self.graph.distance_bound_matrices()
            cost, egrad, ehess = self.create_cost_limits(D_goal, omega, psi_L, psi_U)

        # Generate initialization
        if bounds is not None:
            Y_init = self.generate_initialization(bounds, self.dim, omega, psi_L, psi_U)
        elif Y_init is None:
            raise Exception("If not using bounds, provide an initialization!")

        # Define manifold
        manifold = PSDFixedRank(self.N, self.dim)  # define manifold

        # Define problem
        problem = pymanopt.Problem(
            manifold, cost=cost, egrad=egrad, ehess=ehess, verbosity=1
        )

        # Solve problem
        if output_log:
            self.solver._logverbosity = 2
            Y_sol, optlog = self.solver.solve(problem, x=Y_init)
            return optlog["final_values"]
        else:
            Y_sol = self.solver.solve(problem, x=Y_init)
            return Y_sol

    def solve_experiment_wrapper(
        self,
        D_goal,
        omega,
        use_limits=True,
        bounds=None,
        X=None,
        verbosity=0,
        max_attempts=10,
    ):
        """
        TODO: Refactor solve() to remove duplication in this method.
        :param D_goal:
        :param lower_limits:
        :param upper_limits:
        :param X:
        :param dim:
        :return:
        """
        [N, dim] = [self.N, self.dim]
        # if not use_limits:
        #     cost, egrad, ehess = self.create_cost(D_goal, F)
        # else:
        #     Bl = self.graph.distance_bound_matrix()
        #     cost, egrad, ehess = self.create_cost_limits(D_goal, F, Bl)
        if not use_limits:
            [psi_L, psi_U] = [0 * omega, 0 * omega]
            cost, egrad, ehess = self.create_cost(D_goal, omega)
        else:
            psi_L, psi_U = self.graph.distance_bound_matrices()
            cost, egrad, ehess = self.create_cost_limits(D_goal, omega, psi_L, psi_U)

        manifold = PSDFixedRank(self.N, dim)
        problem = pymanopt.Problem(
            manifold, cost=cost, egrad=egrad, ehess=ehess, verbosity=verbosity
        )

        self.solver._logverbosity = 2
        # Generate initialization
        if bounds is not None:
            X = self.generate_initialization(bounds, dim, omega, psi_L, psi_U)

        if self.solver._logverbosity < 2:
            Y_sol = self.solver.solve(problem, x=X)
            optlog = None
        else:
            Y_sol, optlog = self.solver.solve(problem, x=X)

        # self.solver.linesearch = None

        return Y_sol, optlog


if __name__ == "__main__":
    from graphik.utils.roboturdf import load_ur10
    from graphik.utils.geometry import trans_axis
    import timeit

    robot, graph = load_ur10()
    n = 6
    solver = RiemannianSolver(graph)
    q_goal = graph.robot.random_configuration()
    G_goal = graph.realization(q_goal)
    X_goal = pos_from_graph(G_goal)
    D_goal = graph.distance_matrix_from_joints(q_goal)
    T_goal = robot.get_pose(q_goal, f"p{n}")

    goals = {
        f"p{n}": T_goal.trans,
        f"q{n}": T_goal.dot(trans_axis(1, "z")).trans,
    }
    G = graph.complete_from_pos(goals)
    omega = adjacency_matrix_from_graph(G)
    inds = np.nonzero(omega)
    cost, egrad, ehess = solver.create_cost(D_goal, omega, False)
    cost2, egrad2, ehess2 = solver.create_cost(D_goal, omega, True)
    Y = pos_from_graph(graph.realization(robot.random_configuration()))
    Z = np.random.rand(Y.shape[0], Y.shape[1])

    print(ehess(Y, Z) - ehess2(Y, Z))
    # print(egrad(Y) - egrad2(Y))
    # print(cost(Y) - cost2(Y))

    print(
        np.average(
            timeit.repeat(
                "cost2(Y)",
                globals=globals(),
                number=1,
                repeat=10000,
            )
        )
    )
    print(
        np.average(
            timeit.repeat(
                "egrad2(Y)",
                globals=globals(),
                number=1,
                repeat=10000,
            )
        )
    )
    print(
        np.average(
            timeit.repeat(
                "ehess2(Y, Y)",
                globals=globals(),
                number=1,
                repeat=10000,
            )
        )
    )
