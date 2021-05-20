#!/usr/bin/env python3
import pymanopt

from numba import njit
import numpy as np
from pymanopt import tools
from pymanopt.solvers import ConjugateGradient
from graphik.utils import (
    pos_from_graph,
    adjacency_matrix_from_graph,
    distance_matrix_from_gram,
    distance_matrix_from_pos,
    MDS,
    linear_projection,
    gram_from_distance_matrix,
)
from graphik.utils.manifolds.fixed_rank_psd_sym import PSDFixedRank
from graphik.solvers.trust_region import TrustRegions
from graphik.graphs.graph_base import RobotGraph
from graphik.solvers.costgrd import jcost, jgrad, jhess, lcost, lgrad, lhess
BetaTypes = tools.make_enum(
    "BetaTypes", "FletcherReeves PolakRibiere HestenesStiefel HagerZhang".split()
)


@njit(cache=True, fastmath=True)
def add_to_diagonal_fast(X: np.ndarray):
    num_atoms = X.shape[0]
    for i in range(num_atoms):
        rowsum = 0
        for j in range(num_atoms):
            rowsum += X[i, j]
        X[i, i] += -rowsum
    return X
    # Non-numba
    # X.ravel()[:: X.shape[1] + 1] += -np.sum(X, axis=0)
    # return X


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

    @staticmethod
    def create_cost(D_goal, omega, jit=True):
        inds = np.nonzero(np.triu(omega))

        if jit:

            def cost(Y):
                return jcost(Y, D_goal, inds)

            def egrad(Y):
                return jgrad(Y, D_goal, inds)

            def ehess(Y, v):
                return jhess(Y, v, D_goal, inds)

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

    def create_cost_limits(self, D_goal, omega, psi_L, psi_U, jit=True):
        inds = np.nonzero(np.triu(omega) + np.triu(psi_L>0) + np.triu(psi_U>0))
        L = np.triu(psi_L>0)
        U = np.triu(psi_U>0)

        if jit:
            def cost(Y):
                return lcost(Y, D_goal, omega, psi_L, psi_U, inds)

            def egrad(Y):
                return lgrad(Y, D_goal, omega, psi_L, psi_U, inds)

            def ehess(Y, v):
                return lhess(Y, v, D_goal, omega, psi_L, psi_U, inds)
        else:
            # NOTE not tested
            def cost(Y):
                D = distmat(Y)
                E0 = omega * (D_goal - D)
                E1 = np.maximum(psi_L - L * D, 0)
                E2 = np.maximum(-psi_U + U * D, 0)
                return 0.5 * (np.linalg.norm(E0)**2 + np.linalg.norm(E1)**2 + np.linalg.norm(E2)**2)

            def egrad(Y):
                D = distmat(Y)
                E0 = omega * (D_goal - D)
                dE0dY = 4 * (E0 - np.diag(np.sum(E0, axis=1))).dot(Y)
                E1 = np.maximum(psi_L - L * D, 0)
                dE1dY = 4 * (E1 - np.diag(np.sum(E1, axis=1))).dot(Y)
                E2 = np.maximum(-psi_U + U * D, 0)
                dE2dY = 4 * (E2 - np.diag(np.sum(E2, axis=1))).dot(Y)
                return 0.5 * (dE0dY + dE1dY + dE2dY)

            def ehess(Y, Z):
                D = distance_matrix_from_pos(Y)
                E0 = omega * (D_goal - D)
                dDdZ = distance_matrix_from_gram(Y.dot(Z.T) + Z.dot(Y.T))
                dE0dZ = -omega * dDdZ
                d1 = 4 * (dE0dZ - np.diag(np.sum(dE0dZ, axis=1))).dot(Y)
                d2 = 4 * (E0 - np.diag(np.sum(E0, axis=1))).dot(Z)
                HE0 = d1 + d2

                E1 = np.maximum(psi_L - L * D, 0)
                dE1dZ1 = np.where(E1 > 0, 1, 0) * (-L * dDdZ)
                d1 = 4 * (dE1dZ1 - np.diag(np.sum(dE1dZ1, axis=1))).dot(Y)
                d2 = 4 * (E1 - np.diag(np.sum(E1, axis=1))).dot(Z)
                HE1 = 4 * (dDdZb.dot(Y) + E1.dot(w))
                return HE0

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
            manifold, cost=cost, egrad=egrad, ehess=ehess, verbosity=0
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
    np.random.seed(22)
    n = 6
    angular_limits = np.minimum(np.random.rand(n) * (np.pi / 2) + np.pi / 2, np.pi)
    ub = angular_limits
    lb = -angular_limits
    robot, graph = load_ur10(limits=(lb,ub))
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

    psi_L, psi_U = solver.graph.distance_bound_matrices()
    # cost2, egrad2, ehess2 = solver.create_cost_limits(D_goal, omega, psi_L, psi_U)
    print("Cost JIT")
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
    print("Grad JIT")
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
    print("Hess JIT")
    print(
        np.average(
            timeit.repeat(
                "ehess2(Y, Z)",
                globals=globals(),
                number=1,
                repeat=10000,
            )
        )
    )
