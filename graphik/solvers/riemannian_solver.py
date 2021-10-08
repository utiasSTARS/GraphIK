#!/usr/bin/env python3
from graphik.utils.utils import list_to_variable_dict
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
from pymanopt.manifolds import Euclidean
from graphik.solvers.trust_region import TrustRegions
from graphik.graphs.graph_base import ProblemGraph
from graphik.solvers.costgrd import jcost, jgrad, jhess, lcost, lgrad, lhess
BetaTypes = tools.make_enum(
    "BetaTypes", "FletcherReeves PolakRibiere HestenesStiefel HagerZhang".split()
)


# @njit(cache=True, fastmath=True)
# def adjoint(X: np.ndarray):
#     num_atoms = X.shape[0]
#     for i in range(num_atoms):
#         rowsum = 0
#         for j in range(num_atoms):
#             rowsum += X[i, j]
#         X[i, i] += -rowsum
#     return X
    # Non-numba
    # X.ravel()[:: X.shape[1] + 1] += -np.sum(X, axis=0)
    # return X


# @njit(cache=True, fastmath=True)
# def distmat(Y: np.ndarray, F: np.ndarray = None):
#     num_atoms = Y.shape[0]
#     dim = Y.shape[1]
#     locs = Y
#     dmat = np.zeros((num_atoms, num_atoms))
#     if F is None:
#         F = np.ones((num_atoms, num_atoms))
#     for i in range(num_atoms):
#         for j in range(i + 1, num_atoms):
#             if F[i, j]:
#                 d = 0
#                 for k in range(dim):
#                     d += (locs[i, k] - locs[j, k]) ** 2
#                 dmat[i, j] = d
#                 dmat[j, i] = d
#     return dmat

@njit(cache=True, fastmath=True, error_model='numpy')
def distmat(Y: np.ndarray):
    num_atoms = Y.shape[0]
    dim = Y.shape[1]
    locs = Y
    dmat = np.zeros((num_atoms, num_atoms))
    for i in np.ndindex(*dmat.shape):
        d = 0
        for k in range(dim):
            d += (locs[i[0], k] - locs[i[1], k]) ** 2
        dmat[i] = d
        dmat[i] = d
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

def adjoint(X: np.ndarray):
    D = np.zeros_like(X)
    np.einsum('ijj->ij',D)[...] = np.sum(X, axis=-1)
    return X - D

class RiemannianSolver:
    def __init__(self, graph: ProblemGraph, params={}):

        self.params = params
        self.graph = graph
        self.dim = graph.dim
        self.N = graph.number_of_nodes()

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
        N = omega.shape[0]
        # K = 1e5 / N
        K = 1
        # np.reciprocal(D_goal, where = omega>0, out=omega)
        # omega = omega**(0.15)

        if jit:

            def cost(Y):
                return K * jcost(Y, D_goal, inds)

            def egrad(Y):
                return K * jgrad(Y, D_goal, inds)

            def ehess(Y, v):
                return K * jhess(Y, v, D_goal, inds)

            return cost, egrad, ehess

        else:

            # def cost(Y):
            #     D = distance_matrix_from_pos(Y)
            #     S = omega * (D - D_goal)
            #     f = np.linalg.norm(S) ** 2
            #     return f

            # def egrad(Y):
            #     D = distance_matrix_from_pos(Y)
            #     S = omega * (D - D_goal)
            #     dfdY = 2 * (np.diag(np.sum(S, axis=1)) - S).dot(Y)
            #     return 2 * dfdY

            # def ehess(Y, Z):
            #     D = distance_matrix_from_pos(Y)
            #     S = omega * (D - D_goal)
            #     dSdZ = omega * distance_matrix_from_gram(Y.dot(Z.T) + Z.dot(Y.T))
            #     d1 = 2 * (np.diag(np.sum(dSdZ, axis=1)) - dSdZ).dot(Y)
            #     d2 = 2 * (np.diag(np.sum(S, axis=1)) - S).dot(Z)
            #     HZ = d1 + d2
            #     return 2 * HZ


            def cost(Y):
                D = distance_matrix_from_pos(Y)
                S = omega * (D_goal - D)
                f = np.linalg.norm(S) ** 2
                return K * 0.5 * f

            def egrad(Y):
                D = distance_matrix_from_pos(Y)
                S = omega * (D_goal - D)
                dfdY = 2 * (S - np.diag(np.sum(S, axis=1))).dot(Y)
                return K * dfdY

            def ehess(Y, Z):
                D = distance_matrix_from_pos(Y)
                S = omega * (D_goal - D)
                dDdZ = distance_matrix_from_gram(Y.dot(Z.T) + Z.dot(Y.T))
                dSdZ = -omega * dDdZ
                d1 = 2 * (dSdZ - np.diag(np.sum(dSdZ, axis=1))).dot(Y)
                d2 = 2 * (S - np.diag(np.sum(S, axis=1))).dot(Z)
                HZ = d1 + d2
                return K * HZ

            return cost, egrad, ehess

    def create_cost_limits(self, D_goal, omega, psi_L, psi_U, jit=True):
        diff = psi_L!=psi_U
        # inds = np.nonzero(np.triu(omega) + np.triu(psi_L>0) + np.triu(psi_U>0))
        inds = np.nonzero(np.triu(omega) + np.triu( diff * (psi_L>0)) + np.triu(diff * (psi_U>0)) )
        LL = diff*(psi_L>0)
        UU = diff*(psi_U>0)
        N = omega.shape[0]
        # K = 1 / N
        K = 1

        if jit:
            def cost(Y):
                return K * lcost(Y, D_goal, omega, psi_L, psi_U, inds)

            def egrad(Y):
                return K * lgrad(Y, D_goal, omega, psi_L, psi_U, inds)

            def ehess(Y, v):
                return K * lhess(Y, v, D_goal, omega, psi_L, psi_U, inds)
        else:
            # NOTE not tested
            def cost(Y):
                D = distmat(Y)
                E0 = omega * (D_goal - D)
                E1 = np.maximum(psi_L - LL * D, 0)
                E2 = np.maximum(-psi_U + UU * D, 0)
                return K * 0.5 * (np.linalg.norm(E0)**2 + np.linalg.norm(E1)**2 + np.linalg.norm(E2)**2)

            # def cost(A):
            #     n = Y.shape[0]
            #     D = distmat(Y)
            #     A = np.zeros([3,n,n])
            #     A[0,:,:] = omega * (D_goal - D)
            #     A[1,:,:] = np.maximum(psi_L - LL * D, 0)
            #     A[2,:,:] = -np.maximum(-psi_U + UU * D, 0)
            #     return 0.5 * np.sum(np.linalg.norm(A[:3,:,:], axis=(1,2))**2,axis=0)

            # def egrad(Y):
            #     D = distmat(Y)
            #     S = omega * (D_goal - D)
            #     dfdY = 4 * (adjoint(S)).dot(Y)
            #     Sl = np.maximum(psi_L - LL * D, 0)
            #     dSldY = 4 * (adjoint(Sl)).dot(Y)
            #     Su = np.maximum(-psi_U + UU * D, 0)
            #     dSudY = 4 * (adjoint(-Su)).dot(Y)
            #     return 0.5 * (dfdY + dSldY + dSudY)

            def egrad(Y):
                n = Y.shape[0]
                A = np.zeros([3, n, n])
                D = distmat(Y)
                A[0,:,:] = omega * (D_goal - D)
                A[1,:,:] = np.maximum(psi_L - LL * D, 0)
                A[2,:,:] = -np.maximum(-psi_U + UU * D, 0)
                C = adjoint(A).dot(Y)
                return K * 2*np.sum(C,axis=0)

            # def ehess(Y, Z):
            #     D = distmat(Y)
            #     S = omega * (D_goal - D)
            #     dDdZ = distance_matrix_from_gram(Y.dot(Z.T) + Z.dot(Y.T))
            #     dSdZ = -omega * dDdZ
            #     d1 = 4 * (dSdZ - np.diag(np.sum(dSdZ, axis=1))).dot(Y)
            #     d2 = 4 * (S - np.diag(np.sum(S, axis=1))).dot(Z)
            #     HZ = d1 + d2

            #     Sl = np.maximum(psi_L - LL * D, 0)
            #     wl = np.where(Sl > 0, 1, 0)
            #     dSldZ = - wl * (LL * dDdZ) # change only where active
            #     d1 = 4 * (dSldZ - np.diag(np.sum(dSldZ, axis=1))).dot(Y)
            #     d2 = 4 * (Sl - np.diag(np.sum(Sl, axis=1))).dot(Z)
            #     HZl = d1 + d2

            #     Su = np.maximum(-psi_U + UU * D, 0)
            #     wu = np.where(Su > 0, 1, 0)
            #     dSudZ = wu * (UU * dDdZ)
            #     d1 = 4 * (-dSudZ + np.diag(np.sum(dSudZ, axis=1))).dot(Y)
            #     d2 = 4 * (-Su + np.diag(np.sum(Su, axis=1))).dot(Z)
            #     HZu = d1 + d2
            #     return 0.5*(HZ + HZl + HZu)

            def ehess(Y, Z):
                n = Y.shape[0]
                D = distmat(Y)

                A = np.zeros([6, n, n])
                A[0,:,:] = omega * (D_goal - D)
                A[1,:,:] = np.maximum(psi_L - LL * D, 0)
                A[2,:,:] = -np.maximum(-psi_U + UU * D, 0)
                A[3,:,:] = -omega # dSdZ
                A[4,:,:] = - np.where(A[1,:,:] > 0, 1, 0) * LL # dSldZ
                A[5,:,:] = - np.where(-A[2,:,:] > 0, 1, 0) * UU # dSudZ
                A[3:,:,:] *= distance_matrix_from_gram(Y.dot(Z.T) + Z.dot(Y.T))

                A = adjoint(A)

                C = A[3:,:,:].dot(Y) + A[:3,:,:].dot(Z)
                return K * 2*(np.sum(C,axis=0))

        return cost, egrad, ehess

    def solve(
        self,
        D_goal,
        omega,
        use_limits=False,
        bounds=None,
        Y_init=None,
        jit = True,
        output_log=True,
    ):
        # Generate cost, gradient and hessian-vector product
        if not use_limits:
            [psi_L, psi_U] = [0 * omega, 0 * omega]
            cost, egrad, ehess = self.create_cost(D_goal, omega, jit=jit)
        else:
            psi_L, psi_U = self.graph.distance_bound_matrices()
            cost, egrad, ehess = self.create_cost_limits(D_goal, omega, psi_L, psi_U, jit=jit)

        # Generate initialization
        if bounds is not None:
            Y_init = self.generate_initialization(bounds, self.dim, omega, psi_L, psi_U)
        elif Y_init is None:
            raise Exception("If not using bounds, provide an initialization!")

        # Define manifold
        manifold = PSDFixedRank(self.N, self.dim)  # define manifold
        # manifold = Euclidean(self.N, self.dim)

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



if __name__ == "__main__":
    from graphik.utils.roboturdf import load_ur10
    from graphik.utils.geometry import trans_axis
    import timeit
    # np.set_printoptions(precision=2,linewidth=np.nan,threshold=np.nan)
    np.set_printoptions(suppress=True, precision=3, edgeitems=10, linewidth=180)
    # np.random.seed(22)
    n = 6
    # angular_limits = np.minimum(np.random.rand(n) * (np.pi / 2) + np.pi / 2, np.pi)
    angular_limits = np.ones(n) * (np.pi / 100)
    ub = angular_limits
    lb = -angular_limits
    robot, graph = load_ur10(limits=(lb,ub))
    solver = RiemannianSolver(graph)
    q_goal = graph.robot.random_configuration()
    G_goal = graph.realization(q_goal)
    X_goal = pos_from_graph(G_goal)
    D_goal = graph.distance_matrix_from_joints(q_goal)
    T_goal = robot.pose(q_goal, f"p{n}")

    goals = {
        f"p{n}": T_goal.trans,
        f"q{n}": T_goal.dot(trans_axis(1, "z")).trans,
    }
    G = graph.from_pos(goals)
    omega = adjacency_matrix_from_graph(G)
    inds = np.nonzero(omega)
    # cost, egrad, ehess = solver.create_cost(D_goal, omega, False)
    # cost2, egrad2, ehess2 = solver.create_cost(D_goal, omega, True)
    psi_L, psi_U = solver.graph.distance_bound_matrices()
    cost, egrad, ehess = solver.create_cost_limits(D_goal, omega, psi_L, psi_U, False)
    cost2, egrad2, ehess2 = solver.create_cost_limits(D_goal, omega, psi_L, psi_U, True)
    # cost2, egrad2, ehess2 = solver.create_cost(D_goal, omega, True)
    q = list_to_variable_dict([2,1,2,1,2,3])
    Y = pos_from_graph(graph.realization(q))
    Z = np.random.rand(Y.shape[0], Y.shape[1])
    inds = np.nonzero(np.triu(omega))
    # D = distmat(Y)

    print(cost(Y) - cost2(Y))
    print(egrad(Y) - egrad2(Y))
    print(ehess(Y,Z) - ehess2(Y,Z))
    # print(list(zip(*inds)))


    # diff = psi_L!=psi_U
    # inds = np.nonzero(np.triu(omega) + np.triu( diff * (psi_L>0)) + np.triu(diff * (psi_U>0)) )
    # L = np.triu(psi_L>0).astype(np.float32)
    # U = np.triu(psi_U>0).astype(np.float32)

    # print(
    #     np.average(
    #         timeit.repeat(
    #             "np.zeros(1)",
    #             globals=globals(),
    #             number=1,
    #             repeat=10000,
    #         )
    #     )
    # )
    print("Cost JIT")
    print(
        np.average(
            timeit.repeat(
                "cost(Y)",
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
                "egrad(Y)",
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
                "ehess(Y, Z)",
                globals=globals(),
                number=1,
                repeat=10000,
            )
        )
    )
