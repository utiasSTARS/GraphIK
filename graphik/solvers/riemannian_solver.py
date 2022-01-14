#!/usr/bin/env python3
from graphik.utils.dgp import adjacency_matrix_from_graph, bound_smoothing, distance_matrix_from_graph, graph_from_pos
import pymanopt

from numba import njit
import numpy as np
from pymanopt import tools
from pymanopt.solvers import ConjugateGradient
from graphik.utils import (
    distance_matrix_from_gram,
    distance_matrix_from_pos,
    MDS,
    linear_projection,
    gram_from_distance_matrix,
)
from graphik.utils.manifolds.fixed_rank_psd_sym import PSDFixedRank
from graphik.solvers.trust_region import TrustRegions
from graphik.graphs.graph_base import ProblemGraph
from graphik.solvers.costgrd import jcost, jgrad, jhess, lcost, lgrad, lhess
BetaTypes = tools.make_enum(
    "BetaTypes", "FletcherReeves PolakRibiere HestenesStiefel HagerZhang".split()
)

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
        D_rand = (lb + 0.9 * (ub - lb)) ** 2
        X_rand = MDS(gram_from_distance_matrix(D_rand), eps=1e-8)
        Y_rand = linear_projection(X_rand, omega, dim)
        return Y_rand

    @staticmethod
    def create_cost(D_goal, omega, jit=True):
        inds = np.nonzero(np.triu(omega))
        K = 1

        if jit:

            def cost(Y):
                return K * jcost(Y, D_goal, inds)

            def egrad(Y):
                return K * jgrad(Y, D_goal, inds)

            def ehess(Y, v):
                return K * jhess(Y, v, D_goal, inds)

            return cost, egrad, ehess

        else:

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

    @staticmethod
    def create_cost_limits(D_goal, omega, psi_L, psi_U, jit=True):
        diff = psi_L!=psi_U
        # inds = np.nonzero(np.triu(omega) + np.triu(psi_L>0) + np.triu(psi_U>0))
        inds = np.nonzero(np.triu(omega) + np.triu( diff * (psi_L>0)) + np.triu(diff * (psi_U>0)) )
        LL = diff*(psi_L>0)
        UU = diff*(psi_U>0)
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

            def egrad(Y):
                n = Y.shape[0]
                A = np.zeros([3, n, n])
                D = distmat(Y)
                A[0,:,:] = omega * (D_goal - D)
                A[1,:,:] = np.maximum(psi_L - LL * D, 0)
                A[2,:,:] = -np.maximum(-psi_U + UU * D, 0)
                C = adjoint(A).dot(Y)
                return K * 2*np.sum(C,axis=0)

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

def solve_with_riemannian(graph, T_goal, use_jit=True):
    G = graph.from_pose(T_goal)
    solver = RiemannianSolver(graph)
    D_goal = distance_matrix_from_graph(G)
    omega = adjacency_matrix_from_graph(G)
    lb, ub = bound_smoothing(G)
    sol_info = solver.solve(D_goal, omega, use_limits=True, bounds=(lb, ub), jit=use_jit)
    G_sol = graph_from_pos(sol_info["x"], graph.node_ids)
    q_sol = graph.joint_variables(G_sol, {f"p{graph.robot.n}": T_goal})

    broken_limits = graph.check_distance_limits(graph.realization(q_sol), tol=1e-6)
    if len(broken_limits) > 0:
        return None, None
    else:
        return q_sol, sol_info["x"]
