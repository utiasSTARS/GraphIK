#!/usr/bin/env python3
from graphik.utils.dgp import adjacency_matrix_from_graph, bound_smoothing, distance_matrix_from_graph, graph_from_pos, sample_matrix
import pymanopt

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

#Workaround for SciPy bug: https://github.com/scipy/scipy/pull/8082
try:
    from graphik.solvers.costgrd import jcost, jgrad, jhess, lcost, lgrad, lhess
except ModuleNotFoundError as err:
    print("AOT compiled functions not found. To improve performance please run solvers/costs.py.")

def add_to_diagonal_fast(X: np.ndarray):
    X.ravel()[:: X.shape[1] + 1] += -np.sum(X, axis=0)

BetaTypes = tools.make_enum(
    "BetaTypes", "FletcherReeves PolakRibiere HestenesStiefel HagerZhang".split()
)

def adjoint(X: np.ndarray) -> np.ndarray:
    D = np.zeros_like(X)
    np.einsum('ijj->ij',D)[...] = np.sum(X, axis=-1)
    return X - D

class RiemannianSolver:
    def __init__(self, graph: ProblemGraph, cost_type="dense", jit=False, *args, **kwargs):

        self.params = {}
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.graph = graph
        self.dim = graph.dim
        self.N = graph.number_of_nodes()
        self.jit = jit

        solver_type = self.params.get("solver", "TrustRegions")

        if solver_type == "TrustRegions":
            self.solver = TrustRegions(
                mingradnorm=self.params.get("mingradnorm", 0.5*1e-9),
                logverbosity=self.params.get("logverbosity", 0),
                maxiter=self.params.get("maxiter", 3000),
                theta=self.params.get("theta", 1.0),
                kappa=self.params.get("kappa", 0.1),
            )
        elif solver_type == "ConjugateGradient":
            self.solver = ConjugateGradient(
                mingradnorm=self.params.get("mingradnorm",1e-9),
                logverbosity=self.params.get("logverbosity", 0),
                maxiter=self.params.get("maxiter", 10e4),
                minstepsize=self.params.get("minstepsize", 1e-10),
                orth_value=self.params.get("orth_value", 10e10),
                beta_type=self.params.get("beta_type", BetaTypes[3]),
            )
        else:
            raise (
                ValueError,
                "params[\"solver\"] must be one of 'ConjugateGradient', 'TrustRegions'",
            )


    @staticmethod
    def generate_initialization(bounds, dim, omega, psi_L, psi_U):
        # Generates a random EDM within the set bounds
        lb = np.sqrt(bounds[0])
        ub = np.sqrt(bounds[1])
        D_rand = (lb + 0.9 * (ub - lb)) ** 2
        X_rand = MDS(gram_from_distance_matrix(D_rand), eps=1e-8)
        Y_rand = linear_projection(X_rand, omega, dim)
        return Y_rand

    def create_cost(self, D_goal, omega):
        inds = np.nonzero(np.triu(omega))

        if self.jit:

            def cost(Y):
                return jcost(Y, D_goal, inds)

            def egrad(Y):
                return jgrad(Y, D_goal, inds)

            def ehessp(Y, Z):
                return jhess(Y, Z, D_goal, inds)

            return cost, egrad, ehessp

        else:

            def cost(Y):
                D = distance_matrix_from_pos(Y)
                S = omega * (D_goal - D)
                f = np.linalg.norm(S) ** 2
                return f/2

            def egrad(Y):
                D = distance_matrix_from_pos(Y)
                S = omega * (D_goal - D)
                np.fill_diagonal(S, S.diagonal() - np.sum(S, axis=1))
                dfdY = 2 * S.dot(Y)
                return dfdY

            def ehessp(Y, Z):
                D = distance_matrix_from_pos(Y)
                YZT = Y.dot(Z.T)
                YZT += YZT.T
                dSdZ = -omega * distance_matrix_from_gram(YZT)
                np.fill_diagonal(dSdZ, dSdZ.diagonal() - np.sum(dSdZ, axis=1))
                S = omega * (D_goal - D)
                np.fill_diagonal(S, S.diagonal() - np.sum(S, axis=1))
                H = 2 * (dSdZ.dot(Y) + S.dot(Z))
                return H

            return cost, egrad, ehessp

    def create_cost_limits(self, D_goal, omega, psi_L, psi_U, jit=True):
        diff = psi_L!=psi_U
        # inds = np.nonzero(np.triu(omega) + np.triu(psi_L>0) + np.triu(psi_U>0))
        inds = np.nonzero(np.triu(omega) + np.triu( diff * (psi_L>0)) + np.triu(diff * (psi_U>0)) )
        LL = diff*(psi_L>0)
        UU = diff*(psi_U>0)

        if self.jit:
            def cost(Y):
                return lcost(Y, D_goal, omega, psi_L, psi_U, inds)

            def egrad(Y):
                return lgrad(Y, D_goal, omega, psi_L, psi_U, inds)

            def ehess(Y, v):
                return lhess(Y, v, D_goal, omega, psi_L, psi_U, inds)
        else:
            def cost(Y):
                D = distance_matrix_from_pos(Y)
                E0 = omega * (D_goal - D)
                E1 = np.maximum(psi_L - LL * D, 0)
                E2 = np.maximum(-psi_U + UU * D, 0)
                return (np.linalg.norm(E0)**2 + np.linalg.norm(E1)**2 + np.linalg.norm(E2)**2)/2

            def egrad(Y):
                n = Y.shape[0]
                A = np.zeros([3, n, n])
                D = distance_matrix_from_pos(Y)
                A[0,:,:] = omega * (D_goal - D)
                A[1,:,:] = np.maximum(psi_L - LL * D, 0)
                A[2,:,:] = -np.maximum(-psi_U + UU * D, 0)
                C = adjoint(A).dot(Y)
                return 2*np.sum(C,axis=0)

            def ehess(Y, Z):
                n = Y.shape[0]
                D = distance_matrix_from_pos(Y)

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
                return 2*(np.sum(C,axis=0))

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
        # Generate cost, gradient and hessian-vector product
        if not use_limits:
            [psi_L, psi_U] = [0 * omega, 0 * omega]
            cost, egrad, ehess = self.create_cost(D_goal, omega)
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
            # manifold, cost=cost, egrad=egrad, ehess=ehess, verbosity=0,
            manifold, cost=cost, egrad=egrad, ehess=ehess, verbosity=0,
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
