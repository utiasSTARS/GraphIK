#!/usr/bin/env python3
from graphik.utils.dgp import adjacency_matrix_from_graph, bound_smoothing, distance_matrix_from_graph, graph_from_pos, sample_matrix
import pymanopt
import pymanopt.function

import numpy as np
from pymanopt.optimizers import ConjugateGradient, TrustRegions
from graphik.utils import (
    distance_matrix_from_gram,
    distance_matrix_from_pos,
    MDS,
    linear_projection,
    gram_from_distance_matrix,
)
from graphik.utils.manifolds.fixed_rank_psd_sym import PSDFixedRank
from graphik.graphs.graph_base import ProblemGraph

try:
    from graphik.solvers.costgrd import jcost, jgrad, jhess, lcost, lgrad, lhess
except ModuleNotFoundError:
    print("AOT compiled functions not found. To improve performance please run solvers/costs.py.")

def adjoint(X: np.ndarray) -> np.ndarray:
    D = np.zeros_like(X)
    np.einsum('ijj->ij', D)[...] = np.sum(X, axis=-1)
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

        common_kwargs = dict(
            min_gradient_norm=self.params.get("mingradnorm", 0.5*1e-9),
            log_verbosity=self.params.get("logverbosity", 0),
            max_iterations=self.params.get("maxiter", 3000),
            verbosity=0,
        )

        if solver_type == "TrustRegions":
            self.solver = TrustRegions(
                **common_kwargs,
                theta=self.params.get("theta", 1.0),
                kappa=self.params.get("kappa", 0.1),
            )
        elif solver_type == "ConjugateGradient":
            self.solver = ConjugateGradient(
                **common_kwargs,
                min_step_size=self.params.get("minstepsize", 1e-10),
                orth_value=self.params.get("orth_value", 10e10),
                beta_rule=self.params.get("beta_type", "HagerZhang"),
            )
        else:
            raise ValueError(
                "params[\"solver\"] must be one of 'ConjugateGradient', 'TrustRegions'"
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

        # Per-Y cache: RTR's inner CG calls ehess many times with fixed Y.
        cache = {"Y": None}

        def _S(Y):
            if cache["Y"] is Y:
                return cache["S"], cache["S_diag"]
            S = omega * (D_goal - distance_matrix_from_pos(Y))
            S_diag = S.copy()
            np.fill_diagonal(S_diag, S_diag.diagonal() - np.sum(S_diag, axis=1))
            cache.update(Y=Y, S=S, S_diag=S_diag)
            return S, S_diag

        def cost(Y):
            S, _ = _S(Y)
            return np.linalg.norm(S) ** 2 / 2

        def egrad(Y):
            _, S_diag = _S(Y)
            return 2 * S_diag.dot(Y)

        def ehessp(Y, Z):
            _, S_diag = _S(Y)
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
            def cost(Y):
                return lcost(Y, D_goal, omega, psi_L, psi_U, inds)

            def egrad(Y):
                return lgrad(Y, D_goal, omega, psi_L, psi_U, inds)

            def ehess(Y, v):
                return lhess(Y, v, D_goal, omega, psi_L, psi_U, inds)

            return cost, egrad, ehess

        # Per-Y cache: A0/A1/A2, the pre-adjoint stack, and the active masks
        # depend only on Y, so they're reused across RTR's inner-CG ehess calls.
        cache = {"Y": None}

        def _Y_state(Y):
            if cache["Y"] is Y:
                return cache
            D = distance_matrix_from_pos(Y)
            A0 = omega * (D_goal - D)
            A1 = np.maximum(psi_L - LL * D, 0)
            A2 = -np.maximum(-psi_U + UU * D, 0)
            A_adj = adjoint(np.stack([A0, A1, A2], axis=0))
            m4 = -np.where(A1 > 0, 1, 0) * LL
            m5 = -np.where(-A2 > 0, 1, 0) * UU
            cache.update(Y=Y, A0=A0, A1=A1, A2=A2, A_adj=A_adj, m4=m4, m5=m5)
            return cache

        def cost(Y):
            s = _Y_state(Y)
            return (np.linalg.norm(s["A0"])**2 + np.linalg.norm(s["A1"])**2 + np.linalg.norm(s["A2"])**2) / 2

        def egrad(Y):
            s = _Y_state(Y)
            return 2 * np.sum(s["A_adj"].dot(Y), axis=0)

        def ehess(Y, Z):
            s = _Y_state(Y)
            d_yz = distance_matrix_from_gram(Y.dot(Z.T) + Z.dot(Y.T))
            A_z = np.stack([-omega * d_yz, s["m4"] * d_yz, s["m5"] * d_yz], axis=0)
            return 2 * np.sum(adjoint(A_z).dot(Y) + s["A_adj"].dot(Z), axis=0)

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
        manifold = PSDFixedRank(self.N, self.dim)

        # Generate cost, gradient and hessian-vector product
        if not use_limits:
            [psi_L, psi_U] = [0 * omega, 0 * omega]
            cost, egrad, ehess = self.create_cost(D_goal, omega)
        else:
            psi_L, psi_U = self.graph.distance_bound_matrices()
            cost, egrad, ehess = self.create_cost_limits(D_goal, omega, psi_L, psi_U)

        numpy_decorator = pymanopt.function.numpy(manifold)
        cost = numpy_decorator(cost)
        egrad = numpy_decorator(egrad)
        ehess = numpy_decorator(ehess)

        # Generate initialization
        if bounds is not None:
            Y_init = self.generate_initialization(bounds, self.dim, omega, psi_L, psi_U)
        elif Y_init is None:
            raise Exception("If not using bounds, provide an initialization!")

        # Define problem
        problem = pymanopt.Problem(
            manifold, cost,
            euclidean_gradient=egrad,
            euclidean_hessian=ehess,
        )

        result = self.solver.run(problem, initial_point=Y_init)
        if output_log:
            return {
                "x": result.point,
                "f(x)": result.cost,
                "iterations": result.iterations,
                "stopping_criterion": result.stopping_criterion,
                "time": result.time,
                "gradnorm": result.gradient_norm,
            }
        else:
            return result.point

def solve_with_riemannian(graph, T_goal, use_jit=True):
    G = graph.from_pose(T_goal)
    solver = RiemannianSolver(graph, jit=use_jit)
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
