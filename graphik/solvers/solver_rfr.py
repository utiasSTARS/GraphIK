from graphik.robots.robot_base import RobotPlanar
from graphik.graphs.graph_base import RobotGraph
import numpy as np

from graphik.utils.manifolds.fixed_rank_psd_sym import PSDFixedRank
from graphik.solvers.solver_base import GraphProblemSolver
from graphik.utils.dgp import (
    adjacency_matrix_from_graph,
    distance_matrix_from_graph,
    distance_matrix_from_pos,
    distance_matrix_from_gram,
    pos_from_graph,
    graph_from_pos,
    bound_smoothing,
)

from pymanopt import Problem
from pymanopt.solvers import ConjugateGradient
from pymanopt.tools import make_enum

from graphik.solvers.trust_region import TrustRegions


BetaTypes = make_enum(
    "BetaTypes", "FletcherReeves PolakRibiere HestenesStiefel HagerZhang".split()
)


def add_to_diagonal_fast(X: np.ndarray):
    X.ravel()[:: X.shape[1] + 1] += -np.sum(X, axis=0)


def frobenius_norm_sq(X: np.ndarray):
    "Return squared frobenius norm"
    return np.einsum("ij,ji->", X, X)


class RiemannianSolver(GraphProblemSolver):
    def __init__(self, params: dict) -> None:
        self.params = params

        if params["solver"] == "TrustRegions":
            self.solver = TrustRegions(
                mingradnorm=params["mingradnorm"],
                maxiter=params["maxiter"],
                logverbosity=params["logverbosity"],
            )
        elif params["solver"] == "ConjugateGradient":
            self.solver = ConjugateGradient(
                mingradnorm=params["mingradnorm"],
                maxiter=params["maxiter"],
                logverbosity=params["logverbosity"],
                minstepsize=1e-6,
                beta_type=BetaTypes[2],
                orth_value=4,
            )
        else:
            raise ValueError(
                "params[\"solver\"] must be one of 'ConjugateGradient', 'TrustRegions'"
            )
        super(RiemannianSolver, self).__init__(params)

    @property
    def params(self) -> dict:
        return self._params

    @params.setter
    def params(self, params: dict) -> None:
        self._params = params

    @staticmethod
    def cost(Y: np.ndarray, D_goal: np.ndarray, omega: np.ndarray):
        D = distance_matrix_from_pos(Y)
        R = omega * (D_goal - D)
        return frobenius_norm_sq(R)

    @staticmethod
    def grad(Y: np.ndarray, D_goal: np.ndarray, omega: np.ndarray):
        D = distance_matrix_from_pos(Y)
        R = omega * (D_goal - D)
        add_to_diagonal_fast(R)
        dfdY = 4 * R.dot(Y)
        return dfdY

    @staticmethod
    def hess(Y: np.ndarray, w: np.ndarray, D_goal: np.ndarray, omega: np.ndarray):
        D = distance_matrix_from_pos(Y)
        R = omega * (D_goal - D)
        dDdZ = -distance_matrix_from_gram(Y.dot(w.T) + w.dot(Y.T))
        FdDdZ = omega * dDdZ
        add_to_diagonal_fast(FdDdZ)
        add_to_diagonal_fast(R)
        Hw = 4 * (FdDdZ.dot(Y) + R.dot(w))
        return Hw

    @staticmethod
    def cost_limits(Y, D_goal, omega, psi_L, psi_U):
        D = distance_matrix_from_pos(Y)
        R = omega * (D_goal - D)
        L = np.maximum(psi_L - (psi_L > 0) * D, 0)
        U = np.maximum(-psi_U + (psi_U > 0) * D, 0)
        return frobenius_norm_sq(R) + frobenius_norm_sq(L) + frobenius_norm_sq(U)

    @staticmethod
    def grad_limits(Y, D_goal, omega, psi_L, psi_U):
        D = distance_matrix_from_pos(Y)
        R = omega * (D_goal - D)
        add_to_diagonal_fast(R)
        dfdY = R.dot(Y)

        L = np.maximum(psi_L - (psi_L > 0) * D, 0)
        add_to_diagonal_fast(L)
        dfdYL = L.dot(Y)

        U = np.maximum(-psi_U + (psi_U > 0) * D, 0)
        add_to_diagonal_fast(U)
        dfdYU = U.dot(Y)

        return 4 * (dfdY + dfdYL + dfdYU)

    @staticmethod
    def hess_limits(Y, w, D_goal, omega, psi_L, psi_U):
        D = distance_matrix_from_pos(Y)
        R = omega * (D_goal - D)
        dDdZ = distance_matrix_from_gram(
            Y @ w.T + w @ Y.T
        )  # directional der of dist matrix
        FdDdZ = omega * dDdZ
        add_to_diagonal_fast(FdDdZ)
        add_to_diagonal_fast(R)
        Hw = 4 * (-FdDdZ.dot(Y) + R.dot(w))

        L = np.maximum(psi_L - (psi_L > 0) * D, 0)
        dDdZL = np.where(L > 0, 1, 0) * ((psi_L > 0) * (-dDdZ))
        add_to_diagonal_fast(dDdZL)
        add_to_diagonal_fast(L)
        HwL = 4 * (dDdZL.dot(Y) + L.dot(w))

        U = np.maximum(-psi_U + (psi_U > 0) * D, 0)
        dDdZU = np.where(U > 0, 1, 0) * ((psi_U > 0) * dDdZ)
        add_to_diagonal_fast(dDdZU)
        add_to_diagonal_fast(U)
        HwU = 4 * (dDdZU.dot(Y) + U.dot(w))

        return Hw + HwL + HwU

    def create_cost(self, D_goal, omega, limits, psi_L=None, psi_U=None):

        if not limits:

            def cost(Y):
                return self.cost(Y, D_goal, omega)

            def egrad(Y):
                return self.grad(Y, D_goal, omega)

            def ehess(Y, v):
                return self.hess(Y, v, D_goal, omega)

        else:

            def cost(Y):
                return self.cost_limits(Y, D_goal, omega, psi_L, psi_U)

            def egrad(Y):
                return self.grad_limits(Y, D_goal, omega, psi_L, psi_U)

            def ehess(Y, v):
                return self.hess_limits(Y, v, D_goal, omega, psi_L, psi_U)

        return cost, egrad, ehess

    def solve(self, graph: RobotGraph, params: dict = None):
        if not params:
            params = {"goals": None, "joint_limits": False, "init": None}

        N, dim = graph.n_nodes, graph.dim

        if params["goals"]:
            G = graph.complete_from_pos(params["goals"])
        else:
            G = graph.directed  # just looking for feasible realization

        D_goal = distance_matrix_from_graph(G)
        omega = adjacency_matrix_from_graph(G)

        if params["joint_limits"]:
            psi_L, psi_U = graph.distance_bound_matrices()
            cost, egrad, ehess = self.create_cost(D_goal, omega, True, psi_L, psi_U)
        else:
            cost, egrad, ehess = self.create_cost(D_goal, omega, False)

        manifold = PSDFixedRank(N, dim)
        problem = Problem(manifold, cost=cost, egrad=egrad, ehess=ehess, verbosity=0)

        X = params["init"]
        Y_sol, optlog = self.solver.solve(problem, x=X)

        return optlog["final_values"]
