from graphik.robots import RobotPlanar
from graphik.graphs.graph_base import ProblemGraph as RobotGraph
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

import pymanopt
import pymanopt.function
from pymanopt.optimizers import ConjugateGradient, TrustRegions


def add_to_diagonal_fast(X: np.ndarray):
    X.ravel()[:: X.shape[1] + 1] += -np.sum(X, axis=0)


def frobenius_norm_sq(X: np.ndarray):
    "Return squared frobenius norm"
    return np.einsum("ij,ji->", X, X)


class RiemannianSolver(GraphProblemSolver):
    def __init__(self, params: dict) -> None:
        self.params = params

        common_kwargs = dict(
            min_gradient_norm=params["mingradnorm"],
            max_iterations=params["maxiter"],
            log_verbosity=params["logverbosity"],
            verbosity=0,
        )

        if params["solver"] == "TrustRegions":
            self.solver = TrustRegions(**common_kwargs)
        elif params["solver"] == "ConjugateGradient":
            self.solver = ConjugateGradient(
                **common_kwargs,
                min_step_size=1e-6,
                beta_rule="PolakRibiere",
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

        manifold = PSDFixedRank(N, dim)

        if params["joint_limits"]:
            psi_L, psi_U = graph.distance_bound_matrices()
            cost, egrad, ehess = self.create_cost(D_goal, omega, True, psi_L, psi_U)
        else:
            cost, egrad, ehess = self.create_cost(D_goal, omega, False)

        numpy_decorator = pymanopt.function.numpy(manifold)
        cost = numpy_decorator(cost)
        egrad = numpy_decorator(egrad)
        ehess = numpy_decorator(ehess)

        problem = pymanopt.Problem(
            manifold, cost,
            euclidean_gradient=egrad,
            euclidean_hessian=ehess,
        )

        X = params["init"]
        result = self.solver.run(problem, initial_point=X)

        return {
            "x": result.point,
            "f(x)": result.cost,
            "iterations": result.iterations,
            "stopping_criterion": result.stopping_criterion,
            "time": result.time,
            "gradnorm": result.gradient_norm,
        }
