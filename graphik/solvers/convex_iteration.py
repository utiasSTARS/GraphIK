"""
Rank constraints via convex iteration (Dattorro's Convex Optimization and Euclidean Distance Geometry textbook).

"""
import numpy as np
import cvxpy as cp
from graphik.solvers.sdp_formulations import SdpSolverParams


def fantope_constraints(n: int , rank: int):
    assert rank < n, "Needs a desired rank less than the problem dimension."
    Z = cp.Variable((n, n), PSD=True)
    constraints = [cp.trace(Z) == float(n - rank), np.eye(Z.shape[0]) - Z >> 0]

    return Z, constraints


def solve_fantope_iterate(G: np.ndarray, Z: cp.Variable, constraints: list, verbose=False, solver_params=None):
    # TODO: templatize for speed? Ask Filip about that feature, I forget what it's called
    prob = cp.Problem(cp.Minimize(cp.trace(G@Z)), constraints)
    if solver_params is None:
        solver_params = SdpSolverParams()
    prob.solve(verbose=verbose, solver="MOSEK", mosek_params=solver_params.mosek_params)
    return prob


if __name__ == '__main__':

    Z, constraints = fantope_constraints(2, 1)

    G = np.array([[1., 0.],
                  [0., 2.]])

    prob = solve_fantope_iterate(G, Z, constraints)
