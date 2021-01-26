"""
Rank constraints via convex iteration (Dattorro's Convex Optimization and Euclidean Distance Geometry textbook).

"""
import numpy as np
import cvxpy as cp

from graphik.solvers.sdp_formulations import SdpSolverParams
from graphik.solvers.sdp_snl import solve_linear_cost_sdp, distance_constraints, extract_full_sdp_solution, extract_solution
from graphik.solvers.constraints import get_full_revolute_nearest_point
from graphik.utils.roboturdf import load_ur10


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


def convex_iterate_sdp_snl(robot, end_effectors, max_iters=10, sparse=False, verbose=False):
    d = robot.dim
    # TODO: add more logging
    eig_value_sum_vs_iterations = []
    full_points = [f'p{idx}' for idx in range(0, robot.n + 1)] + \
                  [f'q{idx}' for idx in range(0, robot.n + 1)]
    canonical_point_order = [point for point in full_points if point not in end_effectors.keys()]
    constraint_clique_dict = distance_constraints(robot, end_effectors, sparse, ee_cost=False)
    n = len(canonical_point_order)
    N = n + d
    C = np.eye(N)
    Z, ft_constraints = fantope_constraints(N, d)
    for iter in range(max_iters):
        solution, prob, sdp_variable_map, _ = solve_linear_cost_sdp(robot, end_effectors, constraint_clique_dict, C,
                                                                    canonical_point_order, verbose=False,
                                                                    inequality_constraints_map=None)
        G = extract_full_sdp_solution(constraint_clique_dict, canonical_point_order, sdp_variable_map, N, d)
        eigvals_G = np.linalg.eigvalsh(G)  # Returns in ascending order (according to docs)
        eig_value_sum_vs_iterations.append(np.sum(eigvals_G[0:n]))
        _ = solve_fantope_iterate(G, Z, ft_constraints, verbose=verbose)
        C = Z.value

    return C, constraint_clique_dict, sdp_variable_map, canonical_point_order, eig_value_sum_vs_iterations, prob


if __name__ == '__main__':

    # # Simple test
    # Z, constraints = fantope_constraints(2, 1)
    # G = np.array([[1., 0.],
    #               [0., 2.]])
    # prob = solve_fantope_iterate(G, Z, constraints)

    # UR10 Test
    robot, graph = load_ur10()
    d = robot.dim
    full_points = [f'p{idx}' for idx in range(0, graph.robot.n + 1)] + \
                  [f'q{idx}' for idx in range(0, graph.robot.n + 1)]

    n_runs = 1
    final_eigvalue_sum_list = []
    for idx in range(n_runs):
        # Generate a random feasible target
        q = robot.random_configuration()
        input_vals = get_full_revolute_nearest_point(graph, q, full_points)

        # End-effectors are 'generalized' to include the base pair ('p0', 'q0')
        end_effectors = {key: input_vals[key] for key in ['p0', 'q0', f'p{robot.n}', f'q{robot.n}']}
        canonical_point_order = [point for point in full_points if point not in end_effectors.keys()]
        C, constraint_clique_dict, sdp_variable_map, canonical_point_order, eig_value_sum_vs_iterations, prob = \
            convex_iterate_sdp_snl(robot, end_effectors)

        # solution = extract_solution(constraint_clique_dict, sdp_variable_map, d)
        print(eig_value_sum_vs_iterations)
        final_eigvalue_sum_list.append(eig_value_sum_vs_iterations[-1])

    print(final_eigvalue_sum_list)

    from matplotlib import pyplot as plt

    plt.hist(np.log10(final_eigvalue_sum_list))
    plt.show()
