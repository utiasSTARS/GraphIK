"""
Rank constraints via convex iteration (Dattorro's Convex Optimization and Euclidean Distance Geometry textbook).

"""
import numpy as np
import cvxpy as cp

from graphik.solvers.sdp_formulations import SdpSolverParams
from graphik.solvers.sdp_snl import (
    distance_range_constraints,
    solve_linear_cost_sdp,
    distance_constraints,
    distance_constraints_graph,
    extract_full_sdp_solution,
    extract_solution,
)
from graphik.solvers.constraints import get_full_revolute_nearest_point
from graphik.utils.roboturdf import load_ur10
from graphik.utils.constants import *
from graphik.graphs.graph_base import RobotGraph


def random_psd_matrix(N: int, d: int = None, normalize: bool = True) -> np.ndarray:
    """
    Return a random PSD matrix.

    :param N: size of the matrix (NxN)
    :param d: rank of the matrix (full rank if not specified)
    """
    d = N if d is None else d
    P = np.random.rand(N, d)
    W = P @ P.T
    if normalize:
        W = W / np.linalg.norm(W, ord="fro")
    return W


def solve_fantope_closed_form(G: np.ndarray, d:int):
    """

    :param G:
    :param d:
    """
    _, Q = np.linalg.eigh(G)
    Q = np.flip(Q, 1)
    U = Q[:, d+1:]
    return U@U.T


def solve_fantope_sdp(G: np.ndarray, d: int):
    n = G.shape[0]
    Z, cons = fantope_constraints(n, d)
    solve_fantope_iterate(G, Z, cons)
    return Z.value


def fantope_constraints(n: int, rank: int):
    assert rank < n, "Needs a desired rank less than the problem dimension."
    Z = cp.Variable((n, n), PSD=True)
    constraints = [cp.trace(Z) == float(n - rank), np.eye(Z.shape[0]) - Z >> 0]

    return Z, constraints


def solve_fantope_iterate(
    G: np.ndarray, Z: cp.Variable, constraints: list, verbose=False, solver_params=None
):
    # TODO: templatize for speed? Ask Filip about that feature, I forget what it's called
    prob = cp.Problem(cp.Minimize(cp.trace(G @ Z)), constraints)
    if solver_params is None:
        solver_params = SdpSolverParams()
    prob.solve(verbose=verbose, solver="MOSEK", mosek_params=solver_params.mosek_params)
    return prob


def convex_iterate_sdp_snl_graph(
    graph: RobotGraph,
    anchors: dict = {},
    ranges=False,
    max_iters=10,
    sparse=False,
    verbose=False,
    W_init=None
):
    # get a copy of the current robot + environment graph
    G = graph.directed.copy()

    # remove base nodes and all adjacent edges
    G.remove_node("x")
    G.remove_node("y")

    robot = graph.robot
    d = robot.dim
    eig_value_sum_vs_iterations = []

    # If a position is pre-defined for a node, set to anchor
    for node, data in G.nodes(data=True):
        if data.get(POS, None) is not None:
            anchors[node] = data[POS]

    # full_points = [node for node in G if node not in ["x", "y"]]
    canonical_point_order = [point for point in G if point not in anchors.keys()]
    constraint_clique_dict = distance_constraints_graph(
        G, anchors, sparse, ee_cost=False
    )

    inequality_map = distance_range_constraints(G, constraint_clique_dict, anchors) if ranges else None

    n = len(canonical_point_order)
    N = n + d
    C = np.eye(N) if W_init is None else W_init
    Z, ft_constraints = fantope_constraints(N, d)
    for iter in range(max_iters):
        solution, prob, sdp_variable_map, _ = solve_linear_cost_sdp(
            robot,
            anchors,
            constraint_clique_dict,
            C,
            canonical_point_order,
            verbose=False,
            inequality_constraints_map=inequality_map,
        )
        G = extract_full_sdp_solution(
            constraint_clique_dict, canonical_point_order, sdp_variable_map, N, d
        )
        eigvals_G = np.linalg.eigvalsh(
            G
        )  # Returns in ascending order (according to docs)
        eig_value_sum_vs_iterations.append(np.sum(eigvals_G[0:n]))
        _ = solve_fantope_iterate(G, Z, ft_constraints, verbose=verbose)
        C = Z.value

    return (
        C,
        constraint_clique_dict,
        sdp_variable_map,
        canonical_point_order,
        eig_value_sum_vs_iterations,
        prob,
    )


if __name__ == "__main__":

    # TODO: use the graph convex iteration from above and deprecate the old one
    # UR10 Test
    robot, graph = load_ur10()
    d = robot.dim
    full_points = [f"p{idx}" for idx in range(0, graph.robot.n + 1)] + [
        f"q{idx}" for idx in range(0, graph.robot.n + 1)
    ]

    n_runs = 10
    final_eigvalue_sum_list = []
    for idx in range(n_runs):
        # Generate a random feasible target
        q = robot.random_configuration()
        input_vals = get_full_revolute_nearest_point(graph, q, full_points)

        # End-effectors are 'generalized' to include the base pair ('p0', 'q0')
        end_effectors = {
            key: input_vals[key] for key in [f"p{robot.n}", f"q{robot.n}"]
        }
        canonical_point_order = [
            point for point in full_points if point not in end_effectors.keys()
        ]
        (
            C,
            constraint_clique_dict,
            sdp_variable_map,
            canonical_point_order,
            eig_value_sum_vs_iterations,
            prob,
        ) = convex_iterate_sdp_snl_graph(graph, anchors=end_effectors, ranges=False, max_iters=10,
                                         sparse=False, verbose=False)

        # solution = extract_solution(constraint_clique_dict, sdp_variable_map, d)
        print(eig_value_sum_vs_iterations)
        final_eigvalue_sum_list.append(eig_value_sum_vs_iterations[-1])

    print(final_eigvalue_sum_list)

    from matplotlib import pyplot as plt

    plt.hist(np.log10(final_eigvalue_sum_list))
    plt.show()
