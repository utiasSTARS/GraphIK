"""
Rank constraints via convex iteration (Dattorro's Convex Optimization and Euclidean Distance Geometry textbook).

"""
import numpy as np
import cvxpy as cp
from timeit import default_timer
from progress.bar import ShadyBar as Bar

from graphik.solvers.sdp_formulations import SdpSolverParams
from graphik.solvers.sdp_snl import (
    distance_range_constraints,
    solve_linear_cost_sdp,
    distance_constraints,
    distance_constraints_graph,
    extract_full_sdp_solution,
    extract_solution,
    chordal_sparsity_overlap_constraints
)
from graphik.solvers.constraints import get_full_revolute_nearest_point
from graphik.utils.roboturdf import load_ur10
from graphik.utils.constants import *
from graphik.graphs.graph_base import ProblemGraph


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
    start = default_timer()
    _, Q = np.linalg.eigh(G)
    Q = np.flip(Q, 1)
    U = Q[:, d:]
    return U@U.T, default_timer() - start


def solve_fantope_sdp(G: np.ndarray, d: int):
    n = G.shape[0]
    Z, cons = fantope_constraints(n, d)
    solve_fantope_iterate(G, Z, cons)
    return Z.value


def fantope_constraints(n: int, rank: int, sparsity_pattern: set = None):
    assert rank < n, "Needs a desired rank less than the problem dimension."
    Z = cp.Variable((n, n), PSD=True)
    constraints = [cp.trace(Z) == float(n - rank), np.eye(Z.shape[0]) - Z >> 0]

    if sparsity_pattern is not None:
        for idx in range(n):
            for jdx in range(idx, n):
                if (idx, jdx) not in sparsity_pattern:
                    constraints += [Z[idx, jdx] == 0.]
                    if idx != jdx:
                        constraints += [Z[jdx, idx] == 0.]

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


def solve_fantope_sparse(sdp_variable_map: dict, d: int):
    C_mapping = {}
    t_fantope = 0.0
    for clique in sdp_variable_map:
        G_clique = sdp_variable_map[clique].value  # Assumes it's been solved by cvxpy
        C_mapping[clique], t_clique = solve_fantope_closed_form(G_clique, d)
        t_fantope += t_clique
    return C_mapping, t_fantope


def solve_fantope_sdp_sparse(constraint_clique_dict: dict, sdp_variable_map: dict, d: int, verbose=False,
                             solver_params=None):

    # Make cvxpy variables and constraints for each Fantope
    fantope_sdp_variable_map = {}
    constraints = []
    for clique in sdp_variable_map:
        n_clique = sdp_variable_map[clique].shape[0]
        Z_clique = cp.Variable(sdp_variable_map[clique].shape, PSD=True)
        fantope_sdp_variable_map[clique] = Z_clique
        constraints += [cp.trace(Z_clique) == float(n_clique - d), np.eye(Z_clique.shape[0]) - Z_clique >> 0]

    # Get the overlap constraints that link each Fantope's overlapping variables
    constraints += chordal_sparsity_overlap_constraints(constraint_clique_dict, fantope_sdp_variable_map, d)

    # Solve the sparse Fantope SDP
    if solver_params is None:
        solver_params = SdpSolverParams()
    cost = 0.
    for clique in sdp_variable_map:
        Z_clique = fantope_sdp_variable_map[clique]
        G_clique = sdp_variable_map[clique].value
        cost += cp.trace(G_clique @ Z_clique)
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(verbose=verbose, solver="MOSEK", mosek_params=solver_params.mosek_params)

    # Return the desired cost function matrices
    C_mapping = {}
    for clique in sdp_variable_map:
        C_mapping[clique] = fantope_sdp_variable_map[clique].value
    return C_mapping, prob.solver_stats.solve_time


def sparse_eigenvalue_sum(sdp_variable_map: dict, d: int):
    running_eigenvalue_sum = 0.
    for clique in sdp_variable_map:
        Z_clique = sdp_variable_map[clique].value
        running_eigenvalue_sum += np.sum(np.linalg.eigvalsh(Z_clique)[:-d])
    return running_eigenvalue_sum


def get_sparsity_pattern(G, canonical_point_order: list) -> set:
    sparsity_pattern = set()
    G = G.copy()
    # remove the edges that don't have distances defined
    edges = []
    for u, v, data in G.edges(data=True):
        if not data.get(DIST, False):
            edges += [(u, v)]
    G.remove_edges_from(edges)
    undirected = G.to_undirected()
    for idx, point_idx in enumerate(canonical_point_order):
        for jdx in range(idx, len(canonical_point_order)):
            point_jdx = canonical_point_order[jdx]
            if undirected.has_edge(point_idx, point_jdx):
                sparsity_pattern.add((idx, jdx))

    return sparsity_pattern


def convex_iterate_sdp_snl_graph(
    graph: ProblemGraph,
    anchors: dict = {},
    ranges=False,
    max_iters=10,
    sparse=False,
    verbose=False,
    closed_form=True,
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
        G, anchors, sparse, ee_cost=False, angle_limits=ranges  # TODO: is a param other than ranges needed?
    )

    # Add inequalities (angluar limits, obstacles) if present
    inequality_map = distance_range_constraints(G, constraint_clique_dict, anchors) if ranges else None

    # Save runtimes
    primal_sdp_runtime = 0.
    fantope_solver_runtime = 0.

    n = len(canonical_point_order)
    N = n + d
    C = np.eye(N) if W_init is None else W_init  # Identity satisfies any sparsity pattern by default
    # bar = Bar("Convex iteration", max=max_iters, check_tty=False, hide_cursor=False)
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
        primal_sdp_runtime += prob.solver_stats.solve_time
        if not sparse:
            G = extract_full_sdp_solution(constraint_clique_dict, canonical_point_order, sdp_variable_map, N, d)
            eigvals_G = np.linalg.eigvalsh(G)  # Returns in ascending order (according to docs)
            eig_value_sum_vs_iterations.append(np.sum(eigvals_G[0:n]))
            C, t_fantope = solve_fantope_closed_form(G, robot.dim)

        else:
            if closed_form:
                C, t_fantope = solve_fantope_sparse(sdp_variable_map, d)
            else:
                C, t_fantope = solve_fantope_sdp_sparse(constraint_clique_dict, sdp_variable_map, d)
            eig_value_sum_vs_iterations.append(sparse_eigenvalue_sum(sdp_variable_map, d))
        fantope_solver_runtime += t_fantope
    #     bar.next()
    # bar.finish()
    return (
        C,
        constraint_clique_dict,
        sdp_variable_map,
        canonical_point_order,
        eig_value_sum_vs_iterations,
        prob,
        primal_sdp_runtime,
        fantope_solver_runtime
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

    # Store results
    final_eigvalue_sum_list = []
    primal_sdp_runtime = []
    fantope_runtime = []

    final_eigvalue_sum_list_sparse_naive = []
    primal_sdp_runtime_sparse_naive = []
    fantope_runtime_sparse_naive = []

    final_eigvalue_sum_list_sparse_sdp = []
    primal_sdp_runtime_sparse_sdp = []
    fantope_runtime_sparse_sdp = []

    bar = Bar("", max=n_runs, check_tty=False, hide_cursor=False)
    for idx in range(n_runs):
        # Generate a random feasible target
        q = robot.random_configuration()
        input_vals = get_full_revolute_nearest_point(graph, q, full_points)

        # End-effectors don't include the base pair at this step, that's inside of convex_iterate_sdp_snl_graph()
        end_effectors = {
            key: input_vals[key] for key in [f"p{robot.n}", f"q{robot.n}"]
        }

        # Run 'dense' default solver
        (
            _,
            _,
            _,
            _,
            eig_value_sum_vs_iterations,
            _,
            t_primal,
            t_fantope
        ) = convex_iterate_sdp_snl_graph(graph, anchors=end_effectors, ranges=False, max_iters=10,
                                         sparse=False, verbose=False, closed_form=True)
        # solution = extract_solution(constraint_clique_dict, sdp_variable_map, d)
        final_eigvalue_sum_list.append(eig_value_sum_vs_iterations[-1])
        primal_sdp_runtime.append(t_primal)
        fantope_runtime.append(t_fantope)

        # Run 'naive' sparse solver (Wang and Yu, 2018)
        (
            _,
            _,
            _,
            _,
            eig_value_sum_vs_iterations,
            _,
            t_primal,
            t_fantope
        ) = convex_iterate_sdp_snl_graph(graph, anchors=end_effectors, ranges=False, max_iters=10,
                                         sparse=True, verbose=False, closed_form=True)
        final_eigvalue_sum_list_sparse_naive.append(eig_value_sum_vs_iterations[-1])
        primal_sdp_runtime_sparse_naive.append(t_primal)
        fantope_runtime_sparse_naive.append(t_fantope)

        # Run sparse solver where Fantope program has a sparsity pattern matched to the primal SDP's
        (
            _,
            _,
            _,
            _,
            eig_value_sum_vs_iterations,
            _,
            t_primal,
            t_fantope
        ) = convex_iterate_sdp_snl_graph(graph, anchors=end_effectors, ranges=False, max_iters=10,
                                         sparse=True, verbose=False, closed_form=False)
        final_eigvalue_sum_list_sparse_sdp.append(eig_value_sum_vs_iterations[-1])
        primal_sdp_runtime_sparse_sdp.append(t_primal)
        fantope_runtime_sparse_sdp.append(t_fantope)

        bar.next()
    bar.finish()

    # Visualize results
    from matplotlib import pyplot as plt
    from matplotlib import rc
    rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
    rc("text", usetex=True)

    # Plot parameters
    n_bins = 100
    colors = ['r', 'g', 'b']
    # Runtime histograms
    primal_sdp_runtime_full = np.vstack([primal_sdp_runtime, primal_sdp_runtime_sparse_naive, primal_sdp_runtime_sparse_sdp]).T
    primal_sdp_runtime_min = np.min(primal_sdp_runtime_full)
    primal_sdp_runtime_max = np.max(primal_sdp_runtime_full)
    primal_sdp_runtime_bins = np.linspace(primal_sdp_runtime_min, primal_sdp_runtime_max, n_bins)
    plt.figure()
    plt.hist(primal_sdp_runtime_full, bins=primal_sdp_runtime_bins, color=colors)
    plt.legend(('Dense', 'Sparse (Naive)', 'Sparse (SDP)'))
    plt.title('Distribution of Primal SDP Runtimes')
    plt.xlabel('Runtime (s)')
    plt.ylabel('Count')
    plt.grid()
    plt.show()

    fantope_runtime_full = np.vstack([fantope_runtime, fantope_runtime_sparse_naive, fantope_runtime_sparse_sdp]).T
    fantope_runtime_min = np.min(fantope_runtime_full)
    fantope_runtime_max = np.max(fantope_runtime_full)
    fantope_runtime_bins = np.linspace(fantope_runtime_min, fantope_runtime_max, n_bins)
    plt.figure()
    plt.hist(fantope_runtime_full, bins=fantope_runtime_bins, color=colors)
    plt.legend(('Dense', 'Sparse (Naive)', 'Sparse (SDP)'))
    plt.title('Distribution of Fantope Solver Runtimes')
    plt.xlabel('Runtime (s)')
    plt.ylabel('Count')
    plt.grid()
    plt.show()

    # Eigenvalue sum histograms
    eigenvalue_sum_full = np.vstack([final_eigvalue_sum_list, final_eigvalue_sum_list_sparse_naive, final_eigvalue_sum_list_sparse_sdp]).T
    eigenvalue_sum_min = max(np.min(eigenvalue_sum_full), 1e-12)
    eigenvalue_sum_max = np.max(eigenvalue_sum_full)
    # eigenvalue_sum_bins = np.log10(np.linspace(eigenvalue_sum_min, eigenvalue_sum_max, n_bins))
    eigenvalue_sum_bins = np.log10(np.logspace(np.log10(eigenvalue_sum_min), np.log10(eigenvalue_sum_max), n_bins))
    plt.figure()
    plt.hist(np.log10(eigenvalue_sum_full), bins=eigenvalue_sum_bins, color=colors)
    plt.legend(('Dense', 'Sparse (Naive)', 'Sparse (SDP)'))
    plt.title('Distribution of Excess Rank')
    plt.xlabel('log$_{10}$ Excess Rank')
    plt.ylabel('Count')
    plt.grid()
    plt.show()
