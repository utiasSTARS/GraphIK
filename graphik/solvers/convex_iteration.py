"""
Rank constraints via convex iteration (Dattorro's Convex Optimization and Euclidean Distance Geometry textbook).

"""
import numpy as np
import cvxpy as cp
import networkx as nx
from timeit import default_timer

from graphik.solvers.sdp_formulations import SdpSolverParams, solve_sdp
from graphik.solvers.sdp_snl import (
    distance_range_constraints,
    solve_linear_cost_sdp,
    distance_constraints_graph,
    extract_full_sdp_solution,
    extract_solution,
    chordal_sparsity_overlap_constraints
)
from graphik.utils.constants import *
from graphik.graphs import ProblemGraph


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
    solve_sdp(prob, solver_params, verbose=verbose)

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


def convex_iterate_sdp_snl_graph(
    graph: ProblemGraph,
    anchors: dict = None,
    ranges=False,
    max_iters=10,
    sparse=False,
    verbose=False,
    closed_form=True,
    W_init=None,
    abs_eig_sum_tol=1e-6,
    rel_eig_sum_tol=1e-3,
    floor_mode=False,
    scs=False
):
    # get a copy of the current robot + environment graph
    G = nx.DiGraph(graph)
    # G = graph.directed.copy()

    # remove base nodes and all adjacent edges
    G.remove_node("x")
    G.remove_node("y")

    robot = graph.robot
    d = robot.dim
    eig_value_sum_vs_iterations = []

    # Copy: anchors gets extended below and must not leak into the
    # caller's dict (or across calls via a shared default).
    anchors = dict(anchors) if anchors else {}
    # If a position is pre-defined for a node, set to anchor
    for node, data in G.nodes(data=True):
        if data.get(POS, None) is not None:
            if node not in ('p0', 'q0') or not floor_mode:
                anchors[node] = data[POS]
    planar_constraints = {'p0': (np.array([0., 0., 1.]), 0.),
                          'q0': (np.array([0., 0., 1.]), 1.)} if floor_mode else None

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

    # Return feasibility
    feasible = FEASIBLE

    # Track the cost for convergence
    last_cost = 1e6

    n = len(canonical_point_order)
    N = n + d
    C = np.eye(N) if W_init is None else W_init  # Identity satisfies any sparsity pattern by default
    prob = None
    sdp_variable_map = None
    for iter in range(max_iters):
        solution, prob, sdp_variable_map, _ = solve_linear_cost_sdp(
            robot,
            anchors,
            constraint_clique_dict,
            C,
            prob,
            sdp_variable_map,
            canonical_point_order,
            verbose=False,
            inequality_constraints_map=inequality_map,
            planar_constraints=planar_constraints,
            scs=scs,
            warm_start=True
        )
        # Handle infeasibility case
        if solution is INFEASIBLE:
            feasible = INFEASIBLE
            primal_sdp_runtime += prob.solver_stats.solve_time
            break
        elif solution is SOLVER_ERROR:
            feasible = SOLVER_ERROR
            primal_sdp_runtime += 0.0  # TODO: handle this in post-processing
            break
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

        # Check for convergence
        eigval_sum_change = last_cost - prob.value
        rel_change = np.abs(eigval_sum_change)/np.abs(last_cost)
        if np.abs(eigval_sum_change) <= abs_eig_sum_tol or prob.value <= abs_eig_sum_tol or rel_change < rel_eig_sum_tol:
            break
        else:
            last_cost = prob.value

    return (
        C,
        constraint_clique_dict,
        sdp_variable_map,
        canonical_point_order,
        eig_value_sum_vs_iterations,
        prob,
        primal_sdp_runtime,
        fantope_solver_runtime,
        feasible
    )


def solve_with_cidgik(graph: ProblemGraph, T_goal: np.ndarray) -> (dict, dict):
    robot = graph.robot
    n = robot.n

    # Set up the anchors needed as input to CIDGIK
    anchors = {
        "p0": graph.nodes["p0"][POS],
        "q0": graph.nodes["q0"][POS],
        f"p{n}": T_goal[:3, 3],
        f"q{n}": T_goal[:3, 3] + T_goal[:3, 2]
    }

    # Solve with CIDGIK
    _, constraint_clique_dict, sdp_variable_map, _, _, _, _, _, feasible = \
        convex_iterate_sdp_snl_graph(
            graph,
            anchors,
            ranges=True,
            sparse=False,
            closed_form=True,
            scs=False
        )

    # Extract the angular configuration
    if feasible is FEASIBLE:
        solution = extract_solution(constraint_clique_dict, sdp_variable_map, robot.dim)

        # Add the end-effector goal points to the solution
        solution[f"p{robot.n}"] = anchors[f"p{robot.n}"]
        solution[f"q{robot.n}"] = anchors[f"q{robot.n}"]

        # Add the base points to the solution
        base_nodes = ["p0", "x", "y", "q0"]
        for node in base_nodes:
            solution[node] = graph.nodes[node][POS]
        G_sol = graph.from_pos(solution)
        q_sol = graph.joint_variables(G_sol, {f"p{n}": T_goal})

        return q_sol, solution
    else:
        return None, None

