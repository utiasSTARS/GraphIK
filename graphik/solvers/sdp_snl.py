"""
Rank-{2,3} SDP relaxation tailored to sensor network localization (SNL) applied to our DG/QCQP IK formulation.
"""
import numpy as np
import networkx as nx
import cvxpy as cp

from graphik.utils.roboturdf import load_ur10
from graphik.robots.robot_base import RobotRevolute
from graphik.graphs.graph_base import RobotRevoluteGraph
from graphik.solvers.constraints import get_full_revolute_nearest_point


def greedy_set_cover(cliques: set, targets) -> set:
    covering_cliques = set()
    targets_remaining = set(targets)
    while len(targets_remaining) > 0:
        n_covered = 0
        best_clique = None
        for clique in cliques.difference(covering_cliques):
            if len(clique.intersection(targets_remaining)) > n_covered:
                best_clique = clique
                n_covered = len(clique.intersection(targets_remaining))
        covering_cliques.add(best_clique)
        targets_remaining = targets_remaining.difference(best_clique)

    return covering_cliques


def augment_square_matrix(A:np.ndarray, d: int) -> np.ndarray:
    assert A.shape[0] == A.shape[1]
    A_aug = np.zeros((A.shape[0] + d, A.shape[0]+d))
    A_aug[0:A.shape[0], 0:A.shape[0]] = A
    A_aug[-d:, -d:] = np.eye(d)
    return A_aug


def linear_matrix_equality(i: int, j: int, n_vars: int) -> np.ndarray:
    """
    Convert a distance constraint into a LME for internal variables (no constant so no linear term).
    """
    A = np.zeros((n_vars, n_vars))
    A[i, i] = 1.
    A[j, j] = 1.
    A[i, j] = -1.
    A[j, i] = -1.
    return A


def linear_matrix_equality_with_anchor(i: int, n_vars: int, ee: np.ndarray) -> np.ndarray:
    """
    Convert a distance constraint into a LME for an internal variable and a constant end-effector (needs homog. vars).
    """
    A = np.zeros((n_vars, n_vars))
    d = len(ee)
    A[i, i] = 1.
    A[i, -d:] = -ee
    A[-d:, i] = -ee
    return A


def distance_clique_linear_map(graph: nx.Graph, clique: frozenset,
                               ee_assignments: dict = None, ee_cost: bool = False) -> (list, list, dict, bool):
    """
    Produce a set of SDP-relaxed constraints in the form of A: S^n -> R^m (with constant values b) for a clique of
    variables in a DG problem.

    """
    ees_clique = [key for key in ee_assignments if key in clique]
    d = 0 if len(ees_clique) == 0 else len(list(ee_assignments.values())[0])
    n_ees = len(ees_clique)
    n_vars = len(clique) if ee_cost else len(clique) - n_ees  # If not using ee's in the cost, they are not variables
    n = n_vars + d  # The SDP needs a homogenizing identity matrix for linear terms involving ees

    # Linear map from S^n -> R^m
    A = []
    b = []
    A_mapping = {}
    constraint_idx = 0
    var_idx = 0
    for u in clique:  # Populate the index mapping
        if u not in ees_clique or ee_cost:
            A_mapping[u] = var_idx
            var_idx += 1
    assert n_vars == var_idx, print(f"n_vars:{n_vars}, var_idx:{var_idx}")
    for u in clique:
        for v in clique:
            #   TODO: factor out this whole loop
            if (u, v) in graph.edges and frozenset((u, v)) not in A_mapping:
                if not ee_cost:
                    if u in ees_clique and v in ees_clique:  # Don't need the const. dist between two assigned end-effectors
                        continue
                    elif u in ees_clique:
                        A_uv = linear_matrix_equality_with_anchor(A_mapping[v], n, ee_assignments[u])
                        b_uv = graph[u][v]['weight']**2 - ee_assignments[u].dot(ee_assignments[u])
                    elif v in ees_clique:
                        A_uv = linear_matrix_equality_with_anchor(A_mapping[u], n, ee_assignments[v])
                        b_uv = graph[u][v]['weight']**2 - ee_assignments[v].dot(ee_assignments[v])
                    else:
                        A_uv = linear_matrix_equality(A_mapping[u], A_mapping[v], n)
                        b_uv = graph[u][v]['weight']**2

                else:
                    A_uv = linear_matrix_equality(A_mapping[u], A_mapping[v], n)
                    b_uv = graph[u][v]['weight']**2  # TODO: squared or not? I forget
                A.append(A_uv)
                b.append(b_uv)
                A_mapping[frozenset((u, v))] = constraint_idx
                constraint_idx += 1
    return A, b, A_mapping, d > 0


def distance_constraints(robot: RobotRevolute, end_effectors: dict, sparse: bool=False,
                         ee_cost: bool=False) -> (np.ndarray, dict):
    """
    Produce an SDP-relaxed linear mapping for the equality constraints describing our DG problem instance.

    :param robot:
    :param end_effectors: dict of end-effector assignments
    :param sparse:
    :param ee_cost: whether to use end-effectors in the cost function (as opposed to constraints)
    :return: linear mapping and variable mapping
    """
    undirected = nx.Graph(robot.structure_graph())
    equality_cliques = nx.chordal_graph_cliques(undirected)

    if not sparse:
        full_set = frozenset()
        for clique in equality_cliques:
            full_set = full_set.union(clique)
        equality_cliques = [full_set]
    clique_dict = {}
    for clique in equality_cliques:
        clique_dict[clique] = distance_clique_linear_map(undirected, clique, end_effectors, ee_cost)

    return clique_dict


def evaluate_linear_map(clique: frozenset, A:list, b: list, mapping: dict, input_vals: dict) -> list:
    """
    Evaluate the linear map given by A, b, mapping over the variables in clique for input_vals.
    """
    d = len(list(input_vals.values())[0])
    n_vars = len(mapping) - len(A)

    X = np.zeros((d, n_vars))
    for var in clique:
        if var in mapping:
            X[:, mapping[var]] = input_vals[var]
    if A[0].shape[0] != n_vars:
        # assert d == A[0].shape[0] - n, print(f"len(A): {A[0].shape[0]}, n:{n}")
        X = np.hstack([X, np.eye(d)])
    Z = X.T@X
    output = [np.trace(A[idx]@Z) - b[idx] for idx in range(len(A))]

    return output


def constraint_clique_dict_to_sdp(constraint_clique_dict: dict, nearest_points: dict):
    """

    :param constraints_clique_dict: output of distance_constraints function
    :param nearest_points: defines the cost function with squared distances
    """
    sdp_variable_map = {}
    sdp_constraints_map = {}
    sdp_cost_map = {}
    # Need to cover the cost ee's with augmented cliques (for linear terms)
    d = len(list(nearest_points.values())[0])

    # Prepare the set cover problem
    targets_to_cover = list(nearest_points.keys())
    cliques_remaining = set()
    for clique in constraint_clique_dict:
        _, _, _, is_augmented = constraint_clique_dict[clique]
        if is_augmented:
            for joint in clique:
                if joint in targets_to_cover:
                    targets_to_cover.remove(joint)
        else:
            cliques_remaining.add(clique)
    # Only need to augment non-zero nearest points (important for nuclear norm!)
    for target in nearest_points.keys():
        if target in targets_to_cover and np.all(nearest_points[target] == np.zeros(d)):
            targets_to_cover.remove()
    # Solve the set cover problem and augment the needed cliques to accommodate linear terms
    cliques_to_cover = greedy_set_cover(cliques_remaining, targets_to_cover)
    for clique in constraint_clique_dict:
        if clique in cliques_to_cover:
            for idx, A_matrix in enumerate(constraint_clique_dict[clique][0]):
                constraint_clique_dict[clique][0][idx] = augment_square_matrix(A_matrix, d)

            # Replace augmented variable (sloppy)
            new_tuple = constraint_clique_dict[clique][0], constraint_clique_dict[clique][1],\
                        constraint_clique_dict[clique][2], True
            constraint_clique_dict[clique] = new_tuple

    remaining_nearest_points = list(nearest_points.keys())
    for clique in constraint_clique_dict:
        A, b, mapping, is_augmented = constraint_clique_dict[clique]
        # Construct the SDP variable and constraints
        Z_clique = cp.Variable(A[0].shape, PSD=True)
        sdp_variable_map[clique] = Z_clique
        constraints_clique = [cp.trace(A[idx]@Z_clique) == b[idx] for idx in range(len(A))]
        if is_augmented:
            constraints_clique += [Z_clique[-d:, -d:] == np.eye(d)]
            # Construct the cost function
            C_clique = []
            for joint in clique:
                if joint in remaining_nearest_points:
                    # if not np.all(nearest_points[ee] == np.zeros(d)):
                    C = np.zeros(A[0].shape)
                    C[mapping[joint], mapping[joint]] = 1.
                    if is_augmented and np.any(nearest_points[joint] != np.zeros(d)):
                        C[mapping[joint], -d:] = nearest_points[joint]
                        C[-d:, mapping[joint]] = nearest_points[joint]
                    C_clique.append(C)
                    remaining_nearest_points.remove(joint)
            if len(C_clique) > 0:
                sdp_cost_map[clique] = C_clique
        sdp_constraints_map[clique] = constraints_clique
    assert len(remaining_nearest_points) == 0
    return sdp_variable_map, sdp_constraints_map, sdp_cost_map


def form_sdp_problem(constraint_clique_dict, sdp_variable_map, sdp_constraints_map, sdp_cost_map, d) -> cp.Problem:
    constraints = [cons for cons_clique in sdp_constraints_map.values() for cons in cons_clique]
    # Link up the repeated values via equality constraints
    seen_vars = {}
    for clique in constraint_clique_dict:
        _, _, mapping, is_augmented = constraint_clique_dict[clique]

        # Find overlap (all shared variables) and equate their cross terms as well
        overlapping_vars = list(set(mapping.keys()).intersection(seen_vars.keys()))  # Overlap of clique's vars and seen_vars
        overlapping_vars_cliques = [seen_vars[var] for var in overlapping_vars]  # Cliques of seen_vars that overlap
        assert len(np.unique(overlapping_vars_cliques)) <= 1  # Assert that 1 or 0 cliques exist
        if len(np.unique(overlapping_vars_cliques)) == 1:
            # Add the linking equalities
            target_clique = overlapping_vars_cliques[0]
            _, _, target_mapping, target_is_augmented = constraint_clique_dict[target_clique]
            Z = sdp_variable_map[clique]
            Z_target = sdp_variable_map[target_clique]
            for idx, var1 in enumerate(overlapping_vars):
                for jdx, var2 in enumerate(overlapping_vars[idx:]):  # Don't double count
                    if idx == jdx:
                        constraints += [Z[mapping[var1], mapping[var1]] ==
                                        Z_target[target_mapping[var1], target_mapping[var1]]]
                        if is_augmented and target_is_augmented:
                            constraints += [Z[mapping[var1], -d:] == Z_target[-d:, target_mapping[var1]]]
                    else:
                        constraints += [
                            Z[mapping[var1], mapping[var2]] == Z_target[target_mapping[var1], target_mapping[var2]],
                            Z[mapping[var2], mapping[var1]] == Z_target[target_mapping[var2], target_mapping[var1]]
                        ]
        # Add all vars to seen_vars
        for var in clique:
            if var not in seen_vars:
                seen_vars[var] = clique

    # Convert constraint matrices to cvxpy cost function
    cost = 0.
    for clique in sdp_cost_map:
        C_list = sdp_cost_map[clique]
        C_clique = 0.
        for C in C_list:
            C_clique += C
        cost += cp.trace(C@sdp_variable_map[clique])

    return cp.Problem(cp.Minimize(cost), constraints)


if __name__ == '__main__':
    # TODO: unit test on random q's with all 4 combos of sparse and ee_cost values for UR10
    sparse = False  # Whether to exploit chordal sparsity in the SDP formulation
    ee_cost = False  # Whether to treat the end-effectors as variables with targets in the cost

    # robot, graph = load_ur10()
    n = 2
    dof = n
    a_full = [0, -0.612, -0.5723, 0, 0, 0]
    d_full = [0.1273, 0, 0, 0.1639, 0.1157, 0.0922]
    al_full = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]
    th_full = [0, 0, 0, 0, 0, 0]
    a = a_full[0:n]
    d = d_full[0:n]
    al = al_full[0:n]
    th = th_full[0:n]
    ub = (np.pi) * np.ones(n)
    lb = -ub
    ub = np.minimum(np.random.rand(n) * (np.pi / 2) + np.pi / 2, np.pi)
    lb = -ub
    modified_dh = False
    params = {
        "a": a[:n],
        "alpha": al[:n],
        "d": d[:n],
        "theta": th[:n],
        "lb": lb[:n],
        "ub": ub[:n],
        "modified_dh": modified_dh,
    }
    robot = RobotRevolute(params)
    graph = RobotRevoluteGraph(robot)

    q = robot.random_configuration()
    full_points = [f'p{idx}' for idx in range(0, graph.robot.n + 1)] + \
                  [f'q{idx}' for idx in range(0, graph.robot.n + 1)]
    input_vals = get_full_revolute_nearest_point(graph, q, full_points)
    end_effectors = {key: input_vals[key] for key in ['p0', 'q0', f'p{robot.n}', f'q{robot.n}']}

    constraint_clique_dict = distance_constraints(robot, end_effectors, sparse, ee_cost)
    A, b, mapping, _ = list(constraint_clique_dict.values())[0]

    for clique in constraint_clique_dict:
        A, b, mapping, _ = constraint_clique_dict[clique]
        print(evaluate_linear_map(clique, A, b, mapping, input_vals))

    # Make cost function stuff
    interior_nearest_points = {key: input_vals[key] for key in input_vals if key not in ['p0', 'q0', f'p{robot.n}', f'q{robot.n}']}
    sdp_variable_map, sdp_constraints_map, sdp_cost_map = constraint_clique_dict_to_sdp(constraint_clique_dict,
                                                                                        interior_nearest_points)

    prob = form_sdp_problem(constraint_clique_dict, sdp_variable_map, sdp_constraints_map, sdp_cost_map, 3)

    sol = prob.solve(verbose=True, solver='CVXOPT')
