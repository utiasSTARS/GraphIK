"""
Rank-{2,3} SDP relaxation tailored to sensor network localization (SNL) applied to our DG/QCQP IK formulation.
"""
import numpy as np
import networkx as nx
import cvxpy as cp

from graphik.utils.roboturdf import load_ur10, load_truncated_ur10
from graphik.utils.constants import *
from graphik.utils.chordal import complete_to_chordal_graph
from graphik.robots.robot_base import RobotRevolute
from graphik.graphs.graph_base import RobotGraph
from graphik.solvers.constraints import get_full_revolute_nearest_point
from graphik.solvers.sdp_formulations import SdpSolverParams


def prepare_set_cover_problem(
    constraint_clique_dict: dict, nearest_points: dict, d: int
):
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
    # We also only need to augment non-zero nearest points (important for nuclear norm case)
    for target in nearest_points.keys():
        if target in targets_to_cover and np.all(nearest_points[target] == np.zeros(d)):
            targets_to_cover.remove()

    return cliques_remaining, targets_to_cover


def greedy_set_cover(sources: set, targets: list) -> set:
    """
    Perform a greedy set cover (https://en.wikipedia.org/wiki/Set_cover_problem).
    Returns a subset of the sets in sources such that the union of this subset contains all elements in targets.

    This assumes that

    :param sources: set of sets with which to compute the cover.
    :param targets: list of elements which need to be covered.

    :return: subset of sources whose union covers targets.
    """
    covering_sets = set()
    targets_remaining = set(targets)
    while len(targets_remaining) > 0:
        # Greedily take the next source that covers the most remaining elements in targets_remaining
        n_covered = 0
        best_source = None
        for source in sources.difference(covering_sets):
            if len(source.intersection(targets_remaining)) > n_covered:
                best_source = source
                n_covered = len(source.intersection(targets_remaining))
        if n_covered == 0:
            print("WARNING: Sources do not cover targets. Returning partial cover.")
            return covering_sets
        covering_sets.add(best_source)
        targets_remaining = targets_remaining.difference(best_source)

    return covering_sets


def sdp_variables_and_cost(constraint_clique_dict: dict, nearest_points: dict, d: int):
    """"""
    sdp_variable_map = {}
    sdp_constraints_map = {}
    sdp_cost_map = {}
    remaining_nearest_points = list(nearest_points.keys())
    for clique in constraint_clique_dict:
        A, b, mapping, is_augmented = constraint_clique_dict[clique]
        # Construct the SDP variable and constraints
        Z_clique = cp.Variable(A[0].shape, PSD=True)
        sdp_variable_map[clique] = Z_clique
        constraints_clique = [
            cp.trace(A[idx] @ Z_clique) == b[idx] for idx in range(len(A))
        ]
        if is_augmented:
            constraints_clique += [Z_clique[-d:, -d:] == np.eye(d)]
            # Construct the cost function
            C_clique = []
            for joint in clique:
                if joint in remaining_nearest_points:
                    # if not np.all(nearest_points[ee] == np.zeros(d)):
                    C = np.zeros(A[0].shape)
                    C[mapping[joint], mapping[joint]] = 1.0
                    if np.any(nearest_points[joint] != np.zeros(d)):
                        C[mapping[joint], -d:] = -nearest_points[joint]
                        C[-d:, mapping[joint]] = -nearest_points[joint]
                        C[-1, -1] = (
                            np.linalg.norm(nearest_points[joint]) ** 2
                        )  # Add the constant part
                    C_clique.append(C)
                    remaining_nearest_points.remove(joint)
            if len(C_clique) > 0:
                sdp_cost_map[clique] = C_clique
        sdp_constraints_map[clique] = constraints_clique
    assert len(remaining_nearest_points) == 0
    return sdp_variable_map, sdp_constraints_map, sdp_cost_map


def augment_square_matrix(A: np.ndarray, d: int) -> np.ndarray:
    """
    Augment the square matrix A with a d-by-d identity matrix and padding zeros.
    Essentially, returns A_ug = [A 0; 0 eye(d)].

    :param A: square matrix representing a linear map on a symmetric matrix.
    :param d: dimension of the points in the SNL/DG problem instance (2 or 3 for our application)
    :return:
    """
    assert A.shape[0] == A.shape[1]
    A_aug = np.zeros((A.shape[0] + d, A.shape[0] + d))
    A_aug[0 : A.shape[0], 0 : A.shape[0]] = A
    A_aug[-d:, -d:] = np.eye(d)
    return A_aug


def linear_matrix_equality(i: int, j: int, n_vars: int) -> np.ndarray:
    """
    Convert a Euclidean distance constraint into a LME for internal variables (no constant so no linear term).
    Essentially, we are converting the expression (x_i - x_j)**2 into tr(A@Z) where Z = X.T @ X.

    :param i: index of one of the two points involved in the LME.
    :param j: index of the otehr point involved in the LME.
    :return: square matrix A representing the linear map
    """
    A = np.zeros((n_vars, n_vars))
    A[i, i] = 1.0
    A[j, j] = 1.0
    A[i, j] = -1.0
    A[j, i] = -1.0
    return A


def linear_matrix_equality_with_anchor(
    i: int, n_vars: int, ee: np.ndarray
) -> np.ndarray:
    """
    Convert a distance constraint into a LME for an internal variable and a constant end-effector (needs homog. vars).
    Essentially, we are converting the expression (x_i - ee)**2 into tr(A@Z) where Z = [X eye(d)].T @ [X eye(d)].

    :param i: index of the variable point involved in the LME.
    :param n_vars: number of variable points in the clique of points that contains this distance constraint.
    :param ee: fixed position of the anchor involved
    """
    A = np.zeros((n_vars, n_vars))
    d = len(ee)
    A[i, i] = 1.0
    A[i, -d:] = -ee
    A[-d:, i] = -ee
    return A


def constraint_for_variable_pair(
    u: str,
    v: str,
    graph: nx.Graph,
    index_mapping: dict,
    ees_clique: list,
    ee_assignments: dict,
    n: int,
    ee_cost: bool,
):
    if (u, v) in graph.edges and frozenset((u, v)) not in index_mapping and DIST in graph[u][v].keys():
        if not ee_cost:
            if (
                u in ees_clique and v in ees_clique
            ):  # Don't need the const. dist between two assigned end-effectors
                return None, None
            elif u in ees_clique:
                A_uv = linear_matrix_equality_with_anchor(
                    index_mapping[v], n, ee_assignments[u]
                )
                b_uv = graph[u][v][DIST] ** 2 - ee_assignments[u].dot(
                    ee_assignments[u]
                )
            elif v in ees_clique:
                A_uv = linear_matrix_equality_with_anchor(
                    index_mapping[u], n, ee_assignments[v]
                )
                b_uv = graph[u][v][DIST] ** 2 - ee_assignments[v].dot(
                    ee_assignments[v]
                )
            else:
                A_uv = linear_matrix_equality(index_mapping[u], index_mapping[v], n)
                b_uv = graph[u][v][DIST] ** 2

        else:
            A_uv = linear_matrix_equality(index_mapping[u], index_mapping[v], n)
            b_uv = graph[u][v][DIST] ** 2
        return A_uv, b_uv
    else:
        return None, None


def distance_clique_linear_map(
    graph: nx.Graph,
    clique: frozenset,
    ee_assignments: dict = None,
    ee_cost: bool = False,
) -> (list, list, dict, bool):
    """
    Produce a set of SDP-relaxed constraints in the form of A: S^n -> R^m (with constant values b) for a clique of
    variables in a DG problem.

    :param graph: undirected graph weighted with distances between points.
    :param clique: a maximal clique of graph. We are generating linear matrix equalities (LMEs) corresponding
        to distance constraints between points contained within this clique.
    :param ee_assignments: a dictionary mapping point names (e.g., 'p0', 'q0') to desired targets.
    :param ee_cost: boolean indicating whether the cost function of the SDP we are formulating is made up of squared
        distances between variables and their assignments in ee_assignments. If False, these points are treated as fixed
        parameters with no associated variable.

    :returns: tuple (A, b, index_mapping, is_augmented)
        WHERE
        list A is a list of square np.ndarray's representing the linear matrices in the map from S^n -> R^m
        list b is a list of floats representing the m values in the image of the map S^n -> R^m
        dict index_mapping maps variables (e.g., 'p1', 'q1') in clique to their indices in elements of A, and maps pairs
            (e.g. frozenset(('p1', 'q1'))) to the index of the constraint between them in A and b. For example, to get
            the linear mapping that describes the SDP/LME version of the squared distance constraint between points
            'p0' and 'q1', we write:
                A_p0_q1 = A[frozenset(('p0', 'q1))]; b_p0_q1 = b[frozenset(('p0', 'q1))].
            This is key for keeping track of SDP variables for cvxpy and extracting solutions.
        bool is_augmented indicates whether the matrices in A are augmented via augment_square_matrix() to have a
            d-by-d identity matrix and padding zeros. These are needed for linear terms (as opposed to scalar or
            quadratic) in the constraints described by A and b.
    """
    # Get a list of assigned end-effectors clique
    ees_clique = [key for key in ee_assignments if key in clique]
    # The dimension by which to augment this clique: don't augment it (i.e., d = 0) if there are no ee's in clique
    d = 0 if len(ees_clique) == 0 else len(list(ee_assignments.values())[0])
    n_ees = len(ees_clique)
    n_vars = (
        len(clique) if ee_cost else len(clique) - n_ees
    )  # If not using ee's in the cost, they are not variables
    n = (
        n_vars + d
    )  # The SDP needs a homogenizing identity matrix for linear terms involving ees

    # Linear map from S^n -> R^m
    A = []
    b = []
    index_mapping = {}
    constraint_idx = 0
    var_idx = 0
    for u in clique:  # Populate the index mapping
        if u not in ees_clique or ee_cost:
            index_mapping[u] = var_idx
            var_idx += 1
    assert n_vars == var_idx, print(f"n_vars:{n_vars}, var_idx:{var_idx}")
    for u in clique:
        for v in clique:
            A_uv, b_uv = constraint_for_variable_pair(
                u, v, graph, index_mapping, ees_clique, ee_assignments, n, ee_cost
            )
            if A_uv is not None:
                A.append(A_uv)
                b.append(b_uv)
                index_mapping[frozenset((u, v))] = constraint_idx
                constraint_idx += 1
    return A, b, index_mapping, d > 0


def distance_constraints_graph(
    G: nx.Graph, anchors: dict = {}, sparse: bool = False, ee_cost=None, angle_limits=False
) -> dict:

    G = G.copy()
    if angle_limits:
        # Remove the edges that don't have a
        typ = nx.get_edge_attributes(G, name=BOUNDED)
        edges = []
        for u, v, data in G.edges(data=True):
            if (not data.get(DIST, False) and ("below" not in data[BOUNDED] and "above" not in data[BOUNDED])):
                edges += [(u, v)]

    else:
        # remove the edges that don't have distances defined
        edges = []
        for u, v, data in G.edges(data=True):
            if not data.get(DIST, False):
                edges += [(u, v)]
    G.remove_edges_from(edges)

    undirected = G.to_undirected()
    if not nx.is_chordal(undirected):
        undirected, _ = complete_to_chordal_graph(undirected)
    equality_cliques = nx.chordal_graph_cliques(
        undirected
    )  # Returns maximal cliques (in spite of name)

    if not sparse:
        full_set = frozenset()
        for clique in equality_cliques:
            full_set = full_set.union(clique)
        equality_cliques = [full_set]
    clique_dict = {}
    for clique in equality_cliques:
        clique_dict[clique] = distance_clique_linear_map(
            undirected, clique, anchors, ee_cost
        )
    return clique_dict


def distance_constraints(
    robot: RobotRevolute,
    end_effectors: dict,
    sparse: bool = False,
    ee_cost: bool = False,
) -> dict:
    """
    Produce an SDP-relaxed linear mappings (LMEs) for the equality constraints describing our DG problem instance.
    If sparse, use the maximal cliques to create a sparse relaxation.
    If not sparse, use the full set of variables for a dense or standard SDP relaxation.


    :param robot: robot whose structure we're using
    :param end_effectors: dict of end-effector assignments
    :param sparse: whether to use a chordal decomposition
    :param ee_cost: whether to use end-effectors in the cost function (as opposed to constraints)
    :return: mapping from cliques to LMEs
    """
    undirected = nx.Graph(
        robot.generate_structure_graph()
    )  # This graph must be chordal # NOTE creates a new structure graph?
    equality_cliques = nx.chordal_graph_cliques(
        undirected
    )  # Returns maximal cliques (in spite of name)

    if not sparse:
        full_set = frozenset()
        for clique in equality_cliques:
            full_set = full_set.union(clique)
        equality_cliques = [full_set]
    clique_dict = {}
    for clique in equality_cliques:
        clique_dict[clique] = distance_clique_linear_map(
            undirected, clique, end_effectors, ee_cost
        )

    return clique_dict


def distance_range_constraints(
    G: nx.Graph, constraint_clique_dict: dict, anchors: dict
):
    pairs = []
    dists = []
    upper = []
    for u, v, data in G.edges(data=True):
        if u not in anchors.keys() or v not in anchors.keys():  # If BOTH are anchors we can ignore
            if data.get(BOUNDED, False):
                if "below" in data[BOUNDED]:  #TODO All the bounds are only listed as below, ask Filip!
                    pairs += [frozenset((u, v))]
                    dists += [data[LOWER]]
                    upper += [False]
                if "above" in data[BOUNDED]:
                    pairs += [frozenset((u, v))]
                    dists += [data[UPPER]]
                    upper += [True]

    inequality_map = {}
    for idx, pair in enumerate(pairs):
        dist = dists[idx]
        upper_idx = upper[idx]  # it's either upper or lower
        clique, (A, b) = distance_inequality_constraint(  # TODO: needs anchors to work!
            constraint_clique_dict, pair, dist, upper_idx
        )
        if clique in inequality_map:
            inequality_map[clique] += [(A, b)]
        else:
            inequality_map[clique] = [(A, b)]
    return inequality_map


def evaluate_linear_map(
    clique: frozenset,
    A: list,
    b: list,
    mapping: dict,
    input_vals: dict,
    n_vars: int = None,
) -> list:
    """
    Evaluate the linear map given by A, b, mapping over the variables in clique for input_vals.
    """
    d = len(list(input_vals.values())[0])
    n_vars = len(mapping) - len(A) if n_vars is None else n_vars

    X = np.zeros((d, n_vars))
    for var in clique:
        if var in mapping and var in input_vals:
            X[:, mapping[var]] = input_vals[var]
    if A[0].shape[0] != n_vars:
        # assert d == A[0].shape[0] - n, print(f"len(A): {A[0].shape[0]}, n:{n}")
        X = np.hstack([X, np.eye(d)])
    Z = X.T @ X
    output = [np.trace(A[idx] @ Z) - b[idx] for idx in range(len(A))]

    return output


def evaluate_cost(
    constraint_clique_dict: dict, sdp_cost_map: dict, nearest_points: dict
):
    """
    Evaluate a cost function defined by sdp_cost map for points in nearest_points.
    """
    cost = 0.0
    for clique in sdp_cost_map:
        A, _, mapping, _ = constraint_clique_dict[clique]
        n_vars = len(mapping) - len(A)
        C_list = sdp_cost_map[clique]
        b_list = [
            0.0 for _ in C_list
        ]  # The constant part is embedded in the augmented A's final entry (A[-1, -1]).
        cost += sum(
            evaluate_linear_map(clique, C_list, b_list, mapping, nearest_points, n_vars)
        )
    return cost


def constraints_and_nearest_points_to_sdp_vars(
    constraint_clique_dict: dict, nearest_points: dict, d: int
):
    """
    Takes in a dictionary of clique-indexed constraints and a dictionary of nearest-points for a cost function and
    produces SDP variables (cvxpy) that correspond to the constraint LMEs as well as LM cost expressions.

    :param constraints_clique_dict: output of distance_constraints function
    :param nearest_points: defines the cost function with squared distances
    :param d: int representing the dimension of the point variables (2 or 3 for IK)
    :return:
    """
    # Prepare the set cover problem: we only want to augment the minimal set of cliques required to cover all ee's
    cliques_remaining, targets_to_cover = prepare_set_cover_problem(
        constraint_clique_dict, nearest_points, d
    )

    # Solve the set cover problem and augment the needed cliques to accommodate linear terms
    cliques_to_cover = greedy_set_cover(cliques_remaining, targets_to_cover)

    # Augment the cliques our cover has told us need to be augmented
    for clique in constraint_clique_dict:
        if clique in cliques_to_cover:
            for idx, A_matrix in enumerate(constraint_clique_dict[clique][0]):
                constraint_clique_dict[clique][0][idx] = augment_square_matrix(
                    A_matrix, d
                )
            # Replace augmented variable (sloppy)
            new_tuple = (
                constraint_clique_dict[clique][0],
                constraint_clique_dict[clique][1],
                constraint_clique_dict[clique][2],
                True,
            )
            constraint_clique_dict[clique] = new_tuple

    # Construct the cost and SDP variables
    sdp_variable_map, sdp_constraints_map, sdp_cost_map = sdp_variables_and_cost(
        constraint_clique_dict, nearest_points, d
    )
    return sdp_variable_map, sdp_constraints_map, sdp_cost_map


def constraints_and_linear_cost_to_sdp_vars(
    constraint_clique_dict: dict, C: np.ndarray, canonical_point_order: list, d: int
):
    """
    Takes in a dictionary of clique-indexed constraints and a linear cost function and produces SDP variables (cvxpy)
    that correspond to the constraint LMEs as well as LM cost expressions.

    :param constraints_clique_dict: output of distance_constraints function
    :param C: defines the cost function
    :param canonical_point_order: defines the order of points used to define the cost function in C
    :param d: int representing the dimension of the point variables (2 or 3 for IK)
    :return:
    """
    # Construct the cost and SDP variables
    n = C.shape[0]
    assert n == len(canonical_point_order) + d
    sdp_variable_map = {}
    sdp_constraints_map = {}
    sdp_cost_map = {}

    remaining_pairs = [(i, j) for i in range(n) for j in range(i, n) if C[i, j] != 0.0]

    for clique in constraint_clique_dict:
        A, b, mapping, is_augmented = constraint_clique_dict[clique]
        # Construct the SDP variable and constraints
        Z_clique = cp.Variable(A[0].shape, PSD=True)
        sdp_variable_map[clique] = Z_clique
        constraints_clique = [
            cp.trace(A[idx] @ Z_clique) == b[idx] for idx in range(len(A))
        ]
        if is_augmented:
            constraints_clique += [Z_clique[-d:, -d:] == np.eye(d)]
        sdp_constraints_map[clique] = constraints_clique

        # Construct the cost function
        C_clique = np.zeros(A[0].shape)
        pairs_to_remove = (
            []
        )  # Don't want to remove from the list while iterating over it
        for idx, jdx in remaining_pairs:
            idx_is_var = idx < (n - d)
            jdx_is_var = jdx < (n - d)
            if (not idx_is_var or canonical_point_order[idx] in mapping) and (not jdx_is_var or canonical_point_order[jdx] in mapping):
                u = mapping[canonical_point_order[idx]] if idx < (n - d) else None
                v = mapping[canonical_point_order[jdx]] if jdx < (n - d) else None

                u = A[0].shape[0] - (n - idx) if u is None and is_augmented else u
                v = A[0].shape[0] - (n - jdx) if v is None and is_augmented else v

                if u is not None and v is not None:
                    C_clique[u, v] = C[idx, jdx]
                    C_clique[v, u] = C[jdx, idx]
                    pairs_to_remove.append((idx, jdx))
        for pair in pairs_to_remove:
            remaining_pairs.remove(pair)
        sdp_cost_map[clique] = C_clique
    assert len(remaining_pairs) == 0, "Did not get through all index pairs"

    return sdp_variable_map, sdp_constraints_map, sdp_cost_map


def constraints_and_sparse_linear_cost_to_sdp_vars(
    constraint_clique_dict: dict, C: dict, canonical_point_order: list, d: int
):
    """
    Takes in a dictionary of clique-indexed constraints and a sparse linear cost function and produces SDP variables
    (cvxpy) that correspond to the constraint LMEs as well as LM cost expressions.

    :param constraints_clique_dict: output of distance_constraints function
    :param C: a clique-indexed dictionary that defines the cost function
    :param canonical_point_order: defines the order of points used to define the cost function in C
    :param d: int representing the dimension of the point variables (2 or 3 for IK)
    :return:
    """
    # Construct the cost and SDP variables
    sdp_variable_map = {}
    sdp_constraints_map = {}

    for clique in constraint_clique_dict:
        A, b, mapping, is_augmented = constraint_clique_dict[clique]
        # Construct the SDP variable and constraints
        Z_clique = cp.Variable(A[0].shape, PSD=True)
        sdp_variable_map[clique] = Z_clique
        constraints_clique = [
            cp.trace(A[idx] @ Z_clique) == b[idx] for idx in range(len(A))
        ]
        if is_augmented:
            constraints_clique += [Z_clique[-d:, -d:] == np.eye(d)]
        sdp_constraints_map[clique] = constraints_clique

    return sdp_variable_map, sdp_constraints_map, C


def distance_inequality_constraint(
    constraint_clique_dict: dict,
    point_pair: frozenset,
    distance: float,
    upper_bound: bool,
):
    """
    Output a LMI constraint (A_ineq, b) on a pair of points (both variables, not anchors) representing a lower or upper
    distance bound.

    :param constraint_clique_dict: output of distance_constraints function
    :param point_pair: frozenset of the pair of points that this inequality constrains (e.g, frozenset(('p1', 'p2')))
    :param distance: distance (meters) that limits this point
    :param upper_bound: true if this constraint is an upper bound on distance, false if it's a lower bound
    """
    lower = not upper_bound
    for clique in constraint_clique_dict:
        A, _, index_mapping, _ = constraint_clique_dict[clique]
        if point_pair.issubset(clique):
            u = list(point_pair)[0]
            v = list(point_pair)[1]
            A_ineq = linear_matrix_equality(
                index_mapping[u], index_mapping[v], A[0].shape[0]
            ) * (1 - 2 * lower)
            b = distance ** 2 * (1 - 2 * lower)
            return clique, (A_ineq, b)


def cvxpy_inequality_constraints(sdp_variable_map: dict, inequality_map: dict):
    """"""
    constraints = []
    for clique in inequality_map:
        Z_clique = sdp_variable_map[clique]
        constraints += [
            cp.trace(A @ Z_clique) <= b for (A, b) in inequality_map[clique]
        ]

    return constraints


def chordal_sparsity_overlap_constraints(constraint_clique_dict: dict, sdp_variable_map: dict, d: int):
    # Link up the repeated values via equality constraints
    constraints = []
    seen_vars = {}
    for clique in constraint_clique_dict:
        _, _, mapping, is_augmented = constraint_clique_dict[clique]

        # Find overlap (all shared variables) and equate their cross terms as well
        overlapping_vars = list(
            set(mapping.keys()).intersection(seen_vars.keys())
        )  # Overlap of clique's vars and seen_vars
        overlapping_vars_cliques = [
            seen_vars[var] for var in overlapping_vars
        ]  # Cliques of seen_vars that overlap
        # assert (
        #     len(np.unique(overlapping_vars_cliques)) <= 1
        # )  # Assert that 1 or 0 cliques exist that overlap with the current one (WHY?)
        if len(np.unique(overlapping_vars_cliques)) == 1:
            assert len(constraint_clique_dict) != 1, "Dense case entered sparse code!!"
            # Add the linking equalities
            target_clique = overlapping_vars_cliques[0]
            _, _, target_mapping, target_is_augmented = constraint_clique_dict[
                target_clique
            ]
            Z = sdp_variable_map[clique]
            Z_target = sdp_variable_map[target_clique]
            for idx, var1 in enumerate(overlapping_vars):
                for jdx, var2 in enumerate(
                        overlapping_vars[idx:]
                ):  # Don't double count
                    if idx == jdx:
                        constraints += [
                            Z[mapping[var1], mapping[var1]]
                            == Z_target[target_mapping[var1], target_mapping[var1]]
                        ]
                        if is_augmented and target_is_augmented:
                            constraints += [
                                Z[mapping[var1], -d:]
                                == Z_target[-d:, target_mapping[var1]]
                            ]
                    else:
                        constraints += [
                            Z[mapping[var1], mapping[var2]]
                            == Z_target[target_mapping[var1], target_mapping[var2]],
                            Z[mapping[var2], mapping[var1]]
                            == Z_target[target_mapping[var2], target_mapping[var1]],
                        ]
        # Add all vars to seen_vars
        for var in clique:
            if var not in seen_vars:
                seen_vars[var] = clique

    return constraints


def form_sdp_problem(
    constraint_clique_dict: dict,
    sdp_variable_map: dict,
    sdp_constraints_map: dict,
    sdp_cost_map: dict,
    d: int,
    extra_constraints: list = None,
) -> cp.Problem:
    # Add constraints in cvxpy's form
    constraints = [
        cons for cons_clique in sdp_constraints_map.values() for cons in cons_clique
    ]
    constraints += chordal_sparsity_overlap_constraints(constraint_clique_dict, sdp_variable_map, d)

    # Convert cost matrices to cvxpy cost function
    cost = lme_to_cvxpy_cost(sdp_cost_map, sdp_variable_map)
    if extra_constraints is not None:
        constraints += extra_constraints
    return cp.Problem(cp.Minimize(cost), constraints)


def lme_to_cvxpy_cost(sdp_cost_map, sdp_variable_map: dict):
    """
    Convert a mapping from cliques to cost matrices and variables to a cvxpy cost function.
    """
    cost = 0.0
    for clique in sdp_cost_map:
        C_list = sdp_cost_map[clique]
        C_clique = 0.0
        if type(C_list) == np.ndarray:
            C_clique = C_list
        else:
            for C in C_list:
                C_clique += C
        cost += cp.trace(C_clique @ sdp_variable_map[clique])
    return cost


def extract_solution(
    constraint_clique_dict: dict, sdp_variable_map: dict, d: int
) -> dict:
    solution = {}
    for clique in constraint_clique_dict:
        _, _, mapping, is_augmented = constraint_clique_dict[clique]
        if is_augmented:  # Might not get all the
            Z_clique = sdp_variable_map[clique].value
            X_clique = Z_clique[-d:, 0:-d]
            for var in clique:
                if var in mapping and var not in solution:
                    solution[var] = X_clique[:, mapping[var]]
            # _, s, vh = np.linalg.svd(Z_clique, hermitian=True)
            # clique_sol = np.diag(np.sqrt(s[0:d]))@vh[0:d, :]
            # for var in clique:
            #     if var in mapping and var not in solution:
            #         solution[var] = clique_sol[:, mapping[var]]
    return solution


def extract_full_sdp_solution(
    constraint_clique_dict, canonical_point_order, sdp_variable_map, n, d
):
    Z = np.zeros((n, n))
    for clique in constraint_clique_dict:
        Z_clique = sdp_variable_map[clique].value
        _, _, mapping, is_augmented = constraint_clique_dict[clique]
        for var1 in clique:
            if var1 in mapping.keys():
                idx = mapping[var1]
                u = canonical_point_order.index(var1)
                for var2 in clique:
                    if var2 in mapping.keys():
                        jdx = mapping[var2]
                        v = canonical_point_order.index(var2)
                        Z[u, v] = Z_clique[idx, jdx]
                        Z[v, u] = Z_clique[jdx, idx]
                if is_augmented:
                    Z[u, -d:] = Z_clique[idx, -d:]
                    Z[-d:, u] = Z_clique[-d:, idx]
                    Z[-d:, -d:] = Z_clique[
                        -d:, -d:
                    ]  # Only need to do this once, whatever

    return Z


def solve_nearest_point_sdp(
    nearest_points: dict,
    end_effectors: dict,
    robot,
    sparse=False,
    solver_params=None,
    verbose=False,
    inequality_constraints_map=None,
):
    constraint_clique_dict = distance_constraints(
        robot, end_effectors, sparse, ee_cost=False
    )
    (
        sdp_variable_map,
        sdp_constraints_map,
        sdp_cost_map,
    ) = constraints_and_nearest_points_to_sdp_vars(
        constraint_clique_dict, nearest_points, robot.dim
    )
    if inequality_constraints_map is not None:
        inequality_constraints = cvxpy_inequality_constraints(
            sdp_variable_map, inequality_constraints_map
        )
    else:
        inequality_constraints = None
    prob = form_sdp_problem(
        constraint_clique_dict,
        sdp_variable_map,
        sdp_constraints_map,
        sdp_cost_map,
        robot.dim,
        extra_constraints=inequality_constraints,
    )
    if solver_params is None:
        solver_params = SdpSolverParams()
    prob.solve(verbose=verbose, solver="MOSEK", mosek_params=solver_params.mosek_params)
    # Z_exact = list(sdp_variable_map_exact.values())[0].value
    # _, s_exact, _ = np.linalg.svd(Z_exact)
    # solution_rank_exact = np.linalg.matrix_rank(Z_exact, tol=1e-6, hermitian=True)
    solution = extract_solution(constraint_clique_dict, sdp_variable_map, robot.dim)

    return solution, prob, constraint_clique_dict, sdp_variable_map


def solve_linear_cost_sdp(
    robot,
    end_effectors: dict,
    constraint_clique_dict: dict,
    C,
    canonical_point_order: list = None,
    solver_params=None,
    verbose=False,
    inequality_constraints_map=None,
):
    if canonical_point_order is None:
        full_points = [f"p{idx}" for idx in range(0, robot.n + 1)] + [
            f"q{idx}" for idx in range(0, robot.n + 1)
        ]
        canonical_point_order = [
            point for point in full_points if point not in end_effectors.keys()
        ]

    if type(C) == dict:
        sdp_variable_map, sdp_constraints_map, sdp_cost_map = \
            constraints_and_sparse_linear_cost_to_sdp_vars(constraint_clique_dict, C, canonical_point_order, robot.dim)
    else:
        (
            sdp_variable_map,
            sdp_constraints_map,
            sdp_cost_map,
        ) = constraints_and_linear_cost_to_sdp_vars(
            constraint_clique_dict, C, canonical_point_order, robot.dim
        )

    if inequality_constraints_map is not None:
        inequality_constraints = cvxpy_inequality_constraints(
            sdp_variable_map, inequality_constraints_map
        )
    else:
        inequality_constraints = None
    prob = form_sdp_problem(
        constraint_clique_dict,
        sdp_variable_map,
        sdp_constraints_map,
        sdp_cost_map,
        robot.dim,
        extra_constraints=inequality_constraints,
    )
    if solver_params is None:
        solver_params = SdpSolverParams()
    prob.solve(verbose=verbose, solver="MOSEK", mosek_params=solver_params.mosek_params)
    # Z_exact = list(sdp_variable_map_exact.values())[0].value
    # _, s_exact, _ = np.linalg.svd(Z_exact)
    # solution_rank_exact = np.linalg.matrix_rank(Z_exact, tol=1e-6, hermitian=True)
    solution = extract_solution(constraint_clique_dict, sdp_variable_map, robot.dim)

    return solution, prob, sdp_variable_map, canonical_point_order


def sym_vec(A: np.ndarray) -> np.ndarray:
    vec = []
    for idx in range(A.shape[0]):
        vec_idx = A[idx, idx:]
        vec_idx[1:] = 2.0 * vec_idx[1:]
        vec.append(vec_idx)
    return np.hstack(vec)


def constraints_list_to_matrix(A_list: list):
    a_list = [sym_vec(A) for A in A_list]
    A_symmetrized = np.vstack(a_list)
    # TODO: does this capture a meaningful rank of the linear operator? I believe so!
    return A_symmetrized


if __name__ == "__main__":
    # Simple examples
    sparse = False  # Whether to exploit chordal sparsity in the SDP formulation
    ee_cost = False  # Whether to treat the end-effectors as variables with targets in the cost.
    # If False, end-effectors are NOT variables (they are baked in to constraints as parameters)
    conic_solver = "MOSEK"  # One of "MOSEK", "CVXOPT" for now
    solver_params = (
        SdpSolverParams()
    )  # Use MOSEK settings that worked well for us before
    # Truncated UR10 (only the first n joints)
    n = 6
    if n == 6:
        robot, graph = load_ur10()
    else:
        robot, graph = load_truncated_ur10(n)

    # Generate a random feasible target
    q = robot.random_configuration()

    # Extract the positions of the points
    full_points = [f"p{idx}" for idx in range(0, graph.robot.n + 1)] + [
        f"q{idx}" for idx in range(0, graph.robot.n + 1)
    ]
    input_vals = get_full_revolute_nearest_point(graph, q, full_points)

    # End-effectors are 'generalized' to include the base pair ('p0', 'q0')
    end_effectors = {
        key: input_vals[key] for key in ["p0", "q0", f"p{robot.n}", f"q{robot.n}"]
    }

    # Form the constraints
    constraint_clique_dict = distance_constraints(robot, end_effectors, sparse, ee_cost)
    # A, b, mapping, _ = list(constraint_clique_dict.values())[0]

    # Different cost function options here - cost function is controlled by a dictionary mapping some subset of the keys
    # of input_vals (all the points' positions) to some nearest point.

    # Nuclear norm - the nearest points are all zero
    nearest_points_nuclear = {
        key: np.zeros(robot.dim)
        for key in input_vals
        if key not in ["p0", "q0", f"p{robot.n}", f"q{robot.n}"]
    }
    (
        sdp_variable_map_nuclear,
        sdp_constraints_map_nuclear,
        sdp_cost_map_nuclear,
    ) = constraints_and_nearest_points_to_sdp_vars(
        constraint_clique_dict, nearest_points_nuclear, robot.dim
    )
    prob_nuclear = form_sdp_problem(
        constraint_clique_dict,
        sdp_variable_map_nuclear,
        sdp_constraints_map_nuclear,
        sdp_cost_map_nuclear,
        robot.dim,
    )
    prob_nuclear.solve(
        verbose=True, solver=conic_solver, mosek_params=solver_params.mosek_params
    )
    # Analysis below assumes dense (sparse = False) case
    Z_nuclear = list(sdp_variable_map_nuclear.values())[0].value
    _, s_nuclear, _ = np.linalg.svd(Z_nuclear)
    solution_rank_nuclear = np.linalg.matrix_rank(Z_nuclear, tol=1e-6, hermitian=True)
    solution_nuclear = extract_solution(
        constraint_clique_dict, sdp_variable_map_nuclear, robot.dim
    )

    # Feasibility (no nearest points means cost function is 0)
    no_nearest_points = {}
    (
        sdp_variable_map,
        sdp_constraints_map,
        sdp_cost_map,
    ) = constraints_and_nearest_points_to_sdp_vars(
        constraint_clique_dict, no_nearest_points, robot.dim
    )
    prob_feas = form_sdp_problem(
        constraint_clique_dict,
        sdp_variable_map,
        sdp_constraints_map,
        sdp_cost_map,
        robot.dim,
    )
    prob_feas.solve(
        verbose=True, solver=conic_solver, mosek_params=solver_params.mosek_params
    )
    # Analysis below assumes dense (sparse = False) case
    Z = list(sdp_variable_map.values())[0].value
    _, s, _ = np.linalg.svd(Z)
    solution_rank = np.linalg.matrix_rank(Z, tol=1e-6, hermitian=True)
    solution = extract_solution(constraint_clique_dict, sdp_variable_map, robot.dim)

    # Exact nearest point - use the true value from q (don't perturb)
    exact_nearest_points = {
        key: input_vals[key]
        for key in input_vals
        if key not in ["p0", "q0", f"p{robot.n}", f"q{robot.n}"]
    }
    (
        sdp_variable_map_exact,
        sdp_constraints_map_exact,
        sdp_cost_map_exact,
    ) = constraints_and_nearest_points_to_sdp_vars(
        constraint_clique_dict, exact_nearest_points, robot.dim
    )
    prob_exact = form_sdp_problem(
        constraint_clique_dict,
        sdp_variable_map_exact,
        sdp_constraints_map_exact,
        sdp_cost_map_exact,
        robot.dim,
    )
    prob_exact.solve(
        verbose=True, solver=conic_solver, mosek_params=solver_params.mosek_params
    )
    Z_exact = list(sdp_variable_map_exact.values())[0].value
    _, s_exact, _ = np.linalg.svd(Z_exact)
    solution_rank_exact = np.linalg.matrix_rank(Z_exact, tol=1e-6, hermitian=True)
    solution_exact = extract_solution(
        constraint_clique_dict, sdp_variable_map_exact, robot.dim
    )

    total_error_exact = 0.0
    total_error_nuclear = 0.0
    total_error_feas = 0.0
    for key in solution_exact:
        print(f"{key}")
        print(f"True value:          {input_vals[key]}")
        print(f"Nearest point value: {solution_exact[key]}")
        print(f"Nuclear norm value:  {solution_nuclear[key]}")
        print(f"Feasibility value:   {solution[key]}")
        print(
            "------------------------------------------------------------------------"
        )
        total_error_exact += np.linalg.norm(input_vals[key] - solution_exact[key])
        total_error_nuclear += np.linalg.norm(input_vals[key] - solution_nuclear[key])
        total_error_feas += np.linalg.norm(input_vals[key] - solution[key])

    # Compare the ranks
    print(f"Feasibility formulation rank: {solution_rank}")
    print(f"Nuclear norm rank:            {solution_rank_nuclear}")
    print(f"Exact nearest point rank:     {solution_rank_exact}")

    # Print the total L2 error
    print(f"Total error feas:          {total_error_feas}")
    print(f"Total error nuclear:       {total_error_nuclear}")
    print(f"Total error exact nearest: {total_error_exact}")

    # TODO: check constraint violations! See Filip's code (or just plug this in there)
