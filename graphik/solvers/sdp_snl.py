"""
Rank-{2,3} SDP relaxation tailored to sensor network localization (SNL) applied to our DG/QCQP IK formulation.
"""
import numpy as np
import networkx as nx
import cvxpy as cp

from graphik.utils.roboturdf import load_ur10, load_truncated_ur10
from graphik.robots.robot_base import RobotRevolute
from graphik.graphs.graph_base import RobotRevoluteGraph
from graphik.solvers.constraints import get_full_revolute_nearest_point


def prepare_set_cover_problem(constraint_clique_dict: dict, nearest_points: dict, d: int):
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


def sdp_variables_and_cost(constraint_clique_dict: dict, nearest_points: dict, d:int):
    """

    """
    sdp_variable_map = {}
    sdp_constraints_map = {}
    sdp_cost_map = {}
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
                    if np.any(nearest_points[joint] != np.zeros(d)):
                        C[mapping[joint], -d:] = -nearest_points[joint]
                        C[-d:, mapping[joint]] = -nearest_points[joint]
                        C[-1, -1] = np.linalg.norm(nearest_points[joint])**2  # Add the constant part
                    C_clique.append(C)
                    remaining_nearest_points.remove(joint)
            if len(C_clique) > 0:
                sdp_cost_map[clique] = C_clique
        sdp_constraints_map[clique] = constraints_clique
    assert len(remaining_nearest_points) == 0
    return sdp_variable_map, sdp_constraints_map, sdp_cost_map


def augment_square_matrix(A:np.ndarray, d: int) -> np.ndarray:
    """
    Augment the square matrix A with a d-by-d identity matrix and padding zeros.
    Essentially, returns A_ug = [A 0; 0 eye(d)].

    :param A: square matrix representing a linear map on a symmetric matrix.
    :param d: dimension of the points in the SNL/DG problem instance (2 or 3 for our application)
    :return:
    """
    assert A.shape[0] == A.shape[1]
    A_aug = np.zeros((A.shape[0] + d, A.shape[0]+d))
    A_aug[0:A.shape[0], 0:A.shape[0]] = A
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
    A[i, i] = 1.
    A[j, j] = 1.
    A[i, j] = -1.
    A[j, i] = -1.
    return A


def linear_matrix_equality_with_anchor(i: int, n_vars: int, ee: np.ndarray) -> np.ndarray:
    """
    Convert a distance constraint into a LME for an internal variable and a constant end-effector (needs homog. vars).
    Essentially, we are converting the expression (x_i - ee)**2 into tr(A@Z) where Z = [X eye(d)].T @ [X eye(d)].

    :param i: index of the variable point involved in the LME.
    :param n_vars: number of variable points in the clique of points that contains this distance constraint.
    :param ee: fixed position of the anchor involved
    """
    A = np.zeros((n_vars, n_vars))
    d = len(ee)
    A[i, i] = 1.
    A[i, -d:] = -ee
    A[-d:, i] = -ee
    return A


def constraint_for_variable_pair(u: str, v: str, graph: nx.Graph, index_mapping: dict, ees_clique: list,
                                 ee_assignments: dict, n: int, ee_cost: bool):
    if (u, v) in graph.edges and frozenset((u, v)) not in index_mapping:
        if not ee_cost:
            if u in ees_clique and v in ees_clique:  # Don't need the const. dist between two assigned end-effectors
                return None, None
            elif u in ees_clique:
                A_uv = linear_matrix_equality_with_anchor(index_mapping[v], n, ee_assignments[u])
                b_uv = graph[u][v]['weight'] ** 2 - ee_assignments[u].dot(ee_assignments[u])
            elif v in ees_clique:
                A_uv = linear_matrix_equality_with_anchor(index_mapping[u], n, ee_assignments[v])
                b_uv = graph[u][v]['weight'] ** 2 - ee_assignments[v].dot(ee_assignments[v])
            else:
                A_uv = linear_matrix_equality(index_mapping[u], index_mapping[v], n)
                b_uv = graph[u][v]['weight'] ** 2

        else:
            A_uv = linear_matrix_equality(index_mapping[u], index_mapping[v], n)
            b_uv = graph[u][v]['weight'] ** 2
        return A_uv, b_uv
    else:
        return None, None


def distance_clique_linear_map(graph: nx.Graph, clique: frozenset,
                               ee_assignments: dict = None, ee_cost: bool = False) -> (list, list, dict, bool):
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
    n_vars = len(clique) if ee_cost else len(clique) - n_ees  # If not using ee's in the cost, they are not variables
    n = n_vars + d  # The SDP needs a homogenizing identity matrix for linear terms involving ees

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
            A_uv, b_uv = constraint_for_variable_pair(u, v, graph, index_mapping, ees_clique,
                                                      ee_assignments, n, ee_cost)
            if A_uv is not None:
                A.append(A_uv)
                b.append(b_uv)
                index_mapping[frozenset((u, v))] = constraint_idx
                constraint_idx += 1
    return A, b, index_mapping, d > 0


def distance_constraints(robot: RobotRevolute, end_effectors: dict, sparse: bool=False,
                         ee_cost: bool=False) -> dict:
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
    undirected = nx.Graph(robot.structure_graph())  # This graph must be chordal
    equality_cliques = nx.chordal_graph_cliques(undirected)  # Returns maximal cliques (in spite of name)

    if not sparse:
        full_set = frozenset()
        for clique in equality_cliques:
            full_set = full_set.union(clique)
        equality_cliques = [full_set]
    clique_dict = {}
    for clique in equality_cliques:
        clique_dict[clique] = distance_clique_linear_map(undirected, clique, end_effectors, ee_cost)

    return clique_dict


def evaluate_linear_map(clique: frozenset, A:list, b: list, mapping: dict, input_vals: dict, n_vars: int = None) -> list:
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
    Z = X.T@X
    output = [np.trace(A[idx]@Z) - b[idx] for idx in range(len(A))]

    return output


def evaluate_cost(constraint_clique_dict: dict, sdp_cost_map: dict, nearest_points: dict):
    """
    Evaluate a cost function defined by sdp_cost map for points in nearest_points.
    """
    cost = 0.
    for clique in sdp_cost_map:
        A, _, mapping, _ = constraint_clique_dict[clique]
        n_vars = len(mapping) - len(A)
        C_list = sdp_cost_map[clique]
        b_list = [0. for _ in C_list]  # The constant part is embedded in the augmented A's final entry (A[-1, -1]).
        cost += sum(evaluate_linear_map(clique, C_list, b_list, mapping, nearest_points, n_vars))
    return cost


def constraints_and_nearest_points_to_sdp_vars(constraint_clique_dict: dict, nearest_points: dict, d: int):
    """
    Takes in a dictionary of clique-indexed constraints and a dictionary of nearest-points for a cost function and
    produces SDP variables (cvxpy) that correspond to the constraint LMEs as well as LM cost expressions.

    :param constraints_clique_dict: output of distance_constraints function
    :param nearest_points: defines the cost function with squared distances
    :param d: int representing the dimension of the point variables (2 or 3 for IK)
    :return:
    """
    # Prepare the set cover problem: we only want to augment the minimal set of cliques required to cover all ee's
    cliques_remaining, targets_to_cover = prepare_set_cover_problem(constraint_clique_dict, nearest_points, d)

    # Solve the set cover problem and augment the needed cliques to accommodate linear terms
    cliques_to_cover = greedy_set_cover(cliques_remaining, targets_to_cover)

    # Augment the cliques our cover has told us need to be augmented
    for clique in constraint_clique_dict:
        if clique in cliques_to_cover:
            for idx, A_matrix in enumerate(constraint_clique_dict[clique][0]):
                constraint_clique_dict[clique][0][idx] = augment_square_matrix(A_matrix, d)
            # Replace augmented variable (sloppy)
            new_tuple = constraint_clique_dict[clique][0], constraint_clique_dict[clique][1],\
                        constraint_clique_dict[clique][2], True
            constraint_clique_dict[clique] = new_tuple

    # Construct the cost and SDP variables
    sdp_variable_map, sdp_constraints_map, sdp_cost_map = sdp_variables_and_cost(constraint_clique_dict,
                                                                                 nearest_points, d)
    return sdp_variable_map, sdp_constraints_map, sdp_cost_map


def form_sdp_problem(constraint_clique_dict: dict, sdp_variable_map: dict, sdp_constraints_map:
                     dict, sdp_cost_map: dict, d: int) -> cp.Problem:
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

    # Convert cost matrices to cvxpy cost function
    cost = lme_to_cvxpy_cost(sdp_cost_map, sdp_variable_map)
    return cp.Problem(cp.Minimize(cost), constraints)


def lme_to_cvxpy_cost(sdp_cost_map: dict, sdp_variable_map: dict):
    """
    Convert a mapping from cliques to cost matrices and variables to a cvxpy cost function.
    """
    cost = 0.
    for clique in sdp_cost_map:
        C_list = sdp_cost_map[clique]
        C_clique = 0.
        for C in C_list:
            C_clique += C
        cost += cp.trace(C @ sdp_variable_map[clique])
    return cost


def extract_solution(constraint_clique_dict: dict, sdp_variable_map: dict, d: int) -> dict:
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


if __name__ == '__main__':
    # Simple examples
    sparse = False  # Whether to exploit chordal sparsity in the SDP formulation
    ee_cost = False  # Whether to treat the end-effectors as variables with targets in the cost.
                     # If False, end-effectors are NOT variables (they are baked in to constraints as parameters)

    # Full UR10 convenience
    # robot, graph = load_ur10()

    # Truncated UR10 (only the first n joints)
    n = 6
    if n == 6:
        robot, graph = load_ur10()
    else:
        robot, graph = load_truncated_ur10(n)

    # Generate a random feasible target
    q = robot.random_configuration()

    # Extract the positions of the points
    full_points = [f'p{idx}' for idx in range(0, graph.robot.n + 1)] + \
                  [f'q{idx}' for idx in range(0, graph.robot.n + 1)]
    input_vals = get_full_revolute_nearest_point(graph, q, full_points)

    # End-effectors are 'generalized' to include the base pair ('p0', 'q0')
    end_effectors = {key: input_vals[key] for key in ['p0', 'q0', f'p{robot.n}', f'q{robot.n}']}

    # Form the constraints
    constraint_clique_dict = distance_constraints(robot, end_effectors, sparse, ee_cost)
    A, b, mapping, _ = list(constraint_clique_dict.values())[0]

    # Different cost function options here - cost function is controlled by a dictionary mapping some subset of the keys
    # of input_vals (all the points' positions) to some nearest point.

    # Nuclear norm - the nearest points are all zero
    nearest_points_nuclear = {key: np.zeros(robot.dim)
                               for key in input_vals if key not in ['p0', 'q0', f'p{robot.n}', f'q{robot.n}']}
    sdp_variable_map_nuclear, sdp_constraints_map_nuclear, sdp_cost_map_nuclear = \
        constraints_and_nearest_points_to_sdp_vars(constraint_clique_dict, nearest_points_nuclear, robot.dim)
    prob_nuclear = form_sdp_problem(constraint_clique_dict, sdp_variable_map_nuclear, sdp_constraints_map_nuclear, sdp_cost_map_nuclear, robot.dim)
    prob_nuclear.solve(verbose=True, solver='CVXOPT')
    # Analysis below assumes dense (sparse = False) case
    Z_nuclear = list(sdp_variable_map_nuclear.values())[0].value
    _, s_nuclear, _ = np.linalg.svd(Z_nuclear)
    solution_rank_nuclear = np.linalg.matrix_rank(Z_nuclear, tol=1e-6, hermitian=True)
    solution_nuclear = extract_solution(constraint_clique_dict, sdp_variable_map_nuclear, robot.dim)

    # Feasibility (no nearest points means cost function is 0)
    no_nearest_points = {}
    sdp_variable_map, sdp_constraints_map, sdp_cost_map = \
        constraints_and_nearest_points_to_sdp_vars(constraint_clique_dict, no_nearest_points, robot.dim)
    prob_feas = form_sdp_problem(constraint_clique_dict, sdp_variable_map, sdp_constraints_map, sdp_cost_map, robot.dim)
    prob_feas.solve(verbose=True, solver='CVXOPT')
    # Analysis below assumes dense (sparse = False) case
    Z = list(sdp_variable_map.values())[0].value
    _, s, _ = np.linalg.svd(Z)
    solution_rank = np.linalg.matrix_rank(Z, tol=1e-6, hermitian=True)
    solution = extract_solution(constraint_clique_dict, sdp_variable_map, robot.dim)

    # Exact nearest point - use the true value from q (don't perturb)
    exact_nearest_points = {key: input_vals[key]
                                for key in input_vals if key not in ['p0', 'q0', f'p{robot.n}', f'q{robot.n}']}
    sdp_variable_map_exact, sdp_constraints_map_exact, sdp_cost_map_exact = \
        constraints_and_nearest_points_to_sdp_vars(constraint_clique_dict, exact_nearest_points, robot.dim)
    prob_exact = form_sdp_problem(constraint_clique_dict, sdp_variable_map_exact, sdp_constraints_map_exact,
                                  sdp_cost_map_exact, robot.dim)
    prob_exact.solve(verbose=True, solver='CVXOPT')
    Z_exact = list(sdp_variable_map_exact.values())[0].value
    _, s_exact, _ = np.linalg.svd(Z_exact)
    solution_rank_exact = np.linalg.matrix_rank(Z_exact, tol=1e-6, hermitian=True)
    solution_exact = extract_solution(constraint_clique_dict, sdp_variable_map_exact, robot.dim)


    total_error_exact = 0.
    total_error_nuclear = 0.
    total_error_feas = 0.
    for key in solution_exact:
        print(f"{key}")
        print(f"True value:          {input_vals[key]}")
        print(f"Nearest point value: {solution_exact[key]}")
        print(f"Nuclear norm value:  {solution_nuclear[key]}")
        print(f"Feasibility value:   {solution[key]}")
        print("------------------------------------------------------------------------")
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

    # TODO: check constraint violations
    