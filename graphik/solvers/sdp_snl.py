"""
Rank-{2,3} SDP relaxation tailored to sensor network localization (SNL) applied to our DG/QCQP IK formulation.
"""
import numpy as np
import networkx as nx

from graphik.utils.roboturdf import load_ur10
from graphik.robots.robot_base import RobotRevolute
from graphik.solvers.constraints import get_full_revolute_nearest_point


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
    A[i, -d:] = ee
    A[-d:, i] = ee
    return A


def distance_clique_linear_map(graph: nx.Graph, clique: frozenset,
                               ee_assignments: dict = None, ee_cost: bool = False) -> (list, list, dict):
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
        if u not in ees_clique:
            A_mapping[u] = var_idx
            var_idx += 1
    assert n_vars == var_idx
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
    return A, b, A_mapping


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
    # internal_nodes = [node for node in undirected.nodes() if node not in end_effectors]
    # internal_graph = undirected.subgraph(internal_nodes)
    # internal_equality_cliques = nx.chordal_graph_cliques(internal_graph)
    clique_dict = {}
    for clique in equality_cliques:
        clique_dict[clique] = distance_clique_linear_map(undirected, clique, end_effectors, ee_cost)

    return clique_dict


def evaluate_linear_map(clique: frozenset, A:list, b: list, mapping: dict, input_vals: dict) -> list:
    """
    Evaluate the linear map given by A, b, mapping over the variables in clique for input_vals.
    """
    n = len(clique)
    d = len(list(input_vals.values())[0])
    X = np.zeros((d, 0))
    for var in clique:
        if var in mapping:
            # X[:, mapping[var]] = input_vals[var]
            X = np.hstack([X, input_vals[var][None].T])
    if A[0].shape[0] > n:
        # assert d == A[0].shape[0] - n, print(f"len(A): {A[0].shape[0]}, n:{n}")
        X = np.hstack([X, np.eye(d)])
    Z = X.T@X
    output = [np.trace(A[idx]@Z) - b[idx] for idx in range(len(A))]

    return output


if __name__ == '__main__':
    sparse = True
    ee_cost = False
    robot, graph = load_ur10()
    q = robot.random_configuration()
    full_points = [f'p{idx}' for idx in range(0, graph.robot.n + 1)] + \
                  [f'q{idx}' for idx in range(0, graph.robot.n + 1)]
    input_vals = get_full_revolute_nearest_point(graph, q, full_points)
    end_effectors = {key: input_vals[key] for key in ['p0', 'q0', 'p6', 'q6']}

    constraint_clique_dict = distance_constraints(robot, end_effectors, sparse, ee_cost)
    A, b, mapping = list(constraint_clique_dict.values())[0]

    for clique in constraint_clique_dict:
        A, b, mapping = constraint_clique_dict[clique]
        print(evaluate_linear_map(clique, A, b, mapping, input_vals))
