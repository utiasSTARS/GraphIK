#!/usr/bin/env python3
from typing import List, Optional
import numpy as np
import networkx as nx
import math
from numpy.typing import NDArray
from graphik.utils.constants import *
from graphik.utils.geometry import best_fit_transform


def orthogonal_procrustes(G1: nx.DiGraph, G2: nx.DiGraph) -> nx.DiGraph:
    """
    Aligns two point clouds represented by graphs by aligning nodes with
    matching labels and returns a graph representing the aligned points.

    Keyword Arguments:
    G1: nx.DiGraph -- Graph representing the point set to align with.
    G2: nx.DiGraph -- Graph representing the point set to be aligned.
    """
    Y = pos_from_graph(G1, node_ids=list(G1))
    X = pos_from_graph(G2, node_ids=list(G1))
    R, t = best_fit_transform(X, Y)
    X = pos_from_graph(G2)
    P_e = (R @ X.T + t.reshape(len(t), 1)).T
    return graph_from_pos(P_e, list(G2))  # not really order-dependent


def gram_from_distance_matrix(D: NDArray) -> NDArray:
    """
    Create an NxN Gram matrix from an NxN Distance matrix
    :param D: n x n Distance matrix
    :returns: n x n Gram matrix
    """
    J = np.identity(D.shape[0]) - (1 / (D.shape[0])) * np.ones(D.shape)
    X = -0.5 * J @ D @ J  # Gram matrix
    return X


def distance_matrix_from_gram(X: NDArray) -> NDArray:
    """
    Create an NxN distance matrix from an NxN Gram matrix
    :param X: n x n Gram matrix
    :returns: n x n matrix of distances between nodes
    """
    return (X.diagonal()[:, np.newaxis] + X.diagonal()) - 2 * X


def distance_matrix_from_pos(Y: NDArray) -> NDArray:
    return distance_matrix_from_gram(Y @ Y.T)


def distance_matrix_from_graph(G: nx.Graph, label=DIST, nonedge=0) -> NDArray:
    """
    Returns the distance matrix of the graph, where distance is the attribute with label.
    :returns: Adjacency matrix
    """
    if isinstance(G, nx.DiGraph):
        # if G is a directional graph, ensure both directions have same distance
        G = G.to_undirected(as_view=True)

    return (
        nx.to_numpy_array(G, weight=DIST, nonedge=nonedge, dtype=np.dtype(float)) ** 2
    )


def adjacency_matrix_from_graph(
    G: nx.Graph, label: str = DIST, nodelist: Optional[List] = None
) -> NDArray:
    """
    Returns the adjacency matrix of the graph, but only for edges with label.
    :returns: Adjacency matrix
    """
    if isinstance(G, nx.DiGraph):
        G = G.to_undirected(as_view=True)
    selected_edges = [(u, v) for u, v, d in G.edges(data=True) if label in d]
    return nx.to_numpy_array(
        G.edge_subgraph(selected_edges), weight="", nodelist=nodelist
    )


def pos_from_graph(G: nx.DiGraph, node_ids=None) -> NDArray:
    """
    Returns an n x m matrix of node positions from a given graph,
    where n is the number of nodes and m is the point dimension.
    :param G: graph where all nodes have a populated POS field
    :returns: n x m matrix of node positions
    """
    if not node_ids:
        node_ids = list(G)
    pos = [G.nodes[n][POS] for n in node_ids if POS in G.nodes[n]]
    return np.stack(pos)


def graph_from_pos(
    P: NDArray, node_ids: Optional[List] = [], dist: bool = True
) -> nx.DiGraph:
    """
    Generates an nx.DiGraph object of the subclass type given
    an n x m matrix where n is the number of nodes and m is the dimension.
    Connects all graph nodes.
    :param P: n x m matrix of node positions
    :param node_ids: sequence of nodes in graph
    :returns: graph where all nodes have a populated POS field + edges
    """
    if not node_ids:
        node_ids = ["p" + str(idx) for idx in range(P.shape[0])]
    return graph_from_pos_dict({n: P[idx] for idx, n in enumerate(node_ids)}, dist=dist)


def graph_from_pos_dict(P: dict, dist=True) -> nx.DiGraph:
    """
    Given a dictionary of node name and position key-value pairs,
    generate a graph and fill the POS attributes of
    nodes corresponding to keys with assigned values.
    Then, populate all edges between nodes with assinged POS attributes.
    :param P: a dictionary of node name position pairs
    :returns: graph with connected nodes with POS attribute
    """
    G = nx.empty_graph(P.keys(), create_using=nx.DiGraph)
    nx.set_node_attributes(G, P, POS)

    if dist:
        G = graph_complete_edges(G)

    return G


def graph_complete_edges(
    G: nx.DiGraph, overwrite=False, bidirectional=False
) -> nx.DiGraph:
    """
    Given a graph with some defined node positions, calculate all possible distances.
    If overwrite==True, replace existing distances.
    If bidirectional==True, populate edges in both directions. This is not neccessary for IK
    since methods account for this in distance matrix computation.
    :param G: Graph with some unknown edges
    :returns: Graph with all known edges
    """
    pos = nx.get_node_attributes(G, POS)  # all position attributes
    dst = nx.get_edge_attributes(G, DIST)  # all distance attributes

    names = list(pos.keys())
    edges = []
    for idx, u in enumerate(names):
        for v in names[idx + 1 :]:
            if (((v, u) not in dst) and ((u, v) not in dst)) or overwrite:
                d = np.linalg.norm(pos[u] - pos[v])
                edges += [(u, v, {DIST: d, LOWER: d, UPPER: d})]
                if bidirectional:
                    edges += [(v, u, {DIST: d, LOWER: d, UPPER: d})]

    G.add_edges_from(edges)
    return G


def factor(A: NDArray):
    n = A.shape[0]
    (evals, evecs) = np.linalg.eigh(A)
    evals[evals < 0] = 0  # closest SDP matrix
    X = evecs  # np.transpose(evecs)
    sqrootdiag = np.eye(n)
    for i in range(n):
        sqrootdiag[i, i] = math.sqrt(evals[i])
    X = X.dot(sqrootdiag)
    return np.fliplr(X)


## perform classic Multidimensional scaling
def MDS(B: NDArray, eps: float = 1e-5):
    n = B.shape[0]
    x = factor(B)
    (evals, evecs) = np.linalg.eigh(x)
    K = len(evals[evals > eps])
    if K < n:
        # only first K columns
        x = x[:, 0:K]
    return x


def linear_projection(P: NDArray, F: NDArray, dim):
    S = 0
    I = np.nonzero(F)
    for kdx in range(len(I[0])):
        idx = I[0][kdx]
        jdx = I[1][kdx]
        S += np.outer(P[idx, :] - P[jdx, :], P[idx, :] - P[jdx, :])

    eigval, eigvec = np.linalg.eigh(S)
    return P @ np.fliplr(eigvec)[:, :dim]


## sample distance matrix
def sample_matrix(lower_limit, upper_limit):
    m, n = lower_limit.shape
    return lower_limit + np.random.rand(m, n) * (upper_limit - lower_limit)


def bound_smoothing(G: nx.DiGraph) -> tuple:
    """
    Given a graph with some edges containing upper and lower bounds on distance,
    calculates approximation on lower and upper bounds on all distance matrix elements.
    Distances known exactly correspond to equal lower and upper limits.

    "Distance Geometry Theory, Algorithms and Chemical Applications", Havel, 2002.
    """

    # Generate bipartite graph from two copies of G
    H = nx.DiGraph()

    for u, v, d in G.edges(data=True):
        H.add_edge(u, f"{u}s", weight=0)
        H.add_edge(v, f"{v}s", weight=0)
        H.add_edge(u, f"{v}s", weight=-G[u][v][LOWER])
        H.add_edge(v, f"{u}s", weight=-G[u][v][LOWER])
        H.add_edge(u, v, weight=G[u][v][UPPER])
        H.add_edge(v, u, weight=G[u][v][UPPER])
        H.add_edge(f"{u}s", f"{v}s", weight=G[u][v][UPPER])
        H.add_edge(f"{v}s", f"{u}s", weight=G[u][v][UPPER])

    # Find all shortest paths in bipirtatie graph
    bounds = dict(nx.all_pairs_bellman_ford_path_length(H, weight=DIST))

    N = len(G)
    lower_bounds = np.zeros([N, N])
    upper_bounds = np.zeros([N, N])

    ids = list(G.nodes())
    for u in G:
        for v in G:
            if bounds[u][v + "s"] < 0:
                lower_bounds[ids.index(u), ids.index(v)] = -bounds[u][v + "s"]
            else:
                lower_bounds[ids.index(u), ids.index(v)] = 0
                # lower_bounds[ids.index(u), ids.index(v)] = bounds[u][f"{v}s"]
            upper_bounds[ids.index(u), ids.index(v)] = bounds[u][v]

    return lower_bounds, upper_bounds


def normalize_positions(Y: NDArray, scale=False):
    Y_c = Y - Y.mean(0)
    C = Y_c.T.dot(Y_c)
    e, v = np.linalg.eig(C)
    Y_cr = Y_c.dot(v)
    if scale:
        Y_crs = Y_cr / (1 / abs(Y_cr).max())
        return Y_crs
    else:
        return Y_cr
