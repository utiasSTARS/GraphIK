#!/usr/bin/env python3
import numpy as np
import networkx as nx
import math

LOWER = "lower_limit"
UPPER = "upper_limit"
BOUNDED = "bounded"
DIST = "weight"
POS = "pos"
ROOT = "p0"
UNDEFINED = None


def dist_to_gram(D):
    # TODO rename to gram_from_distance_matrix
    J = np.identity(D.shape[0]) - (1 / (D.shape[0])) * np.ones(D.shape)
    G = -0.5 * J @ D @ J  # Gram matrix
    return G


def distance_matrix_from_graph(G: nx.DiGraph) -> np.ndarray:
    if not isinstance(G, nx.DiGraph):
        raise TypeError("Input must a DiGraph.")

    selected_edges = [(u, v) for u, v, d in G.edges(data=True) if DIST in d]
    return (
        nx.to_numpy_array(
            nx.to_undirected(G.edge_subgraph(selected_edges)), weight=DIST
        )
        ** 2
    )


def adjacency_matrix_from_graph(G: nx.DiGraph) -> np.ndarray:
    """
    Returns the adjacency matrix representing the edges that are known,
    given the kinematic and base structure, as well as the end-effector targets.
    :returns: Adjacency matrix
    """
    if not isinstance(G, nx.DiGraph):
        raise TypeError("Input must a DiGraph.")

    selected_edges = [(u, v) for u, v, d in G.edges(data=True) if DIST in d]
    return nx.to_numpy_array(
        nx.to_undirected(G.edge_subgraph(selected_edges)), weight=""
    )


def pos_from_graph(G: nx.DiGraph) -> np.ndarray:
    """
    Returns an n x m matrix of node positions from a given graph,
    where n is the number of nodes and m is the point dimension.
    :param G: graph where all nodes have a populated POS field
    :returns: n x m matrix of node positions
    """
    # TODO add check to see if all POS defined
    # X = np.zeros([len(G), self.dim])  # matrix of vertex positions
    X = []
    for idx, name in enumerate(G):
        X += [list(G.nodes[name][POS])]
    return np.array(X)


def graph_complete_edges(G: nx.DiGraph) -> nx.DiGraph:
    """
    Given a graph with all defined node positions, calculate all unknown edges.
    :param G: Graph with some unknown edges
    :returns: Graph with all known edges
    """

    for idx, u in enumerate(G.nodes()):
        for jdx, v in enumerate(G.nodes()):
            # if both nodes have known positions
            if (POS in G.nodes[u]) and (POS in G.nodes[v]) and jdx > idx:
                # if a distance edge exists already in the other direction
                if G.has_edge(v, u):
                    if DIST in G[v][u]:
                        continue
                d = np.linalg.norm(G.nodes[u][POS] - G.nodes[v][POS])
                G.add_edges_from(
                    [
                        (u, v, {DIST: d}),
                        (u, v, {LOWER: d}),
                        (u, v, {UPPER: d}),
                    ]
                )

    return G


def factor(A):
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
def MDS(B, eps=1e-5):
    n = B.shape[0]
    x = factor(B)
    (evals, evecs) = np.linalg.eigh(x)
    K = len(evals[evals > eps])
    if K < n:
        # only first K columns
        x = x[:, 0:K]
    return x


def linear_projection(P, F, dim):
    S = 0
    I = np.nonzero(F)
    for kdx in range(len(I[0])):
        idx = I[0][kdx]
        jdx = I[1][kdx]
        S += np.outer(P[idx, :] - P[jdx, :], P[idx, :] - P[jdx, :])

    eigval, eigvec = np.linalg.eigh(S)
    return P @ np.fliplr(eigvec)[:, :dim]


def linear_projection_randomized(P, F, dim):
    S = 0
    I = np.nonzero(F)
    for kdx in range(len(I[0])):
        idx = I[0][kdx]
        jdx = I[1][kdx]
        S += np.outer(P[idx, :] - P[jdx, :], P[idx, :] - P[jdx, :])

    eigval, eigvec = np.linalg.eigh(S)
    ev = np.fliplr(eigvec)
    q, r = np.linalg.qr(ev[:, :dim])
    U = q
    # U = q[:, :dim]
    return P @ U


## sample distance matrix
def sample_matrix(lower_limit, upper_limit):
    m, n = lower_limit.shape
    return lower_limit + np.random.rand(m, n) * (upper_limit - lower_limit)
    # return lower_limit + np.random.normal(75.0, 0.25, (m, n)) * (
    #     upper_limit - lower_limit
    # )
