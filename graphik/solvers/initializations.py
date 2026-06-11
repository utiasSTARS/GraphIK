"""Shared Y-initialization strategies for distance-geometric solvers.

Each returns an ``(N, dim)`` point-cloud factor ordered consistently with the
problem graph's node order. ``zero_init`` is explicitly ordered by
``graph.node_ids``; the others inherit ordering from ``D_goal`` / ``omega``.
"""
from __future__ import annotations

import numpy as np

from graphik.utils import MDS, POS, gram_from_distance_matrix, linear_projection
from graphik.utils.dgp import bound_smoothing

INIT_STRATEGIES = ("spectral", "bsmooth", "zero")


def spectral_init(D_goal, omega, dim):
    """Partial-Gram spectral initialization with an eigenvalue floor."""
    G = gram_from_distance_matrix(omega * D_goal)
    eigvals, eigvecs = np.linalg.eigh(G)
    top = np.argsort(-eigvals)[:dim]
    lam = np.maximum(eigvals[top], 1e-12)
    return eigvecs[:, top] * np.sqrt(lam)


def _sample_bounds_init(lb, ub, omega, dim):
    """Sample an EDM at 0.9 of (lb, ub), MDS, then project to ``dim``."""
    lb_sqrt = np.sqrt(lb)
    ub_sqrt = np.sqrt(ub)
    D_rand = (lb_sqrt + 0.9 * (ub_sqrt - lb_sqrt)) ** 2
    X_rand = MDS(gram_from_distance_matrix(D_rand), eps=1e-8)
    return linear_projection(X_rand, omega, dim)


def bsmooth_init(G, omega, dim):
    """Bound smoothing plus the legacy sampled-EDM initializer."""
    lb, ub = bound_smoothing(G)
    return _sample_bounds_init(lb, ub, omega, dim)


def zero_init(graph):
    """Use the graph realization at ``robot.zero_configuration()``."""
    G_zero = graph.realization(graph.robot.zero_configuration())
    return np.stack(
        [
            np.asarray(G_zero.nodes[node][POS], dtype=float)
            for node in graph.node_ids
        ]
    )
