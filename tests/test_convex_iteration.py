"""Argument-handling tests for convex_iterate_sdp_snl_graph.

max_iters=0 exercises all the setup (anchor collection, constraint
construction) without running any SDP solve.
"""
import numpy as np

from graphik.solvers.convex_iteration import convex_iterate_sdp_snl_graph
from graphik.utils.constants import POS
from graphik.utils.roboturdf import load_truncated_ur10


def test_caller_anchors_dict_is_not_mutated():
    robot, graph = load_truncated_ur10(3)
    # An obstacle node carries a POS attribute, so the anchor-collection
    # loop would add it to the caller's dict if anchors weren't copied.
    graph.add_spherical_obstacle("o0", np.array([10.0, 10.0, 10.0]), 0.5)

    n = robot.n
    anchors = {
        "p0": graph.nodes["p0"][POS],
        "q0": graph.nodes["q0"][POS],
        f"p{n}": np.array([0.5, 0.5, 0.5]),
        f"q{n}": np.array([0.5, 0.5, 1.5]),
    }
    keys_before = set(anchors)

    convex_iterate_sdp_snl_graph(graph, anchors, max_iters=0)

    assert set(anchors) == keys_before
