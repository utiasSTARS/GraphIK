"""Direct assertions on the anchor (base) subgraph after construction.

The unified ProblemGraph builds a fixed base subgraph with `dim+1`
anchor nodes whose positions are spec'd: 2D gets `{p0=(0,0), x=(-1,0),
y=(0,1)}`; 3D gets `{p0=(0,0,0), x=(L,0,0), y=(0,-L,0), q0=(0,0,L)}`
where L = axis_length. The anchor subgraph forms K_{dim+1} with all
pairwise distances set to the Euclidean distance between anchor
positions.

These tests pin those conventions directly. The previous coverage
(adjacency tests, IK round-trips) relies on these positions implicitly
via downstream computations; here we assert them at the source so a
regression in `_anchor_positions` / `_build_base_subgraph` surfaces with
a precise failure message rather than as a second-order numerical
divergence.
"""
import unittest

import numpy as np
from numpy import pi
from numpy.testing import assert_array_equal

from graphik.graphs import ProblemGraph
from graphik.robots import Robot
from graphik.utils import list_to_variable_dict
from graphik.utils.constants import BASE, BOUNDED, DIST, POS, ROBOT, TYPE


def _make_planar_graph(n=3, axis_length=None):
    params = {
        "link_lengths": list_to_variable_dict(np.ones(n)),
        "theta": list_to_variable_dict(np.zeros(n)),
        "num_joints": n,
    }
    robot = Robot({**params, "dim": 2})
    return ProblemGraph(robot, {"axis_length": axis_length} if axis_length else None)


def _make_revolute_graph(n=2, axis_length=None):
    parents = {f"p{i}": [f"p{i+1}"] for i in range(n)}
    params = {
        "a": {f"p{i}": 1.0 for i in range(1, n + 1)},
        "alpha": {f"p{i}": (pi / 2 if i % 2 == 1 else 0.0) for i in range(1, n + 1)},
        "d": {f"p{i}": 0.0 for i in range(1, n + 1)},
        "theta": {f"p{i}": 0.0 for i in range(1, n + 1)},
        "modified_dh": False,
        "parents": parents,
        "num_joints": n,
    }
    robot = Robot({**params, "dim": 3})
    return ProblemGraph(robot, {"axis_length": axis_length} if axis_length else None)


class TestAnchorConstruction(unittest.TestCase):

    # ------------------------------------------------------------------
    # 2D anchors: p0=(0,0), x=(-1,0), y=(0,1). axis_length is unused in 2D
    # (the planar convention hard-codes unit positions).
    # ------------------------------------------------------------------
    def test_planar_anchor_positions_and_types(self):
        graph = _make_planar_graph()

        self.assertEqual(set(graph.base_nodes), {"p0", "x", "y"})

        assert_array_equal(graph.nodes["p0"][POS], np.array([0, 0]))
        assert_array_equal(graph.nodes["x"][POS], np.array([-1, 0]))
        assert_array_equal(graph.nodes["y"][POS], np.array([0, 1]))

        # TYPE list is membership-checked by all consumers; order is not
        # part of the spec (see compose() merge behaviour).
        self.assertIn(BASE, graph.nodes["p0"][TYPE])
        self.assertIn(ROBOT, graph.nodes["p0"][TYPE])
        self.assertEqual(graph.nodes["x"][TYPE], [BASE])
        self.assertEqual(graph.nodes["y"][TYPE], [BASE])

    def test_planar_anchor_subgraph_is_K3(self):
        graph = _make_planar_graph()
        anchors = ["p0", "x", "y"]

        # The anchor edges, with directions matching the construction:
        #   p0 -> x, p0 -> y, x -> y.  All have BOUNDED=[] (rigid).
        expected_edges = {("p0", "x"), ("p0", "y"), ("x", "y")}
        actual_edges = {(u, v) for u, v in graph.edges() if u in anchors and v in anchors}
        self.assertEqual(actual_edges, expected_edges)

        for u, v in expected_edges:
            d = float(np.linalg.norm(graph.nodes[u][POS] - graph.nodes[v][POS]))
            self.assertAlmostEqual(graph[u][v][DIST], d, places=12)
            self.assertEqual(graph[u][v][BOUNDED], [])

    # ------------------------------------------------------------------
    # 3D anchors: p0=(0,0,0), x=(L,0,0), y=(0,-L,0), q0=(0,0,L)
    # ------------------------------------------------------------------
    def test_revolute_anchor_positions_and_types(self):
        graph = _make_revolute_graph()
        L = graph.axis_length  # 1.0 by default

        self.assertEqual(set(graph.base_nodes), {"p0", "x", "y", "q0"})

        assert_array_equal(graph.nodes["p0"][POS], np.array([0, 0, 0]))
        assert_array_equal(graph.nodes["x"][POS], np.array([L, 0, 0]))
        assert_array_equal(graph.nodes["y"][POS], np.array([0, -L, 0]))
        assert_array_equal(graph.nodes["q0"][POS], np.array([0, 0, L]))

        self.assertIn(BASE, graph.nodes["p0"][TYPE])
        self.assertIn(ROBOT, graph.nodes["p0"][TYPE])
        self.assertEqual(graph.nodes["x"][TYPE], [BASE])
        self.assertEqual(graph.nodes["y"][TYPE], [BASE])
        self.assertIn(BASE, graph.nodes["q0"][TYPE])
        self.assertIn(ROBOT, graph.nodes["q0"][TYPE])

    def test_revolute_anchor_subgraph_is_K4(self):
        graph = _make_revolute_graph()
        anchors = ["p0", "x", "y", "q0"]

        # Construction edge-direction convention (preserved from the old
        # graph_revolute.py): p0 dominates x/y/q0, then x->y, y->q0, q0->x.
        expected_edges = {
            ("p0", "x"), ("p0", "y"), ("p0", "q0"),
            ("x", "y"), ("y", "q0"), ("q0", "x"),
        }
        actual_edges = {(u, v) for u, v in graph.edges() if u in anchors and v in anchors}
        self.assertEqual(actual_edges, expected_edges)

        for u, v in expected_edges:
            d = float(np.linalg.norm(graph.nodes[u][POS] - graph.nodes[v][POS]))
            self.assertAlmostEqual(graph[u][v][DIST], d, places=12)
            self.assertEqual(graph[u][v][BOUNDED], [])

    def test_revolute_anchor_positions_scale_with_axis_length(self):
        L = 2.5
        graph = _make_revolute_graph(axis_length=L)

        self.assertEqual(graph.axis_length, L)
        assert_array_equal(graph.nodes["p0"][POS], np.array([0, 0, 0]))
        assert_array_equal(graph.nodes["x"][POS], np.array([L, 0, 0]))
        assert_array_equal(graph.nodes["y"][POS], np.array([0, -L, 0]))
        assert_array_equal(graph.nodes["q0"][POS], np.array([0, 0, L]))

        # Anchor edge distances scale with L too.
        self.assertAlmostEqual(graph["p0"]["x"][DIST], L, places=12)
        self.assertAlmostEqual(graph["p0"]["y"][DIST], L, places=12)
        self.assertAlmostEqual(graph["p0"]["q0"][DIST], L, places=12)
        self.assertAlmostEqual(graph["x"]["y"][DIST], L * np.sqrt(2), places=12)


if __name__ == "__main__":
    unittest.main()
