import numpy as np
import unittest
import networkx as nx
from numpy.testing import assert_allclose
from graphik.graphs import ProblemGraphRevolute

from graphik.robots.robot_base import Robot
from graphik.robots import RobotRevolute
from graphik.solvers.constraints import get_full_revolute_nearest_point
from graphik.solvers.sdp_snl import (
    distance_constraints,
    distance_constraints_graph,
    evaluate_linear_map,
    constraints_and_nearest_points_to_sdp_vars,
    evaluate_cost,
)
from graphik.utils.constants import *
from graphik.utils.roboturdf import load_ur10


def run_constraint_test(test_case, graph: Robot, sparse=False, ee_cost=False):
    robot = graph.robot
    n = robot.n

    # get a copy of the current robot + environment graph
    G = nx.DiGraph(graph)

    # remove base nodes and all adjacent edges
    G.remove_node("x")
    G.remove_node("y")

    q = robot.random_configuration()

    full_points = [f"p{idx}" for idx in range(0, n + 1)] + [
        f"q{idx}" for idx in range(0, n + 1)
    ]

    true_input_vals = get_full_revolute_nearest_point(graph, q, full_points)
    random_input_vals = {
        key: np.random.rand(graph.robot.dim) for key in true_input_vals
    }

    anchors = {key: true_input_vals[key] for key in ["p0", "q0", f"p{n}", f"q{n}"]}

    # If a position is pre-defined for a node, set to anchor
    for node, data in G.nodes(data=True):
        if data.get(POS, None) is not None:
            anchors[node] = data[POS]

    constraint_clique_dict = distance_constraints_graph(G, anchors, sparse, ee_cost)

    # remove the edges that don't have distances defined
    edges = []
    for u, v, data in G.edges(data=True):
        if not data.get(DIST, False):
            edges += [(u, v)]
    G.remove_edges_from(edges)
    undirected = G.to_undirected()

    for clique in constraint_clique_dict:
        A_clique, b_clique, mapping, _ = constraint_clique_dict[clique]
        true_evaluations = evaluate_linear_map(
            clique, A_clique, b_clique, mapping, true_input_vals
        )
        for true_eval in true_evaluations:
            test_case.assertAlmostEqual(true_eval, 0.0)

        random_evaluations = evaluate_linear_map(
            clique, A_clique, b_clique, mapping, random_input_vals
        )
        # TODO: change this to iterate over edges in the graph object and check that they exist with assert!
        for u in clique:
            for v in clique:
                if frozenset((u, v)) in mapping:
                    idx = mapping[frozenset((u, v))]
                    sdp_residual = random_evaluations[idx]
                    u_val = (
                        anchors[u]
                        if u in anchors and not ee_cost
                        else random_input_vals[u]
                    )
                    v_val = (
                        anchors[v]
                        if v in anchors and not ee_cost
                        else random_input_vals[v]
                    )
                    true_residual = (
                        np.linalg.norm(u_val - v_val) ** 2 - undirected[u][v][DIST] ** 2
                    )

                    test_case.assertAlmostEqual(sdp_residual, true_residual)


def run_cost_test(test_case, robot, graph, sparse=False, ee_cost=False):
    q = robot.random_configuration()
    full_points = [f"p{idx}" for idx in range(0, robot.n + 1)] + [
        f"q{idx}" for idx in range(0, robot.n + 1)
    ]
    input_vals = get_full_revolute_nearest_point(graph, q, full_points)
    end_effectors = {
        key: input_vals[key] for key in ["p0", "q0", f"p{robot.n}", f"q{robot.n}"]
    }

    constraint_clique_dict = distance_constraints(graph, end_effectors, sparse, ee_cost)
    A, b, mapping, _ = list(constraint_clique_dict.values())[0]

    # Make cost function stuff
    interior_nearest_points = (
        {
            key: input_vals[key]
            for key in input_vals
            if key not in ["p0", "q0", f"p{robot.n}", f"q{robot.n}"]
        }
        if not ee_cost
        else end_effectors
    )
    (
        sdp_variable_map,
        sdp_constraints_map,
        sdp_cost_map,
    ) = constraints_and_nearest_points_to_sdp_vars(
        constraint_clique_dict, interior_nearest_points, robot.dim
    )
    cost = evaluate_cost(constraint_clique_dict, sdp_cost_map, interior_nearest_points)
    test_case.assertAlmostEqual(cost, 0.0)
    random_nearest_points = {key: np.random.rand(3) for key in interior_nearest_points}
    cost_bad = evaluate_cost(
        constraint_clique_dict, sdp_cost_map, random_nearest_points
    )
    cost_bad_explicit = sum(
        [
            np.linalg.norm(random_nearest_points[key] - interior_nearest_points[key])
            ** 2
            for key in interior_nearest_points
        ]
    )
    test_case.assertAlmostEqual(cost_bad, cost_bad_explicit)


class TestUR10(unittest.TestCase):
    def setUp(self):
        self.robot, self.graph = load_ur10()

    def test_constraints(self):
        n_runs = 10
        for _ in range(n_runs):
            for sparse in [
                True,
                False,
            ]:  # Whether to exploit chordal sparsity in the SDP formulation
                for ee_cost in [
                    True,
                    False,
                ]:  # Whether to treat the end-effectors as variables with targets in the cost
                    q = self.robot.random_configuration()
                    full_points = [f"p{idx}" for idx in range(0, self.robot.n + 1)] + [
                        f"q{idx}" for idx in range(0, self.robot.n + 1)
                    ]

                    input_vals = get_full_revolute_nearest_point(
                        self.graph, q, full_points
                    )
                    random_input_vals = {key: np.random.rand(3) for key in input_vals}

                    # Copy of the current robot + environment graph
                    G = nx.DiGraph(self.graph)
                    G.remove_node("x")
                    G.remove_node("y")

                    anchors = {
                        key: input_vals[key]
                        for key in ["p0", "q0", f"p{self.robot.n}", f"q{self.robot.n}"]
                    }

                    # If a position is pre-defined for a node, set to anchor
                    for node, data in G.nodes(data=True):
                        if data.get(POS, None) is not None:
                            anchors[node] = data[POS]

                    constraint_clique_dict = distance_constraints_graph(
                        G, anchors, sparse, ee_cost=ee_cost
                    )

                    for clique in constraint_clique_dict:
                        A, b, mapping, _ = constraint_clique_dict[clique]
                        evaluations = evaluate_linear_map(
                            clique, A, b, mapping, input_vals
                        )
                        random_evaluations = evaluate_linear_map(
                            clique, A, b, mapping, random_input_vals
                        )
                        for evaluation in evaluations:
                            self.assertAlmostEqual(evaluation, 0.0)
                        for evaluation in random_evaluations:
                            self.assertNotAlmostEqual(evaluation, 0.0)

    def test_cost(self):
        n_runs = 10
        for _ in range(n_runs):
            for sparse in [True, False]:
                for ee_cost in [True, False]:
                    run_cost_test(self, self.robot, self.graph, sparse, ee_cost)

    def test_distance_constraints(self):
        n_runs = 10
        # sparse = False
        # ee_cost = False
        for _ in range(n_runs):
            for sparse in [True, False]:
                for ee_cost in [True, False]:
                    run_constraint_test(self, self.graph, sparse, ee_cost)


class TestTruncatedUR10(unittest.TestCase):
    def setUp(self):
        self.robot, self.graph = load_ur10()
        n = 3
        a_full = [0, -0.612, -0.5723, 0, 0, 0]
        d_full = [0.1273, 0, 0, 0.1639, 0.1157, 0.0922]
        al_full = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]
        th_full = [0, 0, 0, 0, 0, 0]
        a = a_full[0:n]
        d = d_full[0:n]
        al = al_full[0:n]
        th = th_full[0:n]
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
            "num_joints": n
        }
        self.robot = RobotRevolute(params)
        self.graph = ProblemGraphRevolute(self.robot)

    def test_cost(self):
        n_runs = 10
        # sparse = False
        # ee_cost = False
        for _ in range(n_runs):
            for sparse in [True, False]:
                for ee_cost in [True, False]:
                    run_cost_test(self, self.robot, self.graph, sparse, ee_cost)

    def test_distance_constraints(self):
        n_runs = 10
        # sparse = False
        # ee_cost = False
        for _ in range(n_runs):
            for sparse in [True, False]:
                for ee_cost in [True, False]:
                    run_constraint_test(self, self.graph, sparse, ee_cost)


if __name__ == "__main__":
    unittest.main()
