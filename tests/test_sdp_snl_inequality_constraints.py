"""Inequality-constraint construction and an end-to-end nearest-point SDP.

These run on whatever SDP solver cvxpy has available (Clarabel by default);
no Mosek required. Tolerances are kept loose accordingly.
"""
import numpy as np
import pytest

from graphik.solvers.sdp_snl import (
    distance_constraints_graph,
    distance_range_constraints,
    get_full_revolute_nearest_point,
    solve_nearest_point_sdp,
)
from graphik.utils.constants import POS
from graphik.utils.roboturdf import load_ur10, load_truncated_ur10

import networkx as nx


def _clique_gram(clique_entry, positions, d):
    """Build Z = X^T X for a clique from a {node: position} dict, matching the
    layout used by the constraint LMEs (augmented cliques carry eye(d))."""
    A, _, mapping, is_augmented = clique_entry
    n_lme = A[0].shape[0]
    n_vars = n_lme - d if is_augmented else n_lme
    X = np.zeros((d, n_vars))
    for var, idx in mapping.items():
        if isinstance(var, str) and var in positions:
            X[:, idx] = positions[var]
    if is_augmented:
        X = np.hstack([X, np.eye(d)])
    return X.T @ X


class TestObstacleInequalityConstruction:
    def setup_method(self):
        np.random.seed(7)
        self.robot, self.graph = load_ur10()
        # An obstacle far from the zero-configuration workspace boundary
        # cannot be violated by the true configuration used below.
        self.obstacle_pos = np.array([10.0, 10.0, 10.0])
        self.radius = 0.5
        self.graph.add_spherical_obstacle("o0", self.obstacle_pos, self.radius)

        n = self.robot.n
        q = self.robot.random_configuration()
        full_points = [f"p{i}" for i in range(n + 1)] + [f"q{i}" for i in range(n + 1)]
        self.input_vals = get_full_revolute_nearest_point(self.graph, q, full_points)

        G = nx.DiGraph(self.graph)
        G.remove_node("x")
        G.remove_node("y")
        self.anchors = {
            key: self.input_vals[key] for key in ["p0", "q0", f"p{n}", f"q{n}"]
        }
        for node, data in G.nodes(data=True):
            if data.get(POS, None) is not None:
                self.anchors[node] = data[POS]
        self.G = G
        self.constraint_clique_dict = distance_constraints_graph(
            G, self.anchors, sparse=False, angle_limits=True
        )
        self.inequality_map = distance_range_constraints(
            G, self.constraint_clique_dict, self.anchors
        )

    def test_obstacle_produces_inequality_constraints(self):
        n_ineq = sum(len(v) for v in self.inequality_map.values())
        assert n_ineq > 0

    def test_feasible_configuration_satisfies_inequalities(self):
        positions = {**self.input_vals, "o0": self.obstacle_pos}
        d = self.robot.dim
        for clique, constraints in self.inequality_map.items():
            Z = _clique_gram(self.constraint_clique_dict[clique], positions, d)
            for A_ineq, b in constraints:
                assert np.trace(A_ineq @ Z) <= b + 1e-9

    def test_point_inside_obstacle_violates_an_inequality(self):
        # Move every robot point onto the obstacle centre: clearance bounds
        # (lower bounds on distance) must be violated.
        positions = {key: self.obstacle_pos.copy() for key in self.input_vals}
        positions["o0"] = self.obstacle_pos
        d = self.robot.dim
        violations = 0
        for clique, constraints in self.inequality_map.items():
            Z = _clique_gram(self.constraint_clique_dict[clique], positions, d)
            for A_ineq, b in constraints:
                if np.trace(A_ineq @ Z) > b + 1e-9:
                    violations += 1
        assert violations > 0


class TestNearestPointSdpFreeSolver:
    def test_nuclear_norm_solve_satisfies_equality_constraints(self):
        np.random.seed(11)
        robot, graph = load_truncated_ur10(3)
        n = robot.n
        q = robot.random_configuration()
        full_points = [f"p{i}" for i in range(n + 1)] + [f"q{i}" for i in range(n + 1)]
        input_vals = get_full_revolute_nearest_point(graph, q, full_points)
        end_effectors = {
            key: input_vals[key] for key in ["p0", "q0", f"p{n}", f"q{n}"]
        }
        # Zero nearest points = nuclear-norm cost; also exercises the
        # zero-target drop in prepare_set_cover_problem.
        nearest_points = {
            key: np.zeros(robot.dim)
            for key in input_vals
            if key not in end_effectors
        }

        solution, prob, constraint_clique_dict, sdp_variable_map = (
            solve_nearest_point_sdp(nearest_points, end_effectors, graph)
        )

        assert prob.status in ("optimal", "optimal_inaccurate")
        assert set(nearest_points.keys()) <= set(solution.keys())
        # The SDP variable must satisfy the distance LMEs to solver tolerance.
        for clique, Z_var in sdp_variable_map.items():
            A, b, _, _ = constraint_clique_dict[clique]
            Z = Z_var.value
            residuals = [np.trace(A[i] @ Z) - b[i] for i in range(len(A))]
            assert np.max(np.abs(residuals)) < 1e-5
