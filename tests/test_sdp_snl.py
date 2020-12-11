import numpy as np
import unittest
import networkx as nx
from numpy.testing import assert_allclose
from graphik.graphs.graph_base import RobotRevoluteGraph

from graphik.robots.robot_base import RobotRevolute
from graphik.solvers.constraints import get_full_revolute_nearest_point
from graphik.solvers.sdp_snl import distance_constraints, evaluate_linear_map, constraint_clique_dict_to_sdp
from graphik.utils.roboturdf import load_ur10


class TestUR10(unittest.TestCase):

    def setUp(self) -> None:
        self.robot, self.graph = load_ur10()

    def test_constraints(self):
        n_runs = 10
        for _ in range(n_runs):
            for sparse in [True, False]:  # Whether to exploit chordal sparsity in the SDP formulation
                for ee_cost in [True, False]:  # Whether to treat the end-effectors as variables with targets in the cost
                    q = self.robot.random_configuration()
                    full_points = [f'p{idx}' for idx in range(0, self.robot.n + 1)] + \
                                  [f'q{idx}' for idx in range(0, self.robot.n + 1)]
                    input_vals = get_full_revolute_nearest_point(self.graph, q, full_points)
                    end_effectors = {key: input_vals[key] for key in ['p0', 'q0', f'p{self.robot.n}', f'q{self.robot.n}']}

                    constraint_clique_dict = distance_constraints(self.robot, end_effectors, sparse, ee_cost)
                    A, b, mapping, _ = list(constraint_clique_dict.values())[0]

                    for clique in constraint_clique_dict:
                        A, b, mapping, _ = constraint_clique_dict[clique]
                        evaluations = evaluate_linear_map(clique, A, b, mapping, input_vals)
                        for evaluation in evaluations:
                            self.assertAlmostEqual(evaluation, 0.0)
