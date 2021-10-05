import numpy as np
import unittest
import networkx as nx
from numpy.testing import assert_allclose
from numpy import pi
from graphik.graphs import (
    ProblemGraphPlanar,
    ProblemGraphRevolute,
)

from graphik.robots import RobotRevolute, RobotPlanar
from graphik.utils import best_fit_transform, list_to_variable_dict, MDS, gram_from_distance_matrix, pos_from_graph

class TestDistanceMatrix(unittest.TestCase):
    def test_special_case_3d_tree(self):
        print("\n Testing cases with zeroed out parameters ...")
        modified_dh = False

        n = 5
        parents = {"p0": ["p1"], "p1": ["p2", "p3"], "p2": ["p4"], "p3": ["p5"]}
        a = {"p1": 0, "p2": -0.612, "p3": -0.612, "p4": -0.5732, "p5": -0.5732}
        d = {"p1": 0.1237, "p2": 0, "p3": 0, "p4": 0, "p5": 0}
        al = {"p1": pi / 2, "p2": 0, "p3": 0, "p4": 0, "p5": 0}
        th = {"p1": 0, "p2": 0, "p3": 0, "p4": 0, "p5": 0}

        params = {
            "a": a,
            "alpha": al,
            "d": d,
            "theta": th,
            "modified_dh": modified_dh,
            "parents": parents,
            "num_joints": n
        }
        robot = RobotRevolute(params)
        graph = ProblemGraphRevolute(robot)
        n_nodes = graph.number_of_nodes()

        for _ in range(100):

            q = robot.random_configuration()
            D = graph.distance_matrix_from_joints(q)

            # Reconstruct points
            J = np.identity(n_nodes) - (1 / n_nodes) * np.ones(D.shape)
            G = -0.5 * J @ D @ J  # Gram matrix
            u, s, vh = np.linalg.svd(G, full_matrices=True)
            X = (
                np.concatenate(
                    (
                        np.sqrt(np.diag(s[: robot.dim])),
                        np.zeros((robot.dim, n_nodes - robot.dim)),
                    ),
                    axis=1,
                )
                @ vh
            ).T

            Y = pos_from_graph(graph.realization(q))
            R, t = best_fit_transform(X[[0, 1, 2, 3, -1], :], Y[[0, 1, 2, 3, -1], :])
            P_e = (R @ X.T + t.reshape(3, 1)).T
            self.assertIsNone(assert_allclose(P_e, Y, rtol=1e-5, atol=100))
        pass

    def test_random_params_3d_chain(self):
        for idx in range(100):

            # n = np.random.randint(3, high=20)  # number of joints
            n = 8

            # Generate random DH parameters
            a = np.random.rand(n)
            d = np.random.rand(n)
            al = np.random.rand(n) * pi / 2 - 2 * np.random.rand(n) * pi / 2
            th = 0 * np.ones(n)

            params = {
                "a": a,
                "alpha": al,
                "d": d,
                "theta": th,
                "modified_dh": True,
                "num_joints": n
            }
            robot = RobotRevolute(params)  # instantiate robot
            graph = ProblemGraphRevolute(robot)  # instantiate graph
            n_nodes = graph.number_of_nodes()

            q = robot.random_configuration()
            T = robot.pose(q, "p" + str(n))
            D = graph.distance_matrix_from_joints(q)

            # Reconstruct points
            J = np.identity(n_nodes) - (1 / (n_nodes)) * np.ones(D.shape)
            G = -0.5 * J @ D @ J  # Gram matrix
            u, s, vh = np.linalg.svd(G, full_matrices=True)
            X = (
                np.concatenate(
                    (
                        np.sqrt(np.diag(s[: robot.dim])),
                        np.zeros((robot.dim, n_nodes - robot.dim)),
                    ),
                    axis=1,
                )
                @ vh
            ).T

            Y = pos_from_graph(graph.realization(q))
            R, t = best_fit_transform(X[[0, 1, 2, 3, -1], :], Y[[0, 1, 2, 3, -1], :])
            P_e = (R @ X.T + t.reshape(3, 1)).T
            self.assertIsNone(assert_allclose(P_e, Y, rtol=1e-5, atol=100))

    def test_random_params_2d_chain(self):
        for idx in range(100):

            n = np.random.randint(3, high=20)

            a = list_to_variable_dict(np.ones(n))
            th = list_to_variable_dict(np.zeros(n))
            params = {
                "link_lengths": a,
                "theta": th,
                "num_joints": n
            }

            robot = RobotPlanar(params)
            graph = ProblemGraphPlanar(robot)
            n_nodes = graph.number_of_nodes()

            q = robot.random_configuration()
            D = graph.distance_matrix_from_joints(q)

            # Reconstruct points
            G = gram_from_distance_matrix(D)
            u, s, vh = np.linalg.svd(G, full_matrices=True)
            X = (
                np.concatenate(
                    (
                        np.sqrt(np.diag(s[: robot.dim])),
                        np.zeros((robot.dim, n_nodes - robot.dim)),
                    ),
                    axis=1,
                )
                @ vh
            ).T

            Y = pos_from_graph(graph.realization(q))
            R, t = best_fit_transform(X[[0, 1, 2, -1], :], Y[[0, 1, 2, -1], :])
            P_e = (R @ X.T + t.reshape(2, 1)).T

            self.assertIsNone(assert_allclose(P_e, Y, rtol=1e-5, atol=100))

    def test_random_params_2d_tree(self):
        for idx in range(30):

            height = np.random.randint(2, high=5)
            gen = nx.balanced_tree(2, height, create_using=nx.DiGraph)
            gen = nx.relabel_nodes(gen, {node: f"p{node}" for node in gen})
            n = gen.number_of_edges()

            a = list_to_variable_dict(np.ones(n))
            th = list_to_variable_dict(np.zeros(n))
            parents = nx.to_dict_of_lists(gen)
            lim_u = list_to_variable_dict(pi * np.ones(n))
            lim_l = list_to_variable_dict(-pi * np.ones(n))
            params = {
                "link_lengths": a,
                "theta": th,
                "parents": parents,
                "num_joints": n
            }

            robot = RobotPlanar(params)
            graph = ProblemGraphPlanar(robot)
            n_nodes = graph.number_of_nodes()

            q = robot.random_configuration()
            D = graph.distance_matrix_from_joints(q)

            # Reconstruct points
            G = gram_from_distance_matrix(D)
            u, s, vh = np.linalg.svd(G, full_matrices=True)
            X = (
                np.concatenate(
                    (
                        np.sqrt(np.diag(s[: robot.dim])),
                        np.zeros((robot.dim, n_nodes - robot.dim)),
                    ),
                    axis=1,
                )
                @ vh
            ).T

            Y = pos_from_graph(graph.realization(q))
            R, t = best_fit_transform(X[[0, 1, 2, -1], :], Y[[0, 1, 2, -1], :])
            P_e = (R @ X.T + t.reshape(2, 1)).T

            self.assertIsNone(assert_allclose(P_e, Y, atol=1e-8))

