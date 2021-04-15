import numpy as np
import unittest
import networkx as nx
from numpy.testing import assert_allclose
from numpy import pi
from graphik.graphs import (
    RobotPlanarGraph,
    RobotRevoluteGraph,
    RobotSphericalGraph,
)
from graphik.robots import RobotRevolute, RobotSpherical, RobotPlanar
from graphik.utils.dgp import gram_from_distance_matrix, MDS, pos_from_graph
from graphik.utils.utils import (
    best_fit_transform,
    list_to_variable_dict,
)


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
        ub = list_to_variable_dict((pi) * np.ones(n))
        lb = list_to_variable_dict(-(pi) * np.ones(n))

        params = {
            "a": a,
            "alpha": al,
            "d": d,
            "theta": th,
            "lb": lb,
            "ub": ub,
            "modified_dh": modified_dh,
            "parents": parents,
        }
        robot = RobotRevolute(params)
        graph = RobotRevoluteGraph(robot)

        for _ in range(100):

            q = robot.random_configuration()
            D = graph.distance_matrix_from_joints(q)

            # Reconstruct points
            J = np.identity(graph.n_nodes) - (1 / (graph.n_nodes)) * np.ones(D.shape)
            G = -0.5 * J @ D @ J  # Gram matrix
            u, s, vh = np.linalg.svd(G, full_matrices=True)
            X = (
                np.concatenate(
                    (
                        np.sqrt(np.diag(s[: robot.dim])),
                        np.zeros((robot.dim, graph.n_nodes - robot.dim)),
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
            ub = np.ones(n) * pi - 2 * pi * np.random.rand(n)
            lb = -ub

            params = {
                "a": a,
                "alpha": al,
                "d": d,
                "theta": th,
                "lb": lb,
                "ub": ub,
                "modified_dh": True,
            }
            robot = RobotRevolute(params)  # instantiate robot
            graph = RobotRevoluteGraph(robot)  # instantiate graph

            q = robot.random_configuration()
            T = robot.get_pose(list_to_variable_dict(q), "p" + str(n))
            D = graph.distance_matrix_from_joints(q)

            # Reconstruct points
            J = np.identity(graph.n_nodes) - (1 / (graph.n_nodes)) * np.ones(D.shape)
            G = -0.5 * J @ D @ J  # Gram matrix
            u, s, vh = np.linalg.svd(G, full_matrices=True)
            X = (
                np.concatenate(
                    (
                        np.sqrt(np.diag(s[: robot.dim])),
                        np.zeros((robot.dim, graph.n_nodes - robot.dim)),
                    ),
                    axis=1,
                )
                @ vh
            ).T

            Y = pos_from_graph(graph.realization(q))
            R, t = best_fit_transform(X[[0, 1, 2, 3, -1], :], Y[[0, 1, 2, 3, -1], :])
            P_e = (R @ X.T + t.reshape(3, 1)).T
            self.assertIsNone(assert_allclose(P_e, Y, rtol=1e-5, atol=100))

    def test_random_params_3d_spherical_chain(self):
        for idx in range(100):
            n = np.random.randint(3, high=20)

            # Generate random DH parameters
            a = list_to_variable_dict(0 * np.random.rand(n))
            d = list_to_variable_dict(np.random.rand(n))
            al = list_to_variable_dict(0 * np.random.rand(n))
            th = list_to_variable_dict(0 * np.random.rand(n))
            lim_u = list_to_variable_dict(pi * np.ones(n))
            lim_l = list_to_variable_dict(-pi * np.ones(n))

            params = {
                "a": a,
                "alpha": al,
                "d": d,
                "theta": th,
                "joint_limits_lower": lim_l,
                "joint_limits_upper": lim_u,
            }
            robot = RobotSpherical(params)  # instantiate robot
            graph = RobotSphericalGraph(robot)  # instantiate graph

            q = robot.random_configuration()
            T = robot.get_pose(q, f"p{n}")
            D = graph.distance_matrix_from_joints(q)

            # Reconstruct points
            J = np.identity(graph.n_nodes) - (1 / (graph.n_nodes)) * np.ones(D.shape)
            G = -0.5 * J @ D @ J  # Gram matrix
            u, s, vh = np.linalg.svd(G, full_matrices=True)
            X = (
                np.concatenate(
                    (
                        np.sqrt(np.diag(s[: robot.dim])),
                        np.zeros((robot.dim, graph.n_nodes - robot.dim)),
                    ),
                    axis=1,
                )
                @ vh
            ).T

            Y = pos_from_graph(graph.realization(q))
            R, t = best_fit_transform(X[[0, 1, 2, 3, -1], :], Y[[0, 1, 2, 3, -1], :])
            P_e = (R @ X.T + t.reshape(3, 1)).T
            self.assertIsNone(assert_allclose(P_e, Y, rtol=1e-5, atol=100))

    def test_random_params_3d_spherical_tree(self):
        # TODO include randomized theta to FK and such
        print("Testing randomly generated params 3d ... \n")

        for idx in range(100):

            height = np.random.randint(2, high=5)
            gen = nx.balanced_tree(2, height, create_using=nx.DiGraph)
            gen = nx.relabel_nodes(gen, {node: f"p{node}" for node in gen})
            n = gen.number_of_edges()
            # Generate random DH parameters
            a = list_to_variable_dict(0 * np.random.rand(n))
            d = list_to_variable_dict(np.random.rand(n))
            al = list_to_variable_dict(0 * np.random.rand(n))
            th = list_to_variable_dict(0 * np.random.rand(n))
            parents = nx.to_dict_of_lists(gen)
            lim_u = list_to_variable_dict(pi * np.ones(n))
            lim_l = list_to_variable_dict(-pi * np.ones(n))

            params = {
                "a": a,
                "alpha": al,
                "d": d,
                "theta": th,
                "parents": parents,
                "joint_limits_lower": lim_l,
                "joint_limits_upper": lim_u,
            }
            robot = RobotSpherical(params)  # instantiate robot
            graph = RobotSphericalGraph(robot)  # instantiate graph

            q = robot.random_configuration()
            T = robot.get_pose(q, f"p{n}")
            D = graph.distance_matrix_from_joints(q)

            # Reconstruct points
            J = np.identity(graph.n_nodes) - (1 / (graph.n_nodes)) * np.ones(D.shape)
            G = -0.5 * J @ D @ J  # Gram matrix
            u, s, vh = np.linalg.svd(G, full_matrices=True)
            X = (
                np.concatenate(
                    (
                        np.sqrt(np.diag(s[: robot.dim])),
                        np.zeros((robot.dim, graph.n_nodes - robot.dim)),
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
            lim_u = list_to_variable_dict(pi * np.ones(n))
            lim_l = list_to_variable_dict(-pi * np.ones(n))
            params = {
                "a": a,
                "theta": th,
                "joint_limits_upper": lim_u,
                "joint_limits_lower": lim_l,
            }

            robot = RobotPlanar(params)
            graph = RobotPlanarGraph(robot)

            q = robot.random_configuration()
            D = graph.distance_matrix_from_joints(q)

            # Reconstruct points
            G = gram_from_distance_matrix(D)
            u, s, vh = np.linalg.svd(G, full_matrices=True)
            X = (
                np.concatenate(
                    (
                        np.sqrt(np.diag(s[: robot.dim])),
                        np.zeros((robot.dim, graph.n_nodes - robot.dim)),
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
                "a": a,
                "theta": th,
                "parents": parents,
                "joint_limits_upper": lim_u,
                "joint_limits_lower": lim_l,
            }

            robot = RobotPlanar(params)
            graph = RobotPlanarGraph(robot)

            q = robot.random_configuration()
            D = graph.distance_matrix_from_joints(q)

            # Reconstruct points
            G = gram_from_distance_matrix(D)
            u, s, vh = np.linalg.svd(G, full_matrices=True)
            X = (
                np.concatenate(
                    (
                        np.sqrt(np.diag(s[: robot.dim])),
                        np.zeros((robot.dim, graph.n_nodes - robot.dim)),
                    ),
                    axis=1,
                )
                @ vh
            ).T

            Y = pos_from_graph(graph.realization(q))
            R, t = best_fit_transform(X[[0, 1, 2, -1], :], Y[[0, 1, 2, -1], :])
            P_e = (R @ X.T + t.reshape(2, 1)).T

            self.assertIsNone(assert_allclose(P_e, Y, atol=1e-8))


if __name__ == "__main__":

    test = TestDistanceMatrix()
    test.test_random_params_2d_tree()
