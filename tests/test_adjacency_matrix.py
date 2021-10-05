import numpy as np
import networkx as nx
import random
from numpy import pi
from numpy.testing import assert_array_equal
import unittest
from graphik.utils.constants import *
from itertools import combinations, groupby
from graphik.graphs import RobotPlanarGraph
from graphik.robots import RobotPlanar

from graphik.utils.dgp import adjacency_matrix_from_graph
from graphik.utils.utils import list_to_variable_dict

def gnp_random_connected_graph(n, p):
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi
    graph, but enforcing that the resulting graph is connected
    """
    edges = combinations(range(n), 2)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if p <= 0:
        return G
    if p >= 1:
        return nx.complete_graph(n, create_using=G)
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < p:
                G.add_edge(*e)
    return G

class TestAdjacencyMatrices(unittest.TestCase):
    def test_random_graph(self):
        NUM_TESTS = 100

        # pre-generate a list of random graph sizes
        n = np.random.randint(4,30,size=NUM_TESTS)

        for idx in range(NUM_TESTS):
            # generate random ladder graph and set distances to 1
            G = gnp_random_connected_graph(n[idx],0.2)
            nx.set_edge_attributes(G, 1, DIST)

            # get adjacency with networkx
            F_gt = nx.adjacency_matrix(G, weight = DIST).todense()

            # get adjacency matrix using our library
            F = adjacency_matrix_from_graph(nx.DiGraph(G))

            assert_array_equal(F, F_gt)


    def test_planar_chain_pose_goal(self):
        n = 3
        a = list_to_variable_dict(np.ones(n))
        th = list_to_variable_dict(np.zeros(n))
        params = {
            "link_lengths": a,
            "theta": th,
            "num_joints": n
        }

        # Adjacency matrix derived by hand
        F_gt = np.array(
            [
                [0, 1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1, 1],
                [1, 1, 0, 0, 1, 1],
                [1, 0, 0, 0, 1, 0],
                [1, 1, 1, 1, 0, 1],
                [1, 1, 1, 0, 1, 0],
            ]
        )

        robot = RobotPlanar(params)
        graph = RobotPlanarGraph(robot)

        q_goal = graph.robot.random_configuration()
        goals = {
            f"p{n}": robot.pose(q_goal, f"p{n}").trans,
            f"p{n-1}": robot.pose(q_goal, f"p{n-1}").trans,
        }
        G = graph.complete_from_pos(goals)

        F = adjacency_matrix_from_graph(G)

        assert_array_equal(F, F_gt)

    def test_planar_chain_position_goal(self):
        n = 3
        a = list_to_variable_dict(np.ones(n))
        th = list_to_variable_dict(np.zeros(n))
        params = {
            "link_lengths": a,
            "theta": th,
            "num_joints": n
        }

        # Adjacency matrix derived by hand
        F_gt = np.array(
            [
                [0, 1, 1, 1, 0, 1],
                [1, 0, 1, 0, 0, 1],
                [1, 1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 1],
                [1, 1, 1, 0, 1, 0],
            ]
        )

        robot = RobotPlanar(params)
        graph = RobotPlanarGraph(robot)

        q_goal = graph.robot.random_configuration()
        goals = {
            f"p{n}": robot.pose(q_goal, f"p{n}").trans,
            # f"p{n-1}": robot.pose(q_goal, f"p{n-1}").trans,
        }
        G = graph.complete_from_pos(goals)

        F = adjacency_matrix_from_graph(G)

        assert_array_equal(F, F_gt)

    def test_planar_tree_position_goal(self):
        height = 3
        gen = nx.balanced_tree(2, height, create_using=nx.DiGraph)
        gen = nx.relabel_nodes(gen, {node: f"p{node}" for node in gen})
        n = gen.number_of_edges()
        parents = nx.to_dict_of_lists(gen)
        a = list_to_variable_dict(np.ones(n))
        th = list_to_variable_dict(np.zeros(n))
        params = {
            "link_lengths": a,
            "theta": th,
            "parents": parents,
            "num_joints": n
        }

        # Adjacency matrix derived by hand
        F_gt = np.array(
            [
                [
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                ],
            ]
        )
        robot = RobotPlanar(params)
        graph = RobotPlanarGraph(robot)

        q_goal = robot.random_configuration()
        goals = {}
        for idx, ee in enumerate(robot.end_effectors):
            goals[ee] = robot.pose(q_goal, ee).trans

        G = graph.complete_from_pos(goals)

        idd = graph.node_ids
        for idx, id in enumerate(idd[4:]):
            idd[2 + int(id[1:])] = id

        F = adjacency_matrix_from_graph(G, nodelist = idd).astype(int)

        assert_array_equal(F, F_gt)

    def test_planar_tree_pose_goal(self):
        height = 3
        gen = nx.balanced_tree(2, height, create_using=nx.DiGraph)
        gen = nx.relabel_nodes(gen, {node: f"p{node}" for node in gen})
        n = gen.number_of_edges()
        parents = nx.to_dict_of_lists(gen)
        a = list_to_variable_dict(np.ones(n))
        th = list_to_variable_dict(np.zeros(n))
        lim_u = list_to_variable_dict(np.pi * np.ones(n))
        lim_l = list_to_variable_dict(-np.pi * np.ones(n))
        params = {
            "link_lengths": a,
            "theta": th,
            "parents": parents,
            "ub": lim_u,
            "lb": lim_l,
            "num_joints": n
        }

        # Adjacency matrix derived by hand
        F_gt = np.array(
            [
                [
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    0,
                    1,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    1,
                    0,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                ],
            ]
        )
        robot = RobotPlanar(params)
        graph = RobotPlanarGraph(robot)

        q_goal = robot.random_configuration()
        goals = {}
        for idx, ee in enumerate(robot.end_effectors):
            ee_p = list(robot.predecessors(ee))
            goals[ee] = robot.pose(q_goal, ee).trans
            goals[ee_p[0]] = robot.pose(q_goal, ee_p[0]).trans

        G = graph.complete_from_pos(goals)

        idd = graph.node_ids
        for idx, id in enumerate(idd[4:]):
            idd[2 + int(id[1:])] = id

        F = adjacency_matrix_from_graph(G, nodelist = idd).astype(int)

        assert_array_equal(F, F_gt)
