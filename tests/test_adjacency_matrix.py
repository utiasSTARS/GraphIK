import numpy as np
import networkx as nx
import random
from numpy import pi
from numpy.testing import assert_array_equal
import unittest
from graphik.utils.constants import *
from itertools import combinations, groupby
from graphik.graphs import ProblemGraph
from graphik.robots import Robot

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

        robot = Robot({**params, "dim": 2})
        graph = ProblemGraph(robot)

        q_goal = graph.robot.random_configuration()
        goals = {
            f"p{n}": robot.pose(q_goal, f"p{n}")[:2, 2],
            f"p{n-1}": robot.pose(q_goal, f"p{n-1}")[:2, 2],
        }
        G = graph.from_pos(goals)

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

        robot = Robot({**params, "dim": 2})
        graph = ProblemGraph(robot)

        q_goal = graph.robot.random_configuration()
        goals = {
            f"p{n}": robot.pose(q_goal, f"p{n}")[:2, 2],
            # f"p{n-1}": robot.pose(q_goal, f"p{n-1}").trans,
        }
        G = graph.from_pos(goals)

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
        F_gt = np.array([
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        ])
        robot = Robot({**params, "dim": 2})
        graph = ProblemGraph(robot)

        q_goal = robot.random_configuration()
        goals = {}
        for idx, ee in enumerate(robot.end_effectors):
            goals[ee] = robot.pose(q_goal, ee)[:2, 2]

        G = graph.from_pos(goals)

        idd = graph.node_ids
        for idx, id in enumerate(idd[4:]):
            idd[2 + int(id[1:])] = id

        F = adjacency_matrix_from_graph(G, nodelist = idd).astype(int)

        assert_array_equal(F, F_gt)

    # ------------------------------------------------------------------
    # 3D revolute chain — mirrors the planar chain tests above. n=3 gives
    # a 10x10 matrix (4 anchors + 2 nodes per joint) and exercises both
    # an intermediate joint pair (joint 0->2) and an end-effector joint
    # pair (joint 1->3) in the level-2 set_limits machinery.
    #
    # Graph construction adds DIST to:
    #   - base K4 on {p0, x, y, q0}: 6 edges
    #   - intra-joint (p_i, q_i) for each joint: 3 new edges (p0-q0 dup'd by base)
    #   - inter-joint complete-bipartite K_{2,2} between consecutive joints: 12 edges
    # Construction also adds BOUNDED-only edges (no DIST) for:
    #   - level-2 set_limits (joint 0->2 and joint 1->3): 8 edges
    #   - root_angle_limits (x/y to p1/q1): 4 edges
    # adjacency_matrix_from_graph filters by DIST, so BOUNDED-only edges
    # do not appear unless goal-injection completion later assigns DIST.
    # ------------------------------------------------------------------
    def _revolute_chain_n3_params(self):
        """DH params for a 3-joint 3D revolute chain used below."""
        return {
            "a": {"p1": 1.0, "p2": 1.0, "p3": 1.0},
            "alpha": {"p1": pi / 2, "p2": 0.0, "p3": pi / 2},
            "d": {"p1": 0.0, "p2": 0.0, "p3": 0.0},
            "theta": {"p1": 0.0, "p2": 0.0, "p3": 0.0},
            "modified_dh": False,
            "parents": {"p0": ["p1"], "p1": ["p2"], "p2": ["p3"]},
            "num_joints": 3,
        }

    def test_revolute_chain_construction_only(self):
        # Construction-only adjacency (no goal injection). This isolates
        # graph construction from goal-completion: only DIST set during
        # _build_*_subgraph shows up. The 12 BOUNDED-only edges (8 from
        # level-2 set_limits, 4 from root_angle_limits) do NOT appear
        # here -- they have UPPER/LOWER but no DIST.
        #
        # Node order (insertion order from nx.compose(base, structure)):
        #   0: p0   1: x   2: y   3: q0
        #   4: p1   5: q1   6: p2   7: q2   8: p3   9: q3
        #
        # Expected entries (21 unique undirected edges with DIST):
        #   - base K4 on {p0, x, y, q0}: 6
        #   - intra-joint (p1,q1), (p2,q2), (p3,q3): 3 (plus (p0,q0) shared with base)
        #   - inter-joint complete bipartite for each of 3 consecutive pairs: 12
        F_gt = np.array([
            [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 1, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 1, 1, 1, 0, 0],
            [1, 0, 0, 1, 1, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        ])

        params = self._revolute_chain_n3_params()
        robot = Robot({**params, "dim": 3})
        graph = ProblemGraph(robot)

        F = adjacency_matrix_from_graph(graph).astype(int)
        assert_array_equal(F, F_gt)

    def test_revolute_chain_pose_goal(self):
        # Same node order as the construction-only test.
        #
        # After from_pose(T_goal_p3), positions are set for {p0, x, y, q0,
        # p3, q3} (4 anchors + EE p-node + EE q-node from _pose_goal). The
        # completion step then adds DIST between every pair of positioned
        # nodes (8 new edges total): (p0,p3), (p0,q3), (x,p3), (x,q3),
        # (y,p3), (y,q3), (q0,p3), (q0,q3). All BOUNDED-only edges
        # involving the unpositioned p1/q1/p2/q2 nodes (level-2 set_limits
        # joint 0->2, level-2 set_limits joint 1->3, root_angle_limits)
        # stay zero in the matrix.
        F_gt = np.array([
            [0, 1, 1, 1, 1, 1, 0, 0, 1, 1],
            [1, 0, 1, 1, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 1, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 0, 1, 1, 0, 0, 1, 1],
            [1, 0, 0, 1, 0, 1, 1, 1, 0, 0],
            [1, 0, 0, 1, 1, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 0, 0, 1, 1, 0, 1],
            [1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
        ])

        params = self._revolute_chain_n3_params()
        robot = Robot({**params, "dim": 3})
        graph = ProblemGraph(robot)

        q_goal = robot.random_configuration()
        T_goal = robot.pose(q_goal, "p3")
        G = graph.from_pose(T_goal)

        F = adjacency_matrix_from_graph(G).astype(int)
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
            "joint_limits_upper": lim_u,
            "joint_limits_lower": lim_l,
            "num_joints": n
        }

        # Adjacency matrix derived by hand
        F_gt = np.array([
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
            [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
            [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        ])
        robot = Robot({**params, "dim": 2})
        graph = ProblemGraph(robot)

        q_goal = robot.random_configuration()
        goals = {}
        for idx, ee in enumerate(robot.end_effectors):
            ee_p = list(robot.predecessors(ee))
            goals[ee] = robot.pose(q_goal, ee)[:2, 2]
            goals[ee_p[0]] = robot.pose(q_goal, ee_p[0])[:2, 2]

        G = graph.from_pos(goals)

        idd = graph.node_ids
        for idx, id in enumerate(idd[4:]):
            idd[2 + int(id[1:])] = id

        F = adjacency_matrix_from_graph(G, nodelist = idd).astype(int)

        assert_array_equal(F, F_gt)
