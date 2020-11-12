import numpy as np
import time
import unittest
import networkx as nx
from graphik.robots.revolute import (
    Spherical3dChain,
    Spherical3dTree,
)

from graphik.utils.utils import list_to_variable_dict, list_to_variable_dict_spherical


class TestLambdifiedForwardKinematics(unittest.TestCase):
    def test_simple_random_chain(self):
        n_runs = 100
        n_links = 5
        d = list(np.random.rand(n_links) + 0.5)
        d_dict = list_to_variable_dict(d)
        theta = n_links * [0.0]
        theta_dict = list_to_variable_dict(theta)
        al = n_links * [0.0]
        al_dict = list_to_variable_dict(al)
        a = n_links * [0.0]
        a_dict = list_to_variable_dict(a)
        lim_u = list_to_variable_dict(np.pi * np.ones(n_runs))
        lim_l = list_to_variable_dict(-np.pi * np.ones(n_runs))
        params = {
            "d": d_dict,
            "a": a_dict,
            "alpha": al_dict,
            "theta": theta_dict,
            "joint_limits_upper": lim_u,
            "joint_limits_lower": lim_l,
        }
        robot = Spherical3dChain(params)
        robot.lambdify_get_pose()
        t_fast_total = 0.0
        t_total = 0.0
        for idx in range(n_runs):
            input_list = list(np.random.rand(2 * n_links))
            input = list_to_variable_dict_spherical(input_list)
            t1 = time.time()
            poses_fast = robot.get_full_pose_fast_lambdify(input)
            t_fast_total += time.time() - t1
            poses = {}
            t1 = time.time()
            input_pairs = list_to_variable_dict_spherical(input_list, in_pairs=True)
            for key in input_pairs:
                poses[key] = robot.get_pose(input_pairs, key)
            t_total += time.time() - t1

            for key in input_pairs:
                R_true = poses[key].rot.as_matrix()
                R = poses_fast[key][0:3, 0:3]
                t_true = poses[key].trans
                t = poses_fast[key][0:3, 3]
                self.assertTrue(np.all(np.isclose(R, R_true)))
                self.assertTrue(np.all(np.isclose(t, t_true)))

        print("Fast average runtime: {:}".format(t_fast_total / n_runs))
        print("Slow average runtime: {:}".format(t_total / n_runs))

    def test_random_tree(self):
        n_runs = 10
        height = 3
        gen = nx.balanced_tree(2, height, create_using=nx.DiGraph)
        gen = nx.relabel_nodes(gen, {node: f"p{node}" for node in gen})
        n = gen.number_of_edges()
        parents = nx.to_dict_of_lists(gen)
        d = list(np.random.rand(n) + 0.5)
        d_dict = list_to_variable_dict(d)
        theta = n * [0.0]
        theta_dict = list_to_variable_dict(theta)
        al = n * [0.0]
        al_dict = list_to_variable_dict(al)
        a = n * [0.0]
        a_dict = list_to_variable_dict(a)
        lim_u = list_to_variable_dict(np.pi * np.ones(n))
        lim_l = list_to_variable_dict(-np.pi * np.ones(n))
        params = {
            "d": d_dict,
            "a": a_dict,
            "alpha": al_dict,
            "theta": theta_dict,
            "joint_limits_upper": lim_u,
            "joint_limits_lower": lim_l,
            "parents": parents,
        }
        robot = Spherical3dTree(params)
        robot.lambdify_get_pose()
        t_fast_total = 0.0
        t_total = 0.0
        for idx in range(n_runs):
            input_list = list(np.random.rand(2 * n))
            input = list_to_variable_dict_spherical(input_list)
            t1 = time.time()
            poses_fast = robot.get_full_pose_fast_lambdify(input)
            t_fast_total += time.time() - t1
            poses = {}
            t1 = time.time()
            input_pairs = list_to_variable_dict_spherical(input_list, in_pairs=True)
            for key in input_pairs:
                poses[key] = robot.get_pose(input_pairs, key)
            t_total += time.time() - t1

            for key in input_pairs:
                R_true = poses[key].rot.as_matrix()
                R = poses_fast[key][0:3, 0:3]
                t_true = poses[key].trans
                t = poses_fast[key][0:3, 3]
                self.assertTrue(np.all(np.isclose(R, R_true)))
                self.assertTrue(np.all(np.isclose(t, t_true)))

        print("Fast average runtime: {:}".format(t_fast_total / n_runs))
        print("Slow average runtime: {:}".format(t_total / n_runs))


if __name__ == "__main__":
    unittest.main()
