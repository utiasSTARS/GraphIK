import unittest
import numpy as np
import networkx as nx

from liegroups.numpy import SO3, SE3

from graphik.robots.revolute import Revolute3dChain, Revolute3dTree, Spherical3dTree
from graphik.utils.utils import list_to_variable_dict
from graphik.solvers.local_solver import LocalSolver


class TestRandom(unittest.TestCase):
    def test_random_chains(self):
        np.random.seed(1234567)
        N_runs = 3
        for _ in range(N_runs):
            n = np.random.randint(2, 4)
            a = list_to_variable_dict(np.random.rand(n))
            d = list_to_variable_dict(np.random.rand(n))
            al = list_to_variable_dict(np.random.rand(n))
            th = list_to_variable_dict(0 * np.random.rand(n))
            lim_u = list_to_variable_dict(np.pi * np.ones(n))
            lim_l = list_to_variable_dict(-np.pi * np.ones(n))

            params = {
                "a": a,
                "alpha": al,
                "d": d,
                "theta": th,
                "joint_limits_lower": lim_l,
                "joint_limits_upper": lim_u,
            }
            robot = Revolute3dChain(params)
            ee_goals = {'p' + str(n): SE3(SO3.identity(), np.random.rand(3))}
            variable_angles = ['p' + str(idx) for idx in range(1, n + 1)]

            solver_symb = LocalSolver()
            solver_symb.set_symbolic_cost_function(robot, ee_goals, variable_angles)
            solver_rev = LocalSolver()
            solver_rev.set_revolute_cost_function(robot, ee_goals, variable_angles)

            g_symb = solver_symb.grad(np.zeros(n)).astype(float)
            h_symb = solver_symb.hess(np.zeros(n)).astype(float)
            g_rev = solver_rev.grad(np.zeros(n)).astype(float)
            h_rev = solver_rev.hess(np.zeros(n)).astype(float)

            self.assertTrue(np.all(np.isclose(g_symb, g_rev)))
            self.assertTrue(np.all(np.isclose(h_symb, h_rev)))

    def test_tree(self):
        np.random.seed(1234567)
        N_runs = 1
        for _ in range(N_runs):
            height = 2
            gen = nx.balanced_tree(2, height, create_using=nx.DiGraph)
            gen = nx.relabel_nodes(gen, {node: f"p{node}" for node in gen})
            n = gen.number_of_edges()
            a = list_to_variable_dict(np.random.rand(n))
            d = list_to_variable_dict(np.random.rand(n))
            al = list_to_variable_dict(np.zeros(n))
            th = list_to_variable_dict(0 * np.random.rand(n))
            lim_u = list_to_variable_dict(np.pi * np.ones(n))
            lim_l = list_to_variable_dict(-np.pi * np.ones(n))
            parents = nx.to_dict_of_lists(gen)
            params = {
                "a": a,
                "alpha": al,
                "d": d,
                "theta": th,
                "joint_limits_lower": lim_l,
                "joint_limits_upper": lim_u,
                "parents": parents
            }
            robot = Revolute3dTree(params)

            ee_goals = {'p' + str(n): SE3(SO3.identity(), np.random.rand(3))}
            # ee_goals = {'p' + str(n): np.random.rand(3)}
            # variable_angles = ['p' + str(idx) for idx in range(1, n + 1)]
            variable_angles = list(robot.a.keys())

            solver_symb = LocalSolver()
            solver_symb.set_symbolic_cost_function(robot, ee_goals, variable_angles)
            solver_rev = LocalSolver()
            solver_rev.set_revolute_cost_function(robot, ee_goals, variable_angles)

            input = np.random.rand(n)
            g_symb = solver_symb.grad(input).astype(float)
            h_symb = solver_symb.hess(input).astype(float)
            g_rev = solver_rev.grad(input).astype(float)
            h_rev = solver_rev.hess(input).astype(float)

            self.assertTrue(np.all(np.isclose(g_symb, g_rev)))
            self.assertTrue(np.all(np.isclose(h_symb, h_rev)))

    def test_tree_from_spherical(self):
        """
        This passed as of Nov. 11th, 2020, but is too slow to run frequently as part of the standard unit tests, so it's
        commented out for now.
        """
        pass
        # np.random.seed(1234567)
        # N_runs = 1
        # for _ in range(N_runs):
        #     height = 2  # np.random.randint(2, 3)
        #     gen = nx.balanced_tree(2, height, create_using=nx.DiGraph)
        #     gen = nx.relabel_nodes(gen, {node: f"p{node}" for node in gen})
        #     n = gen.number_of_edges()
        #     parents = nx.to_dict_of_lists(gen)
        #     a = list_to_variable_dict(np.zeros(n))
        #     d = list_to_variable_dict(np.random.rand(n))
        #     al = list_to_variable_dict(np.zeros(n))
        #     th = list_to_variable_dict(np.zeros(n))
        #     lim_u = list_to_variable_dict(np.pi * np.ones(n))
        #     lim_l = list_to_variable_dict(-np.pi * np.ones(n))
        #
        #     params = {
        #         "a": a,
        #         "alpha": al,
        #         "d": d,
        #         "theta": th,
        #         "joint_limits_lower": lim_l,
        #         "joint_limits_upper": lim_u,
        #         "parents": parents
        #     }
        #     robot_spherical = Spherical3dTree(params)
        #
        #     robot, names_map, angles_map = robot_spherical.to_revolute()
        #
        #     ee_goals_sphere = {ee[0]: SE3(SO3.identity(), np.random.rand(3)) for ee in robot_spherical.end_effectors}
        #     ee_goals = {names_map[ee]: ee_goals_sphere[ee] for ee in ee_goals_sphere.keys()}
        #     # ee_goals = {'p' + str(n): np.random.rand(3)}
        #     # variable_angles = ['p' + str(idx) for idx in range(1, n + 1)]
        #     variable_angles_spherical = list(robot_spherical.a.keys())
        #     variable_angles = list(robot.T_zero.keys())[1:]
        #     solver_sphere = LocalSolver()
        #     solver_sphere.set_symbolic_cost_function(robot_spherical, ee_goals_sphere, variable_angles_spherical)
        #     solver_symb = LocalSolver()
        #     solver_symb.set_symbolic_cost_function(robot, ee_goals, variable_angles)
        #     solver_rev = LocalSolver()
        #     solver_rev.set_revolute_cost_function(robot, ee_goals, variable_angles)
        #
        #     input = np.random.rand(2 * n)
        #     g_sphere = solver_sphere.grad(input).astype(float)
        #     h_sphere = solver_sphere.hess(input).astype(float)
        #     g_symb = solver_symb.grad(input).astype(float)
        #     h_symb = solver_symb.hess(input).astype(float)
        #     g_rev = solver_rev.grad(input).astype(float)
        #     h_rev = solver_rev.hess(input).astype(float)
        #
        #     self.assertTrue(np.all(np.isclose(g_symb, g_rev)))
        #     self.assertTrue(np.all(np.isclose(h_symb, h_rev)))
        #     self.assertTrue(np.all(np.isclose(g_sphere, g_rev)))
        #     self.assertTrue(np.all(np.isclose(h_sphere, h_rev)))


if __name__ == "__main__":
    unittest.main()
