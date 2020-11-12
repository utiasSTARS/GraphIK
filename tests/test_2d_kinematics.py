import numpy as np
import time
import unittest
import networkx as nx
from graphik.robots.robot_base import angle_to_se2, fk_tree_2d, fk_2d_symb
from graphik.robots.revolute import (
    Revolute2dChain,
    Revolute2dTree,
)
from graphik.robots.robot_base import fk_2d

from graphik.utils.utils import list_to_variable_dict


class TestForwardKinematics(unittest.TestCase):
    def test_angle_to_se2(self):
        a = np.random.rand() + 0.5
        phi = 0.0
        T = angle_to_se2(a, phi)
        R = T.rot.as_matrix()
        t = T.trans
        self.assertTrue(np.all(np.isclose(R, np.eye(2))))
        self.assertTrue(np.all(np.isclose(t, np.array([a, 0.0]))))

        root_half = np.sqrt(2.0) / 2.0
        phi = np.pi / 4.0
        R_true = np.array([[root_half, -root_half], [root_half, root_half]])
        t_true = a * np.array([root_half, root_half])
        T = angle_to_se2(a, phi)
        R = T.rot.as_matrix()
        t = T.trans
        self.assertTrue(np.all(np.isclose(R, R_true)))
        self.assertTrue(np.all(np.isclose(t, t_true)))

        phi = np.pi / 2.0
        R_true = np.array([[0.0, -1.0], [1.0, 0.0]])
        t_true = np.array([0.0, a])
        T = angle_to_se2(a, phi)
        R = T.rot.as_matrix()
        t = T.trans
        self.assertTrue(np.all(np.isclose(R, R_true)))
        self.assertTrue(np.all(np.isclose(t, t_true)))

        phi = -np.pi / 2.0
        R_true = np.array([[0.0, 1.0], [-1.0, 0.0]])
        t_true = np.array([0.0, -a])
        T = angle_to_se2(a, phi)
        R = T.rot.as_matrix()
        t = T.trans
        self.assertTrue(np.all(np.isclose(R, R_true)))
        self.assertTrue(np.all(np.isclose(t, t_true)))

        phi = np.pi
        R_true = np.array([[-1.0, 0.0], [0.0, -1.0]])
        t_true = np.array([-a, 0.0])
        T = angle_to_se2(a, phi)
        R = T.rot.as_matrix()
        t = T.trans
        self.assertTrue(np.all(np.isclose(R, R_true)))
        self.assertTrue(np.all(np.isclose(t, t_true)))

    def test_fk_2d_two_links(self):
        a = list(np.random.rand(2) + 0.5)
        a_dict = list_to_variable_dict(a)
        theta = [0.0, 0.0]
        theta_dict = list_to_variable_dict(theta)
        lim_u = list_to_variable_dict(np.pi * np.ones(2))
        lim_l = list_to_variable_dict(-np.pi * np.ones(2))
        params = {
            "a": a_dict,
            "theta": theta_dict,
            "joint_limits_upper": lim_u,
            "joint_limits_lower": lim_l,
        }
        robot = Revolute2dChain(params)

        q = [0.0, 0.0]
        q_dict = list_to_variable_dict(q)
        T = fk_2d(a, theta, q)
        R = T.rot.as_matrix()
        t = T.trans
        R_true = np.eye(2)
        t_true = np.array([sum(a), 0.0])
        self.assertTrue(np.all(np.isclose(R, R_true)))
        self.assertTrue(np.all(np.isclose(t, t_true)))
        self.assertTrue(
            np.all(np.isclose(T.as_matrix(), robot.get_pose(q_dict, "p2").as_matrix()))
        )

        q = [np.pi / 2.0, 0.0]
        q_dict = list_to_variable_dict(q)
        T = fk_2d(a, theta, q)
        R = T.rot.as_matrix()
        t = T.trans
        R_true = np.array([[0.0, -1.0], [1.0, 0.0]])
        t_true = np.array([0.0, sum(a)])
        self.assertTrue(np.all(np.isclose(R, R_true)))
        self.assertTrue(np.all(np.isclose(t, t_true)))
        self.assertTrue(
            np.all(np.isclose(T.as_matrix(), robot.get_pose(q_dict, "p2").as_matrix()))
        )

        theta = [-np.pi / 2.0, 0.0]
        theta_dict = list_to_variable_dict(theta)
        params = {
            "a": a_dict,
            "theta": theta_dict,
            "joint_limits_upper": lim_u,
            "joint_limits_lower": lim_l,
        }
        robot = Revolute2dChain(params)
        q = [0.0, np.pi / 2.0]
        q_dict = list_to_variable_dict(q)
        T = fk_2d(a, theta, q)
        R = T.rot.as_matrix()
        t = T.trans
        R_true = np.eye(2)
        t_true = np.array([a[1], -a[0]])
        self.assertTrue(np.all(np.isclose(R, R_true)))
        self.assertTrue(np.all(np.isclose(t, t_true)))
        self.assertTrue(
            np.all(np.isclose(T.as_matrix(), robot.get_pose(q_dict, "p2").as_matrix()))
        )

        theta = [-np.pi / 4.0, 0.0]
        theta_dict = list_to_variable_dict(theta)
        params = {
            "a": a_dict,
            "theta": theta_dict,
            "joint_limits_upper": lim_u,
            "joint_limits_lower": lim_l,
        }
        robot = Revolute2dChain(params)
        q = [-np.pi / 4.0, 0.0]
        q_dict = list_to_variable_dict(q)
        T = fk_2d(a, theta, q)
        R = T.rot.as_matrix()
        t = T.trans
        R_true = np.array([[0.0, 1.0], [-1.0, 0.0]])
        t_true = np.array([0.0, -sum(a)])
        # print("R_true: {:}".format(R_true))
        # print("t_true: {:}".format(t_true))
        # print("R: {:}".format(R))
        # print("t: {:}".format(t))
        self.assertTrue(np.all(np.isclose(R, R_true)))
        self.assertTrue(np.all(np.isclose(t, t_true)))
        self.assertTrue(
            np.all(np.isclose(T.as_matrix(), robot.get_pose(q_dict, "p2").as_matrix()))
        )

    def test_fk_tree_2d(self):
        """
            Neutral position:

           3__ __4  <----
           |             |
           |             |
           1__ __2  <---- Nodes 2 and 4 are end-effectors (leaves of the tree)
           |
           |
           |
        ___0___  <-- The root is the fixed point at the origin (0., 0.)
        ///////
        """
        # parents = {1:0, 2:1, 3:1, 4:3}
        parents = {"p0": ["p1"], "p1": ["p2", "p3"], "p2": [], "p3": ["p4"], "p4": []}
        a = [1.0, 1.0, 1.0, 1.0]
        a_dict = list_to_variable_dict(a)
        theta = [np.pi / 2.0, -np.pi / 2.0, 0.0, -np.pi / 2.0]
        theta_dict = list_to_variable_dict(theta)
        q = len(a) * [0.0]
        q_dict = list_to_variable_dict(q)
        lim_u = list_to_variable_dict(np.pi * np.ones(4))
        lim_l = list_to_variable_dict(-np.pi * np.ones(4))
        params = {
            "a": a_dict,
            "theta": theta_dict,
            "parents": parents,
            "joint_limits_upper": lim_u,
            "joint_limits_lower": lim_l,
        }
        robot = Revolute2dTree(params)

        # Get neutral pose of Node 2
        path_indices = [
            0,
            1,
        ]  # This logic is awkward but handled by the get_pose() method.
        T = fk_tree_2d(a, theta, q, path_indices)
        R = T.rot.as_matrix()
        t = T.trans
        R_true = np.eye(2)
        t_true = np.array([1.0, 1.0])
        self.assertTrue(np.all(np.isclose(R, R_true)))
        self.assertTrue(np.all(np.isclose(t, t_true)))
        self.assertTrue(
            np.all(np.isclose(T.as_matrix(), robot.get_pose(q_dict, "p2").as_matrix()))
        )

        # Get neutral pose of node 3
        path_indices = [
            0,
            2,
        ]  # This logic is awkward but handled by the get_pose() method.
        T = fk_tree_2d(a, theta, q, path_indices)
        R = T.rot.as_matrix()
        t = T.trans
        R_true = np.array([[0.0, -1.0], [1.0, 0.0]])
        t_true = np.array([0.0, 2.0])
        # print("R_true: {:}".format(R_true))
        # print("t_true: {:}".format(t_true))
        # print("R: {:}".format(R))
        # print("t: {:}".format(t))
        self.assertTrue(np.all(np.isclose(R, R_true)))
        self.assertTrue(np.all(np.isclose(t, t_true)))
        self.assertTrue(
            np.all(np.isclose(T.as_matrix(), robot.get_pose(q_dict, "p3").as_matrix()))
        )
        # self.assertTrue(np.all(np.isclose(T.as_matrix(), robot.get_pose(q[0:path_indices[-1]+1]).as_matrix())),
        #                 'T: {:}, \n get pose: {:}'.format(T.as_matrix(), robot.get_pose(q[0:path_indices[-1]+1]).as_matrix()))

        # Get neutral pose of node 4
        path_indices = [
            0,
            2,
            3,
        ]  # This logic is awkward but handled by the get_pose() method.
        T = fk_tree_2d(a, theta, q, path_indices)
        R = T.rot.as_matrix()
        t = T.trans
        R_true = np.eye(2)
        t_true = np.array([1.0, 2.0])
        self.assertTrue(np.all(np.isclose(R, R_true)))
        self.assertTrue(np.all(np.isclose(t, t_true)))
        self.assertTrue(
            np.all(np.isclose(T.as_matrix(), robot.get_pose(q_dict, "p4").as_matrix()))
        )

        """
           Turn the joint at Node 1 by 90 degrees counter clockwise:

             4     2
             |     |
             |     |
             3__ __1
                   |
                   |
                   |
                ___0___  <-- The root is the fixed point at the origin (0., 0.)
                ///////
        """
        # Each joint sharing a parent is independently articulated!
        q[1] = np.pi / 2.0
        q[2] = np.pi / 2.0
        q_dict = list_to_variable_dict(q)
        # Get position of Node 2
        path_indices = [0, 1]
        T = fk_tree_2d(a, theta, q, path_indices)
        R = T.rot.as_matrix()
        t = T.trans
        R_true = np.array([[0.0, -1.0], [1.0, 0.0]])
        t_true = np.array([0.0, 2.0])
        self.assertTrue(np.all(np.isclose(R, R_true)))
        self.assertTrue(np.all(np.isclose(t, t_true)))
        self.assertTrue(
            np.all(np.isclose(T.as_matrix(), robot.get_pose(q_dict, "p2").as_matrix()))
        )

        # Get position of Node 3
        path_indices = [0, 2]
        T = fk_tree_2d(a, theta, q, path_indices)
        R = T.rot.as_matrix()
        t = T.trans
        R_true = -np.eye(2)
        t_true = np.array([-1.0, 1.0])
        # print("R_true: {:}".format(R_true))
        # print("t_true: {:}".format(t_true))
        # print("R: {:}".format(R))
        # print("t: {:}".format(t))
        self.assertTrue(np.all(np.isclose(R, R_true)))
        self.assertTrue(np.all(np.isclose(t, t_true)))
        self.assertTrue(
            np.all(np.isclose(T.as_matrix(), robot.get_pose(q_dict, "p3").as_matrix()))
        )

        # Get position of Node 4
        path_indices = [0, 2, 3]
        T = fk_tree_2d(a, theta, q, path_indices)
        R = T.rot.as_matrix()
        t = T.trans
        R_true = np.array([[0.0, -1.0], [1.0, 0.0]])
        t_true = np.array([-1.0, 2.0])
        # print("R_true: {:}".format(R_true))
        # print("t_true: {:}".format(t_true))
        # print("R: {:}".format(R))
        # print("t: {:}".format(t))
        self.assertTrue(np.all(np.isclose(R, R_true)))
        self.assertTrue(np.all(np.isclose(t, t_true)))
        self.assertTrue(
            np.all(np.isclose(T.as_matrix(), robot.get_pose(q_dict, "p4").as_matrix()))
        )

        # Now add a 'ghost' link so that they're articulated together
        a = [1.0, 0.0, 1.0, 1.0, 1.0]
        a_dict = list_to_variable_dict(a)
        theta = [np.pi / 2.0, 0.0, -np.pi / 2.0, 0.0, -np.pi / 2.0]
        theta_dict = list_to_variable_dict(theta)
        # parents = {1:0, 2:1, 3:2, 4:2, 5:4}
        parents = {
            "p0": ["p1"],
            "p1": ["p2"],
            "p2": ["p3", "p4"],
            "p3": [],
            "p4": ["p5"],
            "p5": [],
        }
        lim_u = list_to_variable_dict(np.pi * np.ones(5))
        lim_l = list_to_variable_dict(-np.pi * np.ones(5))
        params = {
            "a": a_dict,
            "theta": theta_dict,
            "parents": parents,
            "joint_limits_upper": lim_u,
            "joint_limits_lower": lim_l,
        }
        robot = Revolute2dTree(params)
        q = len(a) * [0.0]
        q[1] = np.pi / 2.0  # Only requires one articulation!
        q_dict = list_to_variable_dict(q)
        # Get position of FORMER Node 2 (now it's Node 3)
        path_indices = [0, 1, 2]
        T = fk_tree_2d(a, theta, q, path_indices)
        R = T.rot.as_matrix()
        t = T.trans
        R_true = np.array([[0.0, -1.0], [1.0, 0.0]])
        t_true = np.array([0.0, 2.0])
        self.assertTrue(np.all(np.isclose(R, R_true)))
        self.assertTrue(np.all(np.isclose(t, t_true)))
        self.assertTrue(
            np.all(np.isclose(T.as_matrix(), robot.get_pose(q_dict, "p3").as_matrix()))
        )

        # Get position of FORMER Node 3 (now it's Node 4)
        path_indices = [0, 1, 3]
        T = fk_tree_2d(a, theta, q, path_indices)
        R = T.rot.as_matrix()
        t = T.trans
        R_true = -np.eye(2)
        t_true = np.array([-1.0, 1.0])
        # print("R_true: {:}".format(R_true))
        # print("t_true: {:}".format(t_true))
        # print("R: {:}".format(R))
        # print("t: {:}".format(t))
        self.assertTrue(np.all(np.isclose(R, R_true)))
        self.assertTrue(np.all(np.isclose(t, t_true)))
        self.assertTrue(
            np.all(np.isclose(T.as_matrix(), robot.get_pose(q_dict, "p4").as_matrix()))
        )

        # Get position of FORMER Node 4 (now it's Node 5)
        path_indices = [0, 1, 3, 4]
        T = fk_tree_2d(a, theta, q, path_indices)
        R = T.rot.as_matrix()
        t = T.trans
        R_true = np.array([[0.0, -1.0], [1.0, 0.0]])
        t_true = np.array([-1.0, 2.0])
        # print("R_true: {:}".format(R_true))
        # print("t_true: {:}".format(t_true))
        # print("R: {:}".format(R))
        # print("t: {:}".format(t))
        self.assertTrue(np.all(np.isclose(R, R_true)))
        self.assertTrue(np.all(np.isclose(t, t_true)))
        self.assertTrue(
            np.all(np.isclose(T.as_matrix(), robot.get_pose(q_dict, "p5").as_matrix()))
        )


class TestLambdifiedForwardKinematics(unittest.TestCase):
    def test_simple_random_chain(self):
        n_runs = 1000
        n_links = 5
        a = list(np.random.rand(n_links) + 0.5)
        a_dict = list_to_variable_dict(a)
        theta = n_links * [0.0]
        theta_dict = list_to_variable_dict(theta)
        lim_u = list_to_variable_dict(np.pi * np.ones(n_runs))
        lim_l = list_to_variable_dict(-np.pi * np.ones(n_runs))
        params = {
            "a": a_dict,
            "theta": theta_dict,
            "joint_limits_upper": lim_u,
            "joint_limits_lower": lim_l,
        }
        robot = Revolute2dChain(params)
        robot.lambdify_get_pose()
        t_fast_total = 0.0
        t_total = 0.0
        for idx in range(n_runs):
            input = list_to_variable_dict(np.random.rand(n_links))
            t1 = time.time()
            poses_fast = robot.get_full_pose_fast_lambdify(input)
            t_fast_total += time.time() - t1
            poses = {}
            t1 = time.time()
            for key in input:
                poses[key] = robot.get_pose(input, key)
            t_total += time.time() - t1

            for key in input:
                R_true = poses[key].rot.as_matrix()
                R = poses_fast[key][0:2, 0:2]
                t_true = poses[key].trans
                t = poses_fast[key][0:2, 2]
                self.assertTrue(np.all(np.isclose(R, R_true)))
                self.assertTrue(np.all(np.isclose(t, t_true)))

        print("Fast average runtime: {:}".format(t_fast_total / n_runs))
        print("Slow average runtime: {:}".format(t_total / n_runs))

    def test_random_tree(self):
        n_runs = 100
        height = 4
        gen = nx.balanced_tree(2, height, create_using=nx.DiGraph)
        gen = nx.relabel_nodes(gen, {node: f"p{node}" for node in gen})
        n = gen.number_of_edges()
        dof = n
        print("Number of DOF: {:}".format(n))
        parents = nx.to_dict_of_lists(gen)
        a = list_to_variable_dict(np.random.rand(dof) * 2.0 + 1.0)
        th = list_to_variable_dict(np.zeros(n))
        # Generates random joint limits
        angular_limits = np.minimum(np.random.rand(dof) * np.pi + 0.25, np.pi)
        upper_angular_limits = list_to_variable_dict(angular_limits)
        lower_angular_limits = list_to_variable_dict(-angular_limits)
        params = {
            "a": a,
            "theta": th,
            "parents": parents,
            "joint_limits_upper": upper_angular_limits,
            "joint_limits_lower": lower_angular_limits,
        }
        robot = Revolute2dTree(params)
        robot.lambdify_get_pose()
        t_fast_total = 0.0
        t_total = 0.0
        for idx in range(n_runs):
            input = list_to_variable_dict(np.random.rand(n))
            t1 = time.time()
            poses_fast = robot.get_full_pose_fast_lambdify(input)
            t_fast_total += time.time() - t1
            poses = {}
            t1 = time.time()
            for key in input:
                poses[key] = robot.get_pose(input, key)
            t_total += time.time() - t1

            for key in input:
                R_true = poses[key].rot.as_matrix()
                R = poses_fast[key][0:2, 0:2]
                t_true = poses[key].trans
                t = poses_fast[key][0:2, 2]
                self.assertTrue(np.all(np.isclose(R, R_true)))
                self.assertTrue(np.all(np.isclose(t, t_true)))

        print("Fast average runtime: {:}".format(t_fast_total / n_runs))
        print("Slow average runtime: {:}".format(t_total / n_runs))


class TestFK2DSymb(unittest.TestCase):
    """
    Tests the outputs, with joint limit offsets, for 2D forward kinematics, by comparing the symbolic and procedural versions.
    """
    def test_random_params(self):
        n_runs = 100
        for idx in range(n_runs):
            dof = np.random.randint(2, 15)
            a = np.random.rand(dof) * 5.0 + 0.5
            theta = np.random.rand(dof) * 2.0 * np.pi - np.pi
            q = np.random.rand(dof) * 2.0 * np.pi - np.pi
            pose = fk_2d(a, theta, q)
            pose_symb = fk_2d_symb(a, theta, q)
            self.assertTrue(
                np.all(np.abs(pose.as_matrix() - pose_symb.as_matrix()) < 1e-8)
            )


class TestInverseKinematics(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
