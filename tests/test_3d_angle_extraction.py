#!/usr/bin/env python3
import unittest
import numpy as np
from numpy import pi
from numpy.random import rand, randint

from graphik.utils.kinematics_helpers import dh_to_se3, inverse_dh_frame


class TestInverseDH(unittest.TestCase):
    def test_simple_instance(self):
        a = 1.0
        d = 1.0
        alpha = 0.0
        theta = pi / 2.0
        T = dh_to_se3(a, alpha, d, theta)
        q1 = np.array([0.0, 1.0, 2.0])
        theta_rec, T_rec = inverse_dh_frame(q1, a, d, alpha)
        # print("DH Transform: {:}".format(T))
        # print("Theta: {:}".format(theta))
        # print("Recovered DH Transform: {:}".format(T_rec))
        # print("Recovered theta: {:}".format(theta_rec))
        self.assertAlmostEqual(theta, theta_rec)

    def test_noisy_simple_instance(self):
        a = 1.0
        d = 1.0
        alpha = 0.0
        theta = pi / 2.0
        tol = 1e-5
        T = dh_to_se3(a, alpha, d, theta)
        q1 = np.array([0.0, 1.0, 2.0]) + np.ones(3) * tol
        theta_rec, T_rec = inverse_dh_frame(q1, a, d, alpha, tol=tol)
        # print("DH Transform: {:}".format(T))
        # print("Theta: {:}".format(theta))
        # print("Recovered DH Transform: {:}".format(T_rec))
        # print("Recovered theta: {:}".format(theta_rec))
        self.assertAlmostEqual(theta, theta_rec, places=int(-np.log10(tol)) - 1)

    def test_random_instance(self):
        n_runs = 1000
        a = rand(n_runs) + 0.1
        d = rand(n_runs) + 0.1
        alpha = rand(n_runs) * 2.0 * pi - pi
        theta = rand(n_runs) * 2.0 * pi - pi
        q1 = np.array([0.0, 0.0, 1.0])
        for idx in range(n_runs):
            q1_ref_frame = dh_to_se3(a[idx], alpha[idx], d[idx], theta[idx]).dot(q1)
            q1_ref_frame = q1_ref_frame[0:3]
            theta_rec, _ = inverse_dh_frame(q1_ref_frame, a[idx], d[idx], alpha[idx])
            self.assertAlmostEqual(theta[idx], theta_rec)

    def test_random_noisy_instances(self):
        n_runs = 1000
        a = rand(n_runs) + 0.1
        d = rand(n_runs) + 0.1
        alpha = rand(n_runs) * 2.0 * pi - pi
        theta = rand(n_runs) * 2.0 * pi - pi
        tol = np.power(10, randint(-9, -5, size=n_runs).astype(float))
        q1 = np.array([0.0, 0.0, 1.0])
        for idx in range(n_runs):
            q1_ref_frame = dh_to_se3(a[idx], alpha[idx], d[idx], theta[idx]).dot(q1)
            q1_ref_frame = q1_ref_frame[0:3] + np.ones(3) * tol[idx]
            theta_rec, _ = inverse_dh_frame(q1_ref_frame, a[idx], d[idx], alpha[idx])
            self.assertAlmostEqual(
                theta[idx], theta_rec, places=int(-np.log10(tol[idx])) - 2
            )

    # def test_ur10(self):
    #     n = 6
    #     n_runs = 100
    #     # Standard DH params theta, d, alpha, a
    #     a = [0, -0.612, -0.5723, 0, 0, 0]
    #     d = [0.1273, 0, 0, 0.1639, 0.1157, 0.0922]
    #     al = [pi / 2, 0, 0, pi / 2, -pi / 2, 0]
    #     th = [0, pi, 0, 0, 0, 0]
    #     ub = pi * rand(n)
    #     lb = -ub

    #     params = {
    #         "a": a,
    #         "alpha": al,
    #         "d": d,
    #         "theta": th,
    #         "lb": lb,
    #         "ub": ub,
    #     }
    #     robot = Revolute3dChain(params)
    #     graph = Revolute3dRobotGraph(robot)

    #     for idx in range(n_runs):
    #         # q_goal = robot.random_configuration()
    #         q_goal = np.array(
    #             [
    #                 1.29875968,
    #                 0.06542301,
    #                 -1.66178522,
    #                 0.49670437,
    #                 0.13975263,
    #                 0.09546305,
    #             ]
    #         )
    #         # print("q goal: {:}".format(q_goal))
    #         X = graph.realization(q_goal)
    #         q_rec = robot.joint_angles_from_graph(X)
    #         # print("q recovered: {:}".format(q_rec))
    #         # TODO: Last point's angle not observable through this method. Avoid for now
    #         self.assertIsNone(assert_allclose(q_goal[0:-1], q_rec[0:-1], rtol=1e-5))


if __name__ == "__main__":
    # Number of joints used
    unittest.main()
