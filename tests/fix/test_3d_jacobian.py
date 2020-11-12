from graphik.robots.robot_base import RobotRevolute
import numpy as np
from numpy import pi
import unittest
import networkx as nx
from graphik.robots.revolute import (
    Revolute3dChain,
    Revolute3dTree,
    Spherical3dChain,
    Spherical3dTree,
)
from graphik.solvers.geometric_jacobian import planar_jacobian

from graphik.utils.utils import list_to_variable_dict, list_to_variable_dict_spherical


class TestLambdifiedForwardKinematics(unittest.TestCase):
    def test_simple_random_chain(self):
        # n = 6
        # a = [0, -0.612, -0.5723, 0, 0, 0]
        # d = [0.1273, 0, 0, 0.1639, 0.1157, 0.0922]
        # al = [pi / 2, 0, 0, pi / 2, -pi / 2, 0]
        # th = [0, pi, 0, 0, 0, 0]
        # ub = (pi / 4) * np.ones(n)
        # lb = -ub
        # modified_dh = False

        n = 7
        a = [0, 0, 0, 0, 0, 0, 0]
        d = [0, 0, 0.4, 0, 0.39, 0, 0]
        al = [pi / 2, -pi / 2, -pi / 2, pi / 2, pi / 2, -pi / 2, 0]
        th = [0, 0, 0, 0, 0, 0, 0]
        ub = (pi) * np.ones(n)
        lb = -ub
        modified_dh = True

        params = {
            "a": a,
            "alpha": al,
            "d": d,
            "theta": th,
            "lb": lb,
            "ub": ub,
            "modified_dh": modified_dh,
        }
        robot = RobotRevolute(params)  # instantiate robot
        # robot.lambdify_get_pose()
        q = robot.random_configuration()
        J = robot.jacobian(q, f"p{n}")
        J2 = planar_jacobian(robot, list(q.values()), f"p{n}")
        print(J - J2)

        H = robot.hessian(q, f"p{n}")
        # print(H[0, :, :, 0])

        H_num = np.zeros([1, 6, n, n])
        eps = 1e-6
        for idx in range(n):
            q_p = q
            q_p[f"p{idx+1}"] = q[f"p{idx+1}"] + eps
            J_p = robot.jacobian(q_p, f"p{n}")

            q_m = q
            q_m[f"p{idx+1}"] = q[f"p{idx+1}"] - eps
            J_m = robot.jacobian(q_m, f"p{n}")

            dJ = (J_p - J_m) / (eps)

            for jdx in range(n):
                if idx <= jdx:
                    H_num[0, :, idx, jdx] = dJ[:, jdx]
                    H_num[0, :, jdx, idx] = dJ[:, jdx]

        # print(H_num[0, :, :, 0])
        # print(H_num[0, :, 0, 0])
        # print(H[0, :, 0, 0])
        print(H[0, :, :, :] - H_num[0, :, :, :])


if __name__ == "__main__":
    test = TestLambdifiedForwardKinematics()
    test.test_simple_random_chain()
