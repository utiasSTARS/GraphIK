#!/usr/bin/env python3
import numpy as np
import unittest
from numpy import pi
from graphik.graphs.graph_base import SphericalRobotGraph
from graphik.robots.revolute import (
    Revolute2dChain,
    Revolute2dTree,
    Revolute3dChain,
    Spherical3dChain,
    Spherical3dTree,
)
from graphik.utils.dgp_utils import distmat
from graphik.utils.utils import (
    list_to_variable_dict,
)


def sym_op(X):
    return 0.5 * (X + X.T)


def distmat_gram(X):
    H = np.outer(np.ones(X.shape[0]), np.diag(X))
    D = H + H.T - 2 * X
    return D


def lcost(Y, D_goal, F, Bl, L):
    D = distmat(Y)
    E1 = F * (D_goal - D)
    E2 = np.maximum(Bl - L * D, 0)
    return np.linalg.norm(E1) ** 2 + np.linalg.norm(E2) ** 2


def lgrad(Y, D_goal, F, Bl, L):
    Ds = F * D_goal
    D = distmat(Y)
    R = Ds - F * D  # this should be symmetric so can just to 2R instead of sym_op
    I1 = np.ones(Y.shape[0])
    dfdX = 2 * (-np.diag(I1.T @ R) + R)
    dfdY = 2 * dfdX @ Y
    Rb = np.maximum(Bl - L * D, 0)
    dfdXb = 2 * (-np.diag(I1.T @ Rb) + Rb)
    dfdYb = 2 * dfdXb @ Y
    return dfdY + dfdYb


def lhess(Y, w, D_goal, F, Bl, L):
    Ds = F * D_goal
    D = distmat(Y)
    R = Ds - F * D  # this should be symmetric so can just to 2R instead of sym_op
    I1 = np.ones(Y.shape[0])
    dDdZ = distmat_gram(Y @ w.T + w @ Y.T)  # directional der of dist matrix
    d1 = 2 * (-np.diag(I1.T @ (F * dDdZ)) + (F * dDdZ))
    d2 = 2 * (-np.diag(I1.T @ R) + R)
    Hw = -2 * d1 @ Y + 2 * d2 @ w

    Rb = np.maximum(Bl - L * D, 0)
    dDdZb = (Rb > 0).astype(int) * (-L * dDdZ)
    # dDdZb = np.maximum(-L * dDdZ, 0)
    b1 = 2 * (-np.diag(I1.T @ dDdZb) + dDdZb)
    b2 = 2 * (-np.diag(I1.T @ Rb) + Rb)
    Hwb = 2 * b1 @ Y + 2 * b2 @ w
    return Hw + Hwb


def ugrad(Y, D_goal, S):
    Ds = S * D_goal
    D = distmat(Y)
    g = np.zeros(Y.shape)
    for kdx in range(Y.shape[0]):
        p_k = Y[kdx, :]
        for jdx in range(Y.shape[0]):
            if S[kdx, jdx] and jdx != kdx:
                p_j = Y[jdx, :]
                d_kj = S[kdx, jdx] * (Ds[kdx, jdx] - np.linalg.norm(p_k - p_j) ** 2)
                g[kdx, :] += -4 * d_kj * (p_k - p_j)
    return g


def hessvec(Y, w, D_goal, S):
    Hw = np.zeros(Y.shape)
    for kdx in range(Y.shape[0]):
        p_k = Y[kdx, :]
        for idx in range(Y.shape[0]):
            if S[idx, kdx] and idx != kdx:
                p_i = Y[idx, :]
                d_ki = Ds[kdx, idx] - np.linalg.norm(p_k - p_i) ** 2
                Hw[kdx, :] += (
                    4 * d_ki * np.eye(Y.shape[1]) - 8 * (p_k - p_i).T @ (p_k - p_i)
                ) @ w[idx, :]
            elif idx == kdx:
                for jdx in range(Y.shape[0]):
                    if S[jdx, kdx] and jdx != kdx:
                        p_j = Y[jdx, :]
                        d_kj = Ds[kdx, jdx] - np.linalg.norm(p_k - p_j) ** 2
                        Hw[kdx, :] += (
                            -4 * d_kj * np.eye(Y.shape[1])
                            + 8 * (p_k - p_j).T @ (p_k - p_j)
                        ) @ w[kdx, :]
    return Hw


class TestDerivativesCost(unittest.TestCase):
    def test_random_params(self):
        seed = 8675309
        np.random.seed(seed)
        n = 10
        a = list_to_variable_dict(np.ones(n))
        th = list_to_variable_dict(np.zeros(n))
        lim_u = list_to_variable_dict(0.5 * pi * np.ones(n))
        lim_l = list_to_variable_dict(-0.5 * pi * np.ones(n))
        params = {
            "a": a,
            "theta": th,
            "joint_limits_upper": lim_u,
            "joint_limits_lower": lim_l,
        }

        robot = Revolute2dChain(params)
        graph = SphericalRobotGraph(robot)

        q_goal = robot.random_configuration()
        goals = {
            f"p{n}": robot.get_pose(q_goal, f"p{n}").trans,
            f"p{n-1}": robot.get_pose(q_goal, f"p{n-1}").trans,
        }

        q_rand = robot.random_configuration()
        for key, value in q_rand.items():
            q_rand[key] += pi / 2

        G = graph.complete_from_pos(goals)
        S = graph.adjacency_matrix(G)
        D_goal = graph.distance_matrix(q_goal)
        Y_sol = graph.pos_from_graph(graph.realization(q_goal))
        Y = graph.pos_from_graph(graph.realization(q_rand))
        # Y = Y_sol
        D = distmat(Y)
        Ds = S * D_goal

        # -------------------------------------------
        # UNBOUNDED CASE
        # -------------------------------------------
        cost = np.linalg.norm(Ds - S * D) ** 2

        g = ugrad(Y, D_goal, S)

        # numerical Hessian-vector product
        w = np.random.rand(Y.shape[0], Y.shape[1])
        r = 0.000001
        H_appr = (ugrad(Y + r * w, D_goal, S) - ugrad(Y - r * w, D_goal, S)) / (2 * r)

        R = Ds - S * D
        I1 = np.ones(Y.shape[0])
        dfdX = 2 * (-np.diag(I1.T @ R) + R)
        dfdY = 2 * dfdX @ Y

        # print(g - dfdY)

        Z = w
        dDdZ = distmat_gram(Y @ Z.T + Z @ Y.T)  # directional der of dist matrix
        d1 = 2 * (-np.diag(I1.T @ (S * dDdZ)) + (S * dDdZ))
        d2 = 2 * (-np.diag(I1.T @ R) + R)
        Hw2 = -2 * d1 @ Y + 2 * d2 @ Z
        # print(Hw2 - H_appr)

        # -------------------------------------------
        # BOUNDED CASE
        # -------------------------------------------

        Bl = graph.distance_bound_matrix()
        L = (Bl > 0).astype(int)

        cost = lcost(Y, D_goal, S, Bl, L)
        grad = lgrad(Y, D_goal, S, Bl, L)
        Hw = lhess(Y, w, D_goal, S, Bl, L)

        H_appr = (
            lgrad(Y + r * w, D_goal, S, Bl, L) - lgrad(Y - r * w, D_goal, S, Bl, L)
        ) / (2 * r)

        print(Hw - H_appr)
        assert 2 < 1

    pass
