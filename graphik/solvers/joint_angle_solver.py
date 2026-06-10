import numpy as np
import networkx as nx
from typing import Dict, Any
from scipy.optimize import minimize
from pymlg.numpy import SE3, SO3, SE2, SO2
from graphik.utils.constants import *
from graphik.graphs.graph import ProblemGraph
from graphik.utils.utils import list_to_variable_dict


class LocalSolver:
    def __init__(self, robot_graph: ProblemGraph, params: Dict["str", Any]):
        self.graph = robot_graph
        self.robot = robot_graph.robot
        self.k_map = self.robot.kinematic_map[ROOT]  # get map to all nodes from root
        self.n = self.robot.n
        self.dim = self.graph.dim

        # create obstacle constraints
        typ = nx.get_node_attributes(self.graph, name=TYPE)
        pairs = []
        for u, v, data in self.graph.edges(data=True):
            if BELOW in data.get(BOUNDED, []):
                if ROBOT in typ[u] and OBSTACLE in typ[v] and u != ROOT:
                    pairs += [(u, v)]
        self.m = len(pairs)
        self.g = []
        if len(pairs) > 0:
            fun = self.gen_obstacle_constraints(pairs)
            jac = self.gen_obstacle_constraint_gradient(pairs)
            self.g = [{"type": "ineq", "fun": fun, "jac": jac}]

    def gen_obstacle_constraints(self, pairs: list):
        def obstacle_constraint(q):
            q_dict = list_to_variable_dict(q)
            T_all = self.robot.get_all_poses(q_dict)

            constr = []
            for robot_node, obs_node in pairs:
                p = T_all[robot_node][:-1, -1]
                r = self.graph[robot_node][obs_node][LOWER]
                c = self.graph.nodes[obs_node][POS]
                constr += [(c - p).T @ (c - p) - r ** 2]
            return np.asarray(constr)

        return obstacle_constraint

    def gen_obstacle_constraint_gradient(self, pairs: list):
        if self.dim == 3:
            dim = 3
            ZZ = np.zeros([6, 6])
            ZZ[:3, 3:] = np.eye(3)
            ZZ[3:, :3] = np.eye(3)
            wedge = SO3.wedge
            inverse_se = SE3.inverse
        else:
            dim = 2
            ZZ = np.zeros([4, 4])
            ZZ[:2, 2:] = np.eye(2)
            ZZ[2:, :2] = np.eye(2)
            wedge = SO2.wedge
            inverse_se = SE2.inverse

        def obstacle_gradient(q):
            q_dict = list_to_variable_dict(q)
            T_all = self.robot.get_all_poses(q_dict)
            J_all = self.robot.jacobian(q_dict, list(q_dict.keys()))

            jac = []
            for robot_node, obs_node in pairs:
                T_node = T_all[robot_node]
                R = T_node[:dim, :dim]
                t_inv = inverse_se(T_node)[:dim, -1]
                ZZ[:dim, :dim] = R @ wedge(t_inv) @ R.T
                p = T_node[:dim, -1]
                c = self.graph.nodes[obs_node][POS]
                jac += [-2 * (c - p).T @ (ZZ @ J_all[robot_node])[:dim, :]]
            return np.vstack(jac)

        return obstacle_gradient

    def gen_cost_and_grad_ee(self, point: str, T_goal: np.ndarray):
        joints = self.k_map[point][1:]
        n = len(joints)
        if self.dim == 3:
            log = SE3.Log
            inverse = SE3.inverse
            adjoint = SE3.adjoint
            inv_left_jacobian = SE3.left_jacobian_inv
        else:
            log = SE2.Log
            inverse = SE2.inverse
            adjoint = SE2.adjoint
            inv_left_jacobian = SE2.left_jacobian_inv

        def cost(q):
            q_dict = {joints[idx]: q[idx] for idx in range(n)}
            T = self.robot.pose(q_dict, point)
            T_inv = inverse(T)
            J = self.robot.jacobian(q_dict, [point])
            e = log(T_inv @ T_goal).ravel()
            J_e = inv_left_jacobian(e)
            J[point] = J_e @ adjoint(T_inv) @ J[point]
            jac = -2 * J[point].T @ e
            return e.T @ e, jac

        return cost

    def solve(self, goals: dict, q0: dict):
        # Each per-goal cost reads the leading entries of q along its
        # root-to-goal chain (q0 must be ordered along the kinematic chain)
        # and returns a full-width gradient (robot.jacobian pads with zero
        # columns past the chain), so goals simply sum.
        goal_costs = [
            self.gen_cost_and_grad_ee(node, goal) for node, goal in goals.items()
        ]

        def cost_and_grad(q):
            f = 0.0
            grad = np.zeros(self.robot.n)
            for cg in goal_costs:
                f_i, g_i = cg(q)
                f += f_i
                grad += g_i
            return f, grad

        res = minimize(
            cost_and_grad,
            np.asarray(list(q0.values())),
            jac=True,
            constraints=self.g,
            method="SLSQP",
            options={"ftol": 1e-7},
        )
        return res
