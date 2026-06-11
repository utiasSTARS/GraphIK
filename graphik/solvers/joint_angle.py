"""Joint-space IK solver using SLSQP on SE(n) log pose error."""
from __future__ import annotations

import time

import networkx as nx
import numpy as np
from scipy.optimize import minimize

from graphik.graphs.graph import ProblemGraph
from graphik.solvers import costs
from graphik.solvers.base import IKResult, IKSolver
from graphik.utils.constants import BELOW, BOUNDED, OBSTACLE, ROBOT, ROOT, TYPE


class JointAngleSolver(IKSolver):
    def __init__(self, graph: ProblemGraph):
        super().__init__(graph)
        self.n = self.robot.n
        self.joint_order = [joint for joint in self.robot.joint_ids if joint != ROOT]

        typ = nx.get_node_attributes(self.graph, name=TYPE)
        pairs = []
        for u, v, data in self.graph.edges(data=True):
            if BELOW in data.get(BOUNDED, []):
                if ROBOT in typ[u] and OBSTACLE in typ[v] and u != ROOT:
                    pairs.append((u, v))

        self.constraints = []
        if pairs:
            self.constraints = [
                {
                    "type": "ineq",
                    "fun": costs.obstacle_constraints(self.robot, self.graph, pairs),
                    "jac": costs.obstacle_constraint_gradient(
                        self.robot, self.graph, pairs
                    ),
                }
            ]

    def solve(self, T_goal, *, q_init=None) -> IKResult:
        t0 = time.perf_counter()
        goals = self.goals_from(T_goal)
        if q_init is None:
            q_init = self.robot.zero_configuration()

        goal_costs = [
            costs.pose_cost(self.robot, node, goal) for node, goal in goals.items()
        ]

        def cost_and_grad(q):
            f = 0.0
            grad = np.zeros(self.n)
            for cg in goal_costs:
                f_i, g_i = cg(q)
                f += f_i
                grad += g_i
            return f, grad

        res = minimize(
            cost_and_grad,
            np.asarray([q_init[joint] for joint in self.joint_order]),
            jac=True,
            constraints=self.constraints,
            method="SLSQP",
            options={"ftol": 1e-7},
        )
        q = {
            joint: value
            for joint, value in zip(self.joint_order, res.x)
        }
        return IKResult(
            q=q,
            cost=float(res.fun),
            iterations=int(res.nit),
            time=time.perf_counter() - t0,
            status=str(res.message),
            limit_violations=self.check_limits(q),
            Y=None,
        )
