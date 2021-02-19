import graphik
import numpy as np
import networkx as nx
import numpy.typing as npt
from typing import Dict, List, Any
from scipy.optimize import minimize
from liegroups.numpy import SE3
from numpy import pi
from graphik.utils.roboturdf import RobotURDF
from graphik.utils.constants import *
from graphik.graphs.graph_base import RobotGraph, RobotRevoluteGraph


class LocalSolver:
    def __init__(self, robot_graph: RobotGraph, params: Dict["str", Any]):
        self.graph = robot_graph
        self.robot = robot_graph.robot
        self.k_map = self.robot.kinematic_map[ROOT]  # get map to all nodes from root
        self.W = params["W"]

        # create obstacle constraints
        self.g = []
        typ = nx.get_node_attributes(self.graph.directed, name=TYPE)
        for u, v, data in self.graph.directed.edges(data=True):
            if "below" in data[BOUNDED]:
                if typ[u] == ROBOT and typ[v] == OBSTACLE and u != ROOT:
                    fun = self.gen_obstacle_constraint(u, v)
                    jac = self.gen_obstacle_constraint_gradient(u, v)
                    self.g += [{"type": "ineq", "fun": fun, "jac": jac}]
                # if typ[u] == OBSTACLE and typ[v] == ROBOT:
                #     fun = self.gen_obstacle_constraint(v, u)
                #     jac = self.gen_obstacle_constraint_gradient(v, u)
                #     self.g += [{"type": "ineq", "fun": fun, "jac": jac}]

    def gen_cost(self):
        def cost(q: npt.ArrayLike):
            return q.T @ self.W @ q

        return cost

    def gen_cost_gradient(self):
        def cost_grad(q: npt.ArrayLike):
            return 2 * self.W @ q

        return cost_grad

    def gen_obstacle_constraint(self, robot_node: str, obs_node: str):
        r = self.graph.directed[robot_node][obs_node][LOWER]
        c = self.graph.directed.nodes[obs_node][POS]
        # shape here
        joints = self.k_map[robot_node][1:]

        def obstacle_constraint(q: npt.ArrayLike):
            q_dict = {joints[idx]: q[idx] for idx in range(len(joints))}
            p = self.robot.get_pose(q_dict, robot_node).trans
            return (c - p).T @ (c - p) - r ** 2

        return obstacle_constraint

    def gen_obstacle_constraint_gradient(self, robot_node: str, obs_node: str):
        r = self.graph.directed[robot_node][obs_node][LOWER]
        c = self.graph.directed.nodes[obs_node][POS]
        # shape here
        joints = self.k_map["p6"][1:]

        def obstacle_gradient(q: npt.ArrayLike):
            q_dict = {joints[idx]: q[idx] for idx in range(len(joints))}
            p = self.robot.get_pose(q_dict, robot_node).trans
            J = self.robot.jacobian(q_dict, [robot_node])
            return -2 * (c - p) @ J[robot_node][:3, :]

        return obstacle_gradient

    def gen_end_effector_constraint(self, point: str, T_goal: SE3):
        R_goal = T_goal.rot.as_matrix()
        joints = self.k_map[point][1:]

        def ee_constr(q: npt.ArrayLike):
            q_dict = {joints[idx]: q[idx] for idx in range(len(joints))}
            T = self.robot.get_pose(q_dict, point)
            R = T.rot.as_matrix()
            e_p = T_goal.trans - T.trans
            e_o = 0.5 * (
                np.cross(R[0:3, 0], R_goal[0:3, 0])
                + np.cross(R[0:3, 1], R_goal[0:3, 1])
                + np.cross(R[0:3, 2], R_goal[0:3, 2])
            )
            return np.hstack([e_p, e_o])
            # return e_p

        return ee_constr

    def gen_end_effector_constraint_gradient(self, point: str):
        joints = self.k_map[point][1:]

        def ee_grad(q: npt.ArrayLike):
            q_dict = {joints[idx]: q[idx] for idx in range(len(joints))}
            J = self.robot.jacobian(q_dict, [point])
            return -J[point]

        return ee_grad

    def solve(self, goals: dict, q0: dict):
        cost = self.gen_cost()
        cost_grad = self.gen_cost_gradient()

        # constraints are dicts of form
        # {'type':['ineq', 'eq'], 'fun': f, 'jac':J}

        # parse goals into eq constraints
        h = []
        for node, goal in goals.items():
            fun = self.gen_end_effector_constraint(node, goal)
            jac = self.gen_end_effector_constraint_gradient(node)
            h += [{"type": "eq", "fun": fun, "jac": jac}]

        # solve
        res = minimize(
            cost,
            np.array(list(q0.values())),
            jac=cost_grad,
            constraints=h + self.g,
            method="SLSQP",
        )
        print(fun(res.x))
        return res


if __name__ == "__main__":
    fname = graphik.__path__[0] + "/robots/urdfs/ur10_mod.urdf"
    n = 6
    ub = (pi) * np.ones(n)
    lb = -ub
    urdf_robot = RobotURDF(fname)
    robot = urdf_robot.make_Revolute3d(ub, lb)  # make the Revolute class from a URDF
    graph = RobotRevoluteGraph(robot)

    obstacles = [
        (np.array([0, 1, 0.75]), 0.75),
        (np.array([0, 1, -0.75]), 0.75),
        # (np.array([0, -1, 0.75]), 0.75),
        # (np.array([0, -1, -0.75]), 0.75),
        # (np.array([1, 0, 0.75]), 0.75),
        # (np.array([1, 0, -0.75]), 0.75),
        # (np.array([-1, 0, 0.75]), 0.75),
        # (np.array([-1, 0, -0.75]), 0.75),
    ]

    for idx, obs in enumerate(obstacles):
        graph.add_spherical_obstacle(f"o{idx}", obs[0], obs[1])

    q_goal = robot.random_configuration()
    T_goal = robot.get_pose(q_goal, f"p{n}")
    goals = {f"p{n}": T_goal}

    params = {"W": np.eye(n)}
    solver = LocalSolver(graph, params)
    res = solver.solve(goals, robot.random_configuration())
    print(res)
