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
from graphik.graphs.graph_base import RobotGraph
from graphik.utils.utils import list_to_variable_dict


class LocalSolver:
    def __init__(self, robot_graph: RobotGraph, params: Dict["str", Any]):
        self.graph = robot_graph
        self.robot = robot_graph.robot
        self.k_map = self.robot.kinematic_map[ROOT]  # get map to all nodes from root
        self.W = params["W"]

        # create obstacle constraints
        typ = nx.get_node_attributes(self.graph.directed, name=TYPE)
        pairs = []
        for u, v, data in self.graph.directed.edges(data=True):
            if "below" in data[BOUNDED]:
                if typ[u] == ROBOT and typ[v] == OBSTACLE and u != ROOT:
                    pairs += [(u, v)]
                # if typ[u] == OBSTACLE and typ[v] == ROBOT:
                #     fun = self.gen_obstacle_constraint(v, u)
                #     jac = self.gen_obstacle_constraint_gradient(v, u)
                #     self.g += [{"type": "ineq", "fun": fun, "jac": jac}]
        self.g = []
        if len(pairs) > 0:
            fun = self.gen_obstacle_constraints(pairs)
            jac = self.gen_obstacle_constraint_gradient(pairs)
            self.g = [{"type": "ineq", "fun": fun, "jac": jac}]

    def gen_cost_and_grad_ee(self, point: str, T_goal: SE3):
        R_goal = T_goal.rot.as_matrix()
        joints = self.k_map[point][1:]

        def cost(q: npt.ArrayLike):
            q_dict = {joints[idx]: q[idx] for idx in range(len(joints))}
            T_all = self.robot.get_all_poses(q_dict)
            R = T_all[point].rot.as_matrix()
            e_p = T_goal.trans - T_all[point].trans
            e_o = 0.5 * (
                # np.cross(R[0:3, 0], R_goal[0:3, 0])
                # + np.cross(R[0:3, 1], R_goal[0:3, 1])
                +np.cross(R[0:3, 2], R_goal[0:3, 2])
            )
            e = np.hstack([e_p, e_o])

            J = self.robot.jacobian(q_dict, [point], T_all)
            jac = -2 * J[point].T @ e
            return e.T @ e, jac

        return cost

    def gen_obstacle_constraints(self, pairs: list):
        def obstacle_constraint(q: npt.ArrayLike):
            q_dict = list_to_variable_dict(q)
            T_all = self.robot.get_all_poses(q_dict)

            constr = []
            for robot_node, obs_node in pairs:
                p = T_all[robot_node].trans
                r = self.graph.directed[robot_node][obs_node][LOWER]
                c = self.graph.directed.nodes[obs_node][POS]
                constr += [(c - p).T @ (c - p) - r ** 2]
            return np.asarray(constr)

        return obstacle_constraint

    def gen_obstacle_constraint_gradient(self, pairs: list):
        def obstacle_gradient(q: npt.ArrayLike):
            q_dict = list_to_variable_dict(q)
            T_all = self.robot.get_all_poses(q_dict)
            J_all = self.robot.jacobian(q_dict, list(q_dict.keys()), T_all)

            jac = []
            for robot_node, obs_node in pairs:
                p = T_all[robot_node].trans
                c = self.graph.directed.nodes[obs_node][POS]
                jac += [-2 * (c - p) @ J_all[robot_node][:3, :]]
            return np.vstack(jac)

        return obstacle_gradient

    def solve(self, goals: dict, q0: dict):
        for node, goal in goals.items():
            cost_and_grad = self.gen_cost_and_grad_ee(node, goal)

        # solve
        res = minimize(
            cost_and_grad,
            np.asarray(list(q0.values())),
            jac=True,
            constraints=self.g,
            method="SLSQP",
            options={"ftol": 1e-7},
        )
        return res


if __name__ == "__main__":
    fname = graphik.__path__[0] + "/robots/urdfs/ur10_mod.urdf"
    n = 6
    ub = (pi) * np.ones(n)
    lb = -ub
    urdf_robot = RobotURDF(fname)
    robot = urdf_robot.make_Revolute3d(ub, lb)  # make the Revolute class from a URDF
    import timeit

    f = lambda: robot.jacobian(robot.random_configuration(), ["p6"])

    # print(timeit.timeit(f, number=1000))
    print(
        max(
            timeit.repeat(
                "robot.get_all_poses(robot.random_configuration())",
                globals=globals(),
                number=1,
                repeat=1000,
            )
        )
    )
    print(
        max(
            timeit.repeat(
                'robot.jacobian(robot.random_configuration(), ["p6"])',
                globals=globals(),
                number=1,
                repeat=1000,
            )
        )
    )

    # graph = RobotRevoluteGraph(robot)

    # obstacles = [
    #     (np.array([0, 1, 0.75]), 0.75),
    #     (np.array([0, 1, -0.75]), 0.75),
    #     # (np.array([0, -1, 0.75]), 0.75),
    #     # (np.array([0, -1, -0.75]), 0.75),
    #     # (np.array([1, 0, 0.75]), 0.75),
    #     # (np.array([1, 0, -0.75]), 0.75),
    #     # (np.array([-1, 0, 0.75]), 0.75),
    #     # (np.array([-1, 0, -0.75]), 0.75),
    # ]

    # for idx, obs in enumerate(obstacles):
    #     graph.add_spherical_obstacle(f"o{idx}", obs[0], obs[1])

    # q_goal = robot.random_configuration()
    # T_goal = robot.get_pose(q_goal, f"p{n}")
    # goals = {f"p{n}": T_goal}

    # params = {"W": np.eye(n)}
    # solver = LocalSolver(graph, params)
    # res = solver.solve(goals, robot.random_configuration())
    # print(res)
