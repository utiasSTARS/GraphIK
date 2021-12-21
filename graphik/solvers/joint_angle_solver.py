from graphik.utils.geometry import skew
import graphik
import numpy as np
import networkx as nx
from numpy.typing import ArrayLike
from typing import Dict, List, Any, Union
from scipy.optimize import minimize
from liegroups.numpy import SE3, SO3, SE2, SO2
from numpy import pi
from graphik.utils.roboturdf import RobotURDF
from graphik.utils.constants import *
from graphik.graphs.graph_base import ProblemGraph
from graphik.utils.utils import list_to_variable_dict
from graphik.utils.roboturdf import load_kuka, load_ur10

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
            if "below" in data[BOUNDED]:
                if typ[u] == ROBOT and typ[v] == OBSTACLE and u != ROOT:
                    pairs += [(u, v)]
        self.m = len(pairs)
        self.g = []
        if len(pairs) > 0:
            fun = self.gen_obstacle_constraints(pairs)
            jac = self.gen_obstacle_constraint_gradient(pairs)
            self.g = [{"type": "ineq", "fun": fun, "jac": jac}]
            # self.g = [{"type": "ineq", "fun": fun}]

    def gen_objective_ee(self, point: str, T_goal: Union[SE3, SE2]):
        joints = self.k_map[point][1:]
        n = len(joints)

        if self.dim==3:
            log = SE3.log
        else:
            log = SE2.log

        def objective(q: ArrayLike):
            q_dict = {joints[idx]: q[idx] for idx in range(n)}
            T = self.robot.pose(q_dict, point)
            e = log(T.inv().dot(T_goal)) # body frame
            return e.T @ e

        return objective


    def gen_grad_ee(self, point: str, T_goal: SE3):
        joints = self.k_map[point][1:]
        n = len(joints)

        if self.dim==3:


            log = SE3.log
            inv_left_jacobian = SE3.inv_left_jacobian
        else:
            log = SE2.log
            inv_left_jacobian = SE2.inv_left_jacobian

        def gradient(q: ArrayLike):
            # gradient of objective
            q_dict = {joints[idx]: q[idx] for idx in range(n)}
            T = self.robot.pose(q_dict, point)
            J = self.robot.jacobian(q_dict, [point])
            del_T =T.inv().dot(T_goal)
            e = log(del_T) # body frame
            J_e = inv_left_jacobian(e)
            J[point] = J_e @ T.inv().adjoint() @ J[point]
            jac = -2 * J[point].T  @ e
            return jac
        return gradient

    def gen_obstacle_constraints(self, pairs: list):
        def obstacle_constraint(q: ArrayLike):
            q_dict = list_to_variable_dict(q)
            T_all = self.robot.get_all_poses(q_dict)

            constr = []
            for robot_node, obs_node in pairs:
                p = T_all[robot_node].trans
                r = self.graph[robot_node][obs_node][LOWER]
                c = self.graph.nodes[obs_node][POS]
                constr += [(c - p).T @ (c - p) - r ** 2]
            return np.asarray(constr)

        return obstacle_constraint

    def gen_obstacle_constraint_gradient(self, pairs: list):
        if self.dim==3:
            dim = 3
            ZZ = np.zeros([6,6])
            ZZ[:3,:3] = np.eye(3)
            ZZ[3:,3:] = np.eye(3)
            wedge = SO3.wedge
        else:
            dim = 2
            ZZ = np.zeros([4,4])
            ZZ[:2,:2] = np.eye(2)
            ZZ[2:,2:] = np.eye(2)
            wedge = SO2.wedge

        def obstacle_gradient(q: ArrayLike):
            q_dict = list_to_variable_dict(q)
            T_all = self.robot.get_all_poses(q_dict)
            J_all = self.robot.jacobian(q_dict, list(q_dict.keys()))

            jac = []
            for robot_node, obs_node in pairs:
                R = T_all[robot_node].rot.as_matrix()
                ZZ[:dim,dim:] = R.dot(wedge(T_all[robot_node].inv().trans)).dot(R.T)
                p = T_all[robot_node].trans
                c = self.graph.nodes[obs_node][POS]
                jac += [-2 * (c - p).T @ ZZ.dot(J_all[robot_node])[:dim, :]]
            return np.vstack(jac)

        return obstacle_gradient

    def gen_cost_and_grad_ee(self, point: str, T_goal: SE3):
        # R_goal = T_goal.rot.as_matrix()
        joints = self.k_map[point][1:]
        n = len(joints)
        if self.dim==3:
            log = SE3.log
            inv_left_jacobian = SE3.inv_left_jacobian
        else:
            log = SE2.log
            inv_left_jacobian = lambda x: np.eye(3)

        def cost(q: ArrayLike):
            q_dict = {joints[idx]: q[idx] for idx in range(n)}
            T = self.robot.pose(q_dict, point)
            J = self.robot.jacobian(q_dict, [point])
            del_T =T.inv().dot(T_goal)
            e = log(del_T) # body frame
            J_e = inv_left_jacobian(e)
            J[point] = J_e @ T.inv().adjoint() @ J[point]
            jac = -2 * J[point].T  @ e
            return e.T @ e, jac

        return cost

    def solve(self, goals: dict, q0: dict):
        for node, goal in goals.items():
            cost_and_grad = self.gen_cost_and_grad_ee(node, goal)

        # solve
        res = minimize(
            cost_and_grad,
            # cost,
            np.asarray(list(q0.values())),
            jac=True,
            # jac=grad,
            constraints=self.g,
            method="SLSQP",
            options={"ftol": 1e-7},
        )
        return res


def main():
    #
    # Define the problem
    #

    # robot, graph = load_ur10()
    # scale = 0.75
    # radius = 0.4
    # obstacles = [
    #     (scale * np.asarray([1, 1, 0]), radius),
    #     (scale * np.asarray([1, -1, 0]), radius),
    #     (scale * np.asarray([-1, 1, 0]), radius),
    #     (scale * np.asarray([-1, -1, 0]), radius),
    #     (scale * np.asarray([0, 0, 1]), radius),
    #     (scale * np.asarray([0, 0, -1]), radius),
    # ]

    # for idx, obs in enumerate(obstacles):
    #     graph.add_spherical_obstacle(f"o{idx}", obs[0], obs[1])

    # q_goal = robot.random_configuration()
    # T_goal = robot.pose(q_goal, f"p{robot.n}")
    # goals = {f"p{robot.n}": T_goal}

    # x0 = [0,0,0,0,0,0]
    # problem = LocalSolver(graph,{})
    # sol = problem.solve(goals, robot.random_configuration())
    # # sol = problem.solve(goals, list_to_variable_dict(x0))
    # print(sol)

    from graphik.robots import RobotPlanar
    from graphik.graphs import ProblemGraphPlanar
    n = 10

    a = list_to_variable_dict(np.ones(n))
    th = list_to_variable_dict(np.zeros(n))
    lim_u = list_to_variable_dict(np.pi * np.ones(n))
    lim_l = list_to_variable_dict(-np.pi * np.ones(n))
    params = {
        "link_lengths": a,
        "theta": th,
        "ub": lim_u,
        "lb": lim_l,
        "num_joints": n
    }

    robot = RobotPlanar(params)
    graph = ProblemGraphPlanar(robot)
    q_goal = robot.random_configuration()
    T_goal = robot.pose(q_goal, f"p{robot.n}")
    goals = {f"p{robot.n}": T_goal}
    problem = LocalSolver(graph,{})
    sol = problem.solve(goals, robot.random_configuration())
    q_sol = list_to_variable_dict(sol.x)
    print(T_goal)
    print(robot.pose(q_sol, robot.end_effectors[0]))

if __name__ == '__main__':

    # np.random.seed(24)  # TODO: this seems to have a significant effect on performance
    main()
