import graphik
import numpy as np
import time
import networkx as nx
import numpy.typing as npt
from typing import Dict, List, Any
from scipy.optimize import minimize
from liegroups.numpy import SE3
from numpy import pi
from graphik.utils.roboturdf import RobotURDF
from graphik.graphs.graph_base import RobotGraph
from graphik.utils import *

TOL = 1e-10

class JointAngleSolver:
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
        self.g = []
        if len(pairs) > 0:
            fun = self.gen_obstacle_constraints(pairs)
            jac = self.gen_obstacle_constraint_gradient(pairs)
            self.g = [{"type": "ineq", "fun": fun, "jac": jac}]


    # OPTION A
    def gen_cost_and_grad_ee(self, point: str, T_goal: SE3):
        # R_goal = T_goal.rot
        joints = self.k_map[point][1:]

        def cost(q: npt.ArrayLike):
            q_dict = {joints[idx]: q[idx] for idx in range(len(joints))}
            # T_all = self.robot.get_all_poses(q_dict)
            # R = T_all[point].rot
            # e_p = T_goal.trans - T_all[point].trans

            # COST 1 body linear + rotation error
            # e_o = 0.5 * (
            #     np.cross(R[0:3, 0], R_goal[0:3, 0])
            #     + np.cross(R[0:3, 1], R_goal[0:3, 1])
            #     +np.cross(R[0:3, 2], R_goal[0:3, 2])
            # )
            # L = -0.5*(
            #     # skew(R_goal[0:3, 0])@skew(R[0:3, 0])
            #     # + skew(R_goal[0:3, 1])@skew(R[0:3, 1])
            #     + skew(R_goal.as_matrix()[0:3, 2])@skew(R.as_matrix()[0:3, 2]))
            # J[point] = np.block([[np.eye(3), np.zeros((3,3))], [np.zeros((3,3)), L]])@J[point]
            # e_o = np.cross(R.as_matrix()[0:3, 2], R_goal.as_matrix()[0:3, 2])

            # COST 2 body linear + partial rotation error
            # v = np.cross(R.as_matrix()[0:3, 2], R_goal.as_matrix()[0:3, 2])
            # s = np.linalg.norm(v)
            # c = R.as_matrix()[0:3, 2]@R_goal.as_matrix()[0:3, 2]
            # R = np.eye(3) + skew(v) + (skew(v)@skew(v))*((1-c)/s**2)
            # e_o = SO3(R).log()
            # e = np.hstack([e_p, e_o])
            # J = self.robot.get_jacobian(q_dict, [point], T_all)

            # COST 3 body screw
            T = self.robot.get_pose(q_dict, point)
            J = self.robot.jacobian(q_dict, [point])
            e = (T.inv().dot(T_goal)).log()
            Ad = T.adjoint()
            e = Ad.dot(e)

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
        ZZ = np.zeros([6,6])
        ZZ[:3,:3] = np.eye(3)
        ZZ[3:,3:] = np.eye(3)
        def obstacle_gradient(q: npt.ArrayLike):
            q_dict = list_to_variable_dict(q)
            T_all = self.robot.get_all_poses(q_dict)
            J_all = self.robot.jacobian(q_dict, list(q_dict.keys()))

            jac = []
            for robot_node, obs_node in pairs:
                R = T_all[robot_node].rot.as_matrix()
                ZZ[:3,3:] = R.dot(SO3.wedge(T_all[robot_node].inv().trans)).dot(R.T)
                p = T_all[robot_node].trans
                c = self.graph.directed.nodes[obs_node][POS]
                jac += [-2 * (c - p).T @ ZZ.dot(J_all[robot_node])[:3, :]]
            return np.vstack(jac)

        return obstacle_gradient


    # OPTION B
    def gen_ee_constraint(self, point: str, T_goal: SE3):
        joints = self.k_map[point][1:]
        def ee_constraint(q: npt.ArrayLike):
            q_dict = {joints[idx]: q[idx] for idx in range(len(joints))}
            T = self.robot.get_pose(q_dict, point)
            e = (T.inv().dot(T_goal)).log()
            Ad = T.adjoint()
            e = Ad.dot(e)

            constr = np.asarray([e.T@e])
            return constr

        return ee_constraint

    def gen_ee_constraint_gradient(self, point: str, T_goal: SE3):
        # R_goal = T_goal.rot
        joints = self.k_map[point][1:]
        def ee_gradient(q: npt.ArrayLike):
            q_dict = {joints[idx]: q[idx] for idx in range(len(joints))}
            T = self.robot.get_pose(q_dict, point)
            J = self.robot.jacobian(q_dict, [point])
            e = (T.inv().dot(T_goal)).log()
            Ad = T.adjoint()
            e = Ad.dot(e)
            jac = -2 * J[point].T @ e
            return jac

        return ee_gradient

    def gen_cost(self):
        def cost(q: npt.ArrayLike):
            return q.T@q

        def jac(q: npt.ArrayLike):
            return -2*q
        return cost, jac


    def solve(self, goals: dict, q0: dict):
        for node, goal in goals.items():
            cost_and_grad = self.gen_cost_and_grad_ee(node, goal)

        constr = self.g.copy()
        # for node, goal in goals.items():
        #     fun = self.gen_ee_constraint(node, goal)
        #     jac = self.gen_ee_constraint_gradient(node, goal)
        #     constr+=[{"type": "eq", "fun": fun, "jac": jac}]
        # cost, cost_jac = self.gen_cost()

        #solve
        t = time.time()
        res = minimize(
            cost_and_grad,
            # cost,
            np.asarray(list(q0.values())),
            jac=True,
            # jac=cost_jac,
            constraints=constr,
            method="SLSQP",
            options={"ftol": 1e-7, "maxiter":200},
            # method="BFGS",
            # options={"maxiter":3000, "gtol":1e-06},
        )
        t = time.time() - t
        return list_to_variable_dict(res.x), t, res.nit

def main():
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

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    main()
