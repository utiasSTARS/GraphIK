import time
import graphik
import numpy as np
import networkx as nx
import numpy.typing as npt
from graphik.graphs.graph_revolute import ProblemGraphRevolute
from graphik.utils.roboturdf import RobotURDF
from typing import Dict, List, Any
from numpy import pi
from scipy.optimize import minimize
from graphik.utils import *
from graphik.graphs.graph_base import ProblemGraph
from graphik.solvers.costgrd import lcost, lgrad, lhess, lcost_and_grad

class EuclideanSolver:
    def __init__(self, robot_graph: ProblemGraph, params: Dict["str", Any]):
        self.method = params.get("method","SLSQP")
        self.options = params.get("options",{"ftol": 1e-12, "maxiter":2000})

        self.graph = robot_graph
        self.robot = robot_graph.robot
        self.k_map = self.robot.kinematic_map[ROOT]  # get map to all nodes from root

        # generate omega (the matrix defining the IK problem)
        goals = {}
        for u, v in self.robot.end_effectors:
            goals[u] = np.ones(self.graph.dim)
            goals[v] = np.ones(self.graph.dim)

        G = self.graph.from_pos(goals)
        self.omega = adjacency_matrix_from_graph(G)
        self.psi_L, self.psi_U = self.graph.distance_bound_matrices()

    def gen_lcost(self, D_goal: npt.ArrayLike):
        omega = self.omega
        psi_L, psi_U = self.psi_L, self.psi_U
        # inds = np.nonzero(np.triu(omega) + np.triu(psi_L>0) + np.triu(psi_U>0))
        inds = np.nonzero(np.triu(omega) + np.triu(psi_L!=psi_U))
        N = self.graph.n_nodes
        dim = self.graph.dim

        def cost(Y: npt.ArrayLike):
            Y = Y.reshape(N, dim)
            cost = lcost(Y, D_goal, omega, psi_L, psi_U, inds)
            return cost
        return cost

    def gen_lgrad(self, D_goal: npt.ArrayLike):
        omega = self.omega
        psi_L, psi_U = self.psi_L, self.psi_U
        # inds = np.nonzero(np.triu(omega) + np.triu(psi_L>0) + np.triu(psi_U>0))
        inds = np.nonzero(np.triu(omega) + np.triu(psi_L!=psi_U))
        N = self.graph.n_nodes
        dim = self.graph.dim

        def grad(Y: npt.ArrayLike):
            Y = Y.reshape(N, dim)
            grad = lgrad(Y, D_goal, omega, psi_L, psi_U, inds)
            return grad.flatten()

        return grad

    def gen_lcost_and_grad(self, D_goal: npt.ArrayLike):
        omega = self.omega
        psi_L, psi_U = self.psi_L, self.psi_U
        inds = np.nonzero(np.triu(omega) + np.triu(psi_L!=psi_U))
        N = self.graph.n_nodes
        dim = self.graph.dim

        def cost_and_grad(Y: npt.ArrayLike):
            Y = Y.reshape(N, dim)
            cost, grad = lcost_and_grad(Y, D_goal, omega, psi_L, psi_U, inds)
            return cost, grad.flatten()

        return cost_and_grad

    def gen_lhessv(self, D_goal: npt.ArrayLike):
        omega = self.omega
        psi_L, psi_U = self.psi_L, self.psi_U
        inds = np.nonzero(np.triu(omega) + np.triu(psi_L!=psi_U))
        N = self.graph.n_nodes
        dim = self.graph.dim

        def hessv(Y: npt.ArrayLike, Z: npt.ArrayLike):
            Y = Y.reshape(N, dim)
            Z = Z.reshape(N, dim)
            HZ = lhess(Y, Z, D_goal, omega, psi_L, psi_U, inds)
            return HZ.flatten()

        return hessv

    def solve(
        self,
        D_goal: npt.ArrayLike,
        joint_limits=False,
        Y_init=None,
        output_log=True,
    ):

        cost= self.gen_lcost(D_goal)
        jac = self.gen_lgrad(D_goal)
        hessv = self.gen_lhessv(D_goal)

        t = time.time()
        res = minimize(
            cost,
            Y_init.flatten(),
            jac=jac,
            hessp=hessv,
            method=self.method,
            options=self.options,
        )
        t = time.time() - t

        N = self.graph.n_nodes
        dim = self.graph.dim
        Y_sol = np.reshape(res.x, (N, dim))
        return Y_sol, t, res.nit


if __name__ == "__main__":

    fname = graphik.__path__[0] + "/robots/urdfs/ur10_mod.urdf"
    n = 6
    ub = (pi) * np.ones(n)
    lb = -ub
    urdf_robot = RobotURDF(fname)
    robot = urdf_robot.make_Revolute3d(ub, lb)  # make the Revolute class from a URDF
    graph = ProblemGraphRevolute(robot)
    q_goal = graph.robot.random_configuration()
    D_goal = graph.distance_matrix_from_joints(q_goal)
    q_rand = list_to_variable_dict(robot.n * [0])
    G_rand = graph.realization(q_rand)
    Y_init = pos_from_graph(G_rand)
    solver = EuclideanSolver(graph, {})

    t = time.time()
    res = solver.solve(D_goal, Y_init=Y_init)
    print(time.time() - t)
    print(res)

    # psi_L, psi_U = self.psi_L, self.psi_U
    # psi_L = np.triu(psi_L,1).flatten
    # psi_U = np.triu(psi_U,1)
    # L = np.triu((psi_L>0) * (psi_L - D),1).flatten() # remove nonzero
    # U = np.triu(-(psi_U>0) * (psi_U - D),1).flatten()
    # jac = []
    # for nonzero in range(a):
    #     grd = 2*(0)
    #     pass

    # L = np.triu((psi_L>0) * (psi_L - D),1).flatten()
    # L = L[L!=0] # remove nonzero elements
    # U = np.triu(-(psi_U>0) * (psi_U - D),1).flatten()
    # U = U[U!=0]
    # constr = np.concatenate(L, U) # every element is a <= 0 constraint
