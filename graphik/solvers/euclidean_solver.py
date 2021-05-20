from graphik.utils.utils import list_to_variable_dict
import time
import graphik
import numpy as np
import networkx as nx
import numpy.typing as npt
from graphik.graphs.graph_revolute import RobotRevoluteGraph
from graphik.utils.roboturdf import RobotURDF
from typing import Dict, List, Any
from numpy import pi
from scipy.optimize import minimize
from graphik.utils import *
from graphik.graphs.graph_base import RobotGraph
from graphik.utils.manifolds.fixed_rank_psd_sym import PSDFixedRank
from scipy.optimize import NonlinearConstraint
from graphik.solvers.costgrd import jcost_and_grad, jhess

class EuclideanSolver:
    def __init__(self, robot_graph: RobotGraph, params: Dict["str", Any]):
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

        G = self.graph.complete_from_pos(goals)
        self.omega = adjacency_matrix_from_graph(G)

        typ = nx.get_node_attributes(self.graph.directed, name=TYPE)
        pairs = []
        for u, v, data in self.graph.directed.edges(data=True):
            if "below" in data[BOUNDED]:
                if typ[u] == ROBOT and typ[v] == OBSTACLE and u != ROOT:
                    pairs += [(u, v)]

        if self.method is not "trust-exact":
            self.g = []
            if len(pairs) > 0:
                fun = self.gen_distance_constraints(pairs)
                jac = self.gen_distance_constraints_gradient(pairs)
                self.g = [{"type": "ineq", "fun": fun, "jac": jac}]
        else:
            if len(pairs) > 0:
                fun = self.gen_distance_constraints(pairs)
                jac = self.gen_distance_constraints_gradient(pairs)
                self.g = NonlinearConstraint(fun,0,np.inf,jac)


    def gen_cost_and_grad(self, D_goal: npt.ArrayLike):
        omega = self.omega
        inds = np.nonzero(np.triu(omega))
        N = self.graph.n_nodes
        dim = self.graph.dim

        def cost_and_grad(Y: npt.ArrayLike):
            Y = Y.reshape(N, dim)
            cost, grad = jcost_and_grad(Y, D_goal,inds)
            return cost, grad.flatten()
            # D = distance_matrix_from_pos(Y)
            # S = omega * (D_goal - D)
            # f = np.linalg.norm(S) ** 2
            # dfdY = 4 * (S - np.diag(np.sum(S, axis=1))).dot(Y)
            # return f, dfdY.flatten()

        return cost_and_grad

    def gen_hessv(self, D_goal: npt.ArrayLike):
        omega = self.omega
        inds = np.nonzero(np.triu(omega))
        N = self.graph.n_nodes
        dim = self.graph.dim

        def hessv(Y: npt.ArrayLike, Z: npt.ArrayLike):
            Y = Y.reshape(N, dim)
            Z = Z.reshape(N, dim)
            HZ = jhess(Y,Z,D_goal,inds)
            # D = distance_matrix_from_pos(Y)

            # S = omega * (D_goal - D)
            # dDdZ = distance_matrix_from_gram(Y.dot(Z.T) + Z.dot(Y.T))
            # dSdZ = -omega * dDdZ
            # d1 = 4 * (dSdZ - np.diag(np.sum(dSdZ, axis=1))).dot(Y)
            # d2 = 4 * (S - np.diag(np.sum(S, axis=1))).dot(Z)
            # HZ = d1 + d2
            return HZ.flatten()

        def hessv_rtr(Y: npt.ArrayLike, Z: npt.ArrayLike):
            Y = Y.reshape(N, dim)
            Z = Z.reshape(N, dim)
            D = distance_matrix_from_pos(Y)

            S = omega * (D_goal - D)
            dDdZ = distance_matrix_from_gram(Y.dot(Z.T) + Z.dot(Y.T))
            dSdZ = -omega * dDdZ
            d1 = 4 * (dSdZ - np.diag(np.sum(dSdZ, axis=1))).dot(Y)
            d2 = 4 * (S - np.diag(np.sum(S, axis=1))).dot(Z)
            H = d1 + d2

            return PSDFixedRank.proj(Y,H).flatten()

        return hessv, hessv_rtr

    def gen_distance_constraints(self, pairs):
        N = self.graph.n_nodes
        dim = self.graph.dim
        ind = {}
        for u,v in pairs: # hash the indices
            ind[u] = self.graph.node_ids.index(u)
            ind[v] = self.graph.node_ids.index(v)

        def distance_constraints(Y: npt.ArrayLike):
            Y = Y.reshape(N, dim)
            D = distance_matrix_from_pos(Y)
            constr = []
            for u, v in pairs:
                idxu = ind[u]
                idxv = ind[v]
                if LOWER in self.graph.directed[u][v]:
                    l = self.graph.directed[u][v][LOWER]
                    constr += [-l**2 + D[idxu, idxv]]
                if UPPER in self.graph.directed[u][v]:
                    u = self.graph.directed[u][v][UPPER]
                    constr += [u**2 - D[idxu, idxv]]

            return np.asarray(constr)

        return distance_constraints

    def gen_distance_constraints_gradient(self, pairs):
        N = self.graph.n_nodes
        dim = self.graph.dim
        ind = {}
        for u,v in pairs:
            ind[u] = self.graph.node_ids.index(u)
            ind[v] = self.graph.node_ids.index(v)

        def distance_constraints_gradient(Y: npt.ArrayLike):
            Y = Y.reshape(N, dim)
            jac = []
            for u, v in pairs:
                idxu = ind[u]
                idxv = ind[v]
                if LOWER in self.graph.directed[u][v]:
                    grad_l = np.zeros(Y.shape)
                    grad_l[idxu,:] = 2*(Y[idxu,:]-Y[idxv,:])
                    grad_l[idxv,:] = 2*(-Y[idxu,:]+Y[idxv,:])
                    jac += [grad_l.flatten()]
                if UPPER in self.graph.directed[u][v]:
                    grad_u = np.zeros(Y.shape)
                    grad_u[idxu,:] = -2*(Y[idxu,:]-Y[idxv,:])
                    grad_u[idxv,:] = -2*(-Y[idxu,:]+Y[idxv,:])
                    jac += [grad_u.flatten()]
            return np.vstack(jac)

        return distance_constraints_gradient


    def solve(
        self,
        D_goal: npt.ArrayLike,
        joint_limits=False,
        Y_init=None,
        output_log=True,
    ):
        cost_and_grad = self.gen_cost_and_grad(D_goal)
        hessv, hessv_rtr = self.gen_hessv(D_goal)

        t = time.time()
        res = minimize(
            cost_and_grad,
            Y_init,
            jac=True,
            constraints=self.g,
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
    graph = RobotRevoluteGraph(robot)
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
