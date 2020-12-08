import numpy as np
import sympy as sp
import cvxpy as cp
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import mosek
from cvxik.utils.utils import list_to_variable_dict, constraint_violations, measure_perturbation
from cvxik.solvers.constraints import constraints_from_graph, nearest_neighbour_cost, get_full_revolute_nearest_point
from cvxik.solvers.sdp_formulations import SdpSolverParams
from cvxik.solvers.solver_base import SdpRelaxationSolver
from cvxik.graphs.graph_base import load_ur10, Revolute3dRobotGraph
from cvxik.utils.sdp_experiments import sdp_sol_to_point_matrix
from cvxik.robots.revolute import RobotRevolute

from progress.bar import ShadyBar as Bar
from matplotlib import rc
rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 18})
rc("text", usetex=True)


if __name__ == '__main__':
    # robot, graph = load_ur10()
    # n = robot.n

    #  Test partial UR10
    n = 4
    dof = n
    #  TODO: trying exact rational values for M2
    a_full = [0, -5, -5, 0, 0, 0]  # [0, -0.612, -0.5723, 0, 0, 0]
    d_full = [1, 0, 0, 1, 1, 1]  # [0.1273, 0, 0, 0.1639, 0.1157, 0.0922]
    al_full = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]
    th_full = [0, 0, 0, 0, 0, 0]
    a = a_full[0:n]
    d = d_full[0:n]
    al = al_full[0:n]
    th = th_full[0:n]
    ub = (np.pi) * np.ones(n)
    lb = -ub
    ub = np.minimum(np.random.rand(n) * (np.pi / 2) + np.pi / 2, np.pi)
    lb = -ub
    modified_dh = False
    params = {
        "a": a[:n],
        "alpha": al[:n],
        "d": d[:n],
        "theta": th[:n],
        "lb": lb[:n],
        "ub": ub[:n],
        "modified_dh": modified_dh,
    }
    robot = RobotRevolute(params)
    graph = Revolute3dRobotGraph(robot)

    q = robot.random_configuration()

    #  This is used for the Macaulay2 outputs (which are not very useful, ran out of RAM for simple n=3 (4?) case)
    # full_input = np.zeros(n)
    # full_input = [0, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2]
    # input = full_input[:n]
    # q = list_to_variable_dict(input)

    G = graph.realization(q)
    P = graph.pos_from_graph(G)
    ee_goals = {}
    for pair in graph.robot.end_effectors:
        for ee in pair:
            if 'p' in ee:
                ee_goals[ee] = graph.robot.get_pose(q, ee).trans
            else:
                for idx, key in enumerate(graph.node_ids):
                    if key == ee:
                        ee_goals[ee] = P[idx, :]

    # Form the big Jacobian (symbolically, perhaps?)
    constraints = constraints_from_graph(graph, ee_goals)
    vars = set()

    for cons in constraints:
        vars = vars.union(cons.free_symbols)

    N = len(vars)
    constraints_matrix = sp.Matrix([cons.args[0] - cons.args[1] for cons in constraints])
    J = constraints_matrix.jacobian(list(vars))

    # Substitute in the values at any feasible point to see if it is 'regular' (i.e., satisfies ACQ)
    subs_vals = {}

    for idx in range(1, N//6 + 1):
        p_idx = graph.robot.get_pose(q, f'p{idx}').trans
        subs_vals[sp.Symbol(f'p{idx}_x')] = p_idx[0]
        subs_vals[sp.Symbol(f'p{idx}_y')] = p_idx[1]
        subs_vals[sp.Symbol(f'p{idx}_z')] = p_idx[2]

    for idx, key in enumerate(graph.node_ids):
        if 'q' in key:
            subs_vals[sp.Symbol(key + '_x')] = P[idx, 0]
            subs_vals[sp.Symbol(key + '_y')] = P[idx, 1]
            subs_vals[sp.Symbol(key + '_z')] = P[idx, 2]

    J_subbed = J.subs(subs_vals)

    _, s, _ = np.linalg.svd(np.array(J_subbed).astype(float))
    rank_J_subbed = np.sum(np.abs(s) > 1e-6)  # 21

    #  We want ACQ to hold: rank_J_subbed = n - dim(variety)
    #                                  21 = 30 - dim(variety)
    #  So, the dim of our variety should be 9 - this is way too high, should be 1 (so J should be full rank)
    required_manifold_dim = len(vars) - rank_J_subbed

    #  Try Groebner basis analysis?
    # basis = sp.polys.groebner(constraints)

    #  TODO: can constraints that make the problem Archimedean be added? Hard because we don't have inequalities.
    #  Is the formulation already Archimmedean, though? Should check. Perhaps joint limits will be needed, but

    #  For Macaulay2
    # constraints_rational = [sp.nsimplify(cons) for cons in constraints]
    # constraints_m2 = [cons.args[0] - cons.args[1] for cons in constraints_rational]
    # m2_str = str(constraints_m2)
    # m2_str = m2_str.replace('**', '^')
    # m2_str = m2_str.replace('[', '').replace(']', '').replace('_', '')
    # print('R = QQ[' + str(list(vars)).replace('[', '').replace(']', '').replace('_', '') + ']')
    # print(f'I_dof{n} = ideal(' + m2_str + ')')
