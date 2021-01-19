import numpy as np
from numpy.linalg import norm
import pandas as pd
import cvxpy as cp
import mosek
import sympy as sp
import networkx as nx

from graphik.graphs.graph_base import RobotRevoluteGraph, RobotGraph
from graphik.solvers.sdp_formulations import SdpSolverParams
from graphik.solvers.solver_generic_sdp import SdpRelaxationSolver
from graphik.solvers.sdp_snl import solve_nearest_point_sdp
from graphik.solvers.constraints import get_full_revolute_nearest_points_pose, nearest_neighbour_cost, constraints_from_graph
from graphik.utils.utils import list_to_variable_dict, constraint_violations, safe_arccos
from graphik.utils.dgp import pos_from_graph, graph_from_pos

mosek_params = {
                mosek.iparam.intpnt_scaling: mosek.scalingtype.free,
                mosek.iparam.intpnt_solve_form: mosek.solveform.primal,
                mosek.iparam.ana_sol_print_violated: mosek.onoffkey.on,
                mosek.dparam.intpnt_co_tol_near_rel: 1e5,
}
sdp_solver_params = SdpSolverParams(
            solver=cp.MOSEK,
            abstol=1e-6,
            reltol=1e-6,
            feastol=1e-6,
            max_iters=1000,
            refinement_steps=1,
            kkt_solver="robust",
            alpha=1.8,
            scale=5.0,
            normalize=True,
            use_indirect=True,
            qcp=False,
            mosek_params=mosek_params,
            feasibility=False,
            cost_function=None,
)


def sdp_sol_to_point_matrix(graph, solution_dict, ee_goals):

    P_e = np.zeros((graph.robot.dim, len(graph.directed)))
    for idx, name in enumerate(graph.directed):
        if name == 'p0':
            P_e[:, idx] = np.zeros(3)
        elif name == 'x':
            P_e[:, idx] = np.array([1., 0., 0.])
        elif name == 'y':
            P_e[:, idx] = np.array([0., 1., 0.])
        elif name == 'q0':
            P_e[:, idx] = np.array([0., 0., 1.])
        elif name in ee_goals.keys():
            P_e[:, idx] = ee_goals[name]
        elif name in solution_dict.keys():
            P_e[:, idx] = solution_dict[name]
        else:
            P_e[0, idx] = solution_dict[sp.Symbol(name+'_x')]
            P_e[1, idx] = solution_dict[sp.Symbol(name + '_y')]
            P_e[2, idx] = solution_dict[sp.Symbol(name + '_z')]
    return P_e


def sdp_rank3_constraint_violations(constraint_clique_dict, graph, solution_dict, end_effectors, ee_cost=False):
    undirected = nx.Graph(graph.robot.structure_graph())
    residuals = []
    for clique in constraint_clique_dict:
        A_clique, b_clique, mapping, _ = constraint_clique_dict[clique]
        for u in clique:
            for v in clique:
                if frozenset((u, v)) in mapping:
                    u_val = end_effectors[u] if u in end_effectors and not ee_cost else solution_dict[u]
                    v_val = end_effectors[v] if v in end_effectors and not ee_cost else solution_dict[v]
                    residuals.append(np.linalg.norm(u_val - v_val)**2 - undirected[u][v]['weight']**2)

    return residuals


def run_sdp_rank3_revolute_experiment(graph: RobotGraph, init, q_goal: dict, use_limits: bool,
                                force_dense=False) -> pd.DataFrame:
    # Form the solver - TODO: check pose formulation
    G = graph.realization(q_goal)
    P = pos_from_graph(G)
    ee_goals = {}
    for pair in graph.robot.end_effectors:
        for ee in pair:
            if 'p' in ee:
                ee_goals[ee] = graph.robot.get_pose(q_goal, ee).trans
            else:
                for idx, key in enumerate(graph.node_ids):
                    if key == ee:
                        ee_goals[ee] = P[idx, :]

    q_nearest = list_to_variable_dict(init)
    nearest_points = get_full_revolute_nearest_points_pose(graph, q_nearest)
    solver_failures = 0
    solver_success = False

    while not solver_success:
        try:
            solution_dict, prob, constraint_clique_dict, sdp_variable_map = solve_nearest_point_sdp(nearest_points,
                                                    ee_goals, graph.robot, sparse=not force_dense, solver_params=None)
            solver_success = True
        except cp.error.SolverError as e:
            solver_failures += 1
            q_nearest = list_to_variable_dict(graph.robot.random_configuration())
            nearest_points = get_full_revolute_nearest_points_pose(graph, q_nearest)

    runtime = prob.solver_stats.solve_time
    num_iters = prob.solver_stats.num_iters
    Z = prob.variables()[0].value
    Z_eigvals = np.linalg.eigvalsh(Z)
    second_eigval_mag = np.abs(Z_eigvals[-2])
    eigval_ratio = np.abs(Z_eigvals[-1] / Z_eigvals[-2])
    # Compute relaxation gap
    sdp_cost = prob.value
    primal_cost = sum([np.linalg.norm(solution_dict[key] - nearest_points[key])**2 for key in nearest_points.keys()])
    gap = primal_cost - sdp_cost
    # Compute max constraint violation
    constraint_violations = sdp_rank3_constraint_violations(constraint_clique_dict, graph, solution_dict,
                                                            end_effectors=ee_goals)
    eq_resid_max = np.max(np.abs(constraint_violations))

    # Get points in form needed for initialization of Riemannian solver
    P_e = sdp_sol_to_point_matrix(graph, solution_dict, ee_goals)
    G_e = graph_from_pos(P_e.T, graph.node_ids)
    T_goal = {ee[0]: graph.robot.get_pose(q_goal, ee[0]) for ee in graph.robot.end_effectors}
    q_sdp = graph.robot.joint_variables(G_e, T_goal)

    err_pos = 0.
    err_rot = 0.
    pose_goals = {ee[0]: graph.robot.get_pose(q_goal, ee[0]) for ee in graph.robot.end_effectors}
    for key in pose_goals:
        T_sol = graph.robot.get_pose(list_to_variable_dict(q_sdp), key)
        err_pos += norm(pose_goals[key].trans - T_sol.trans)
        z1 = pose_goals[key].rot.as_matrix()[0:3, -1]
        z2 = T_sol.rot.as_matrix()[0:3, -1]
        err_rot += safe_arccos(z1.dot(z2))

    limit_violations = list_to_variable_dict(graph.robot.n * [0])
    limits_violated = False
    if use_limits:
        for key in graph.robot.limited_joints:
            limit_violations[key] = max(
                graph.robot.lb[key] - q_sdp[key], q_sdp[key] - graph.robot.ub[key]
            )
            if limit_violations[key] > 0.01 * graph.robot.ub[key]:
                limits_violated = True

    data = dict(
        [
            ("Init.", [init]),
            ("Goals", [ee_goals]),
            ("Iterations", [num_iters]),
            ("Runtime", [runtime]),
            ("Solution Config", [q_sdp]),
            ("Pos Error", [err_pos]),
            ("Rot Error", [err_rot]),
            ("Limit Violations", [limit_violations]),
            ("Limits Violated", [limits_violated]),
            ("Solver Failures", [solver_failures]),
            ("Second Eigenvalue Magnitude", [second_eigval_mag]),
            ("Eigenvalue Ratio", [eigval_ratio]),
            ("SDP Cost", [sdp_cost]),
            ("Relaxation Gap", [gap]),
            ("Max Constraint Violation", [eq_resid_max])
        ])
    results = pd.DataFrame(data)
    return results, q_sdp, P_e.T


def run_sdp_revolute_experiment(graph: RobotGraph, init, q_goal: dict, use_limits: bool,
                                force_dense=False) -> pd.DataFrame:

    # Form the solver - TODO: check pose formulation
    G = graph.realization(q_goal)
    P = pos_from_graph(G)
    ee_goals = {}
    for pair in graph.robot.end_effectors:
        for ee in pair:
            if 'p' in ee:
                ee_goals[ee] = graph.robot.get_pose(q_goal, ee).trans
            else:
                for idx, key in enumerate(graph.node_ids):
                    if key == ee:
                        ee_goals[ee] = P[idx, :]

    prob_params = {"end_effector_assignment": ee_goals}
    solver = SdpRelaxationSolver(
        params=sdp_solver_params, verbose=False, force_dense=force_dense
    )
    q_nearest = list_to_variable_dict(init)
    nearest_points = get_full_revolute_nearest_points_pose(graph, q_nearest)
    solver.cost = nearest_neighbour_cost(graph, nearest_points)
    solver_failures = 0
    solver_success = False
    while not solver_success:
        try:
            solution_dict, ranks, prob, constraints = solver.solve(graph, prob_params)
            solver_success = True
        except cp.error.SolverError as e:
            solver_failures += 1
            q_nearest = list_to_variable_dict(graph.robot.random_configuration())
            nearest_points = get_full_revolute_nearest_points_pose(graph, q_nearest)
            solver.cost = nearest_neighbour_cost(graph, nearest_points)

    runtime = prob.solver_stats.solve_time
    num_iters = prob.solver_stats.num_iters
    Z = prob.variables()[0].value
    Z_eigvals = np.linalg.eigvalsh(Z)
    second_eigval_mag = np.abs(Z_eigvals[-2])
    eigval_ratio = np.abs(Z_eigvals[-1] / Z_eigvals[-2])
    # Compute relaxation gap
    sdp_cost = prob.value
    primal_cost = solver.cost.subs(solution_dict)
    gap = primal_cost - sdp_cost
    # Compute max constraint violation
    violations = constraint_violations(constraints, solution_dict)
    # robot_graph.plot_solution(solution_dict)
    eq_resid = [resid for (resid, is_eq) in violations if is_eq]
    # TODO: sort out how to incorporate inequalites (needed for angular limits)
    # ineq_resid = [resid for (resid, is_eq) in violations if not is_eq]
    eq_resid_max = np.max(np.abs(eq_resid))

    # Get points in form needed for initialization of Riemannian solver
    P_e = sdp_sol_to_point_matrix(graph, solution_dict, ee_goals)
    G_e = graph_from_pos(P_e.T, graph.node_ids)
    T_goal = {ee[0]: graph.robot.get_pose(q_goal, ee[0]) for ee in graph.robot.end_effectors}
    q_sdp = graph.robot.joint_variables(G_e, T_goal)

    err_pos = 0.
    err_rot = 0.
    pose_goals = {ee[0]: graph.robot.get_pose(q_goal, ee[0]) for ee in graph.robot.end_effectors}
    for key in pose_goals:
        T_sol = graph.robot.get_pose(list_to_variable_dict(q_sdp), key)
        err_pos += norm(pose_goals[key].trans - T_sol.trans)
        z1 = pose_goals[key].rot.as_matrix()[0:3, -1]
        z2 = T_sol.rot.as_matrix()[0:3, -1]
        err_rot += safe_arccos(z1.dot(z2))

    limit_violations = list_to_variable_dict(graph.robot.n * [0])
    limits_violated = False
    if use_limits:
        for key in graph.robot.limited_joints:
            limit_violations[key] = max(
                graph.robot.lb[key] - q_sdp[key], q_sdp[key] - graph.robot.ub[key]
            )
            if limit_violations[key] > 0.01 * graph.robot.ub[key]:
                limits_violated = True

    data = dict(
            [
                ("Init.", [init]),
                ("Goals", [ee_goals]),
                ("Iterations", [num_iters]),
                ("Runtime", [runtime]),
                ("Solution Config", [q_sdp]),
                ("Pos Error", [err_pos]),
                ("Rot Error", [err_rot]),
                ("Limit Violations", [limit_violations]),
                ("Limits Violated", [limits_violated]),
                ("Solver Failures", [solver_failures]),
                ("Second Eigenvalue Magnitude", [second_eigval_mag]),
                ("Eigenvalue Ratio", [eigval_ratio]),
                ("SDP Cost", [sdp_cost]),
                ("Relaxation Gap", [gap]),
                ("Max Constraint Violation", [eq_resid_max])
            ])
    results = pd.DataFrame(data)
    return results, q_sdp, P_e.T


if __name__ == '__main__':
    pass
