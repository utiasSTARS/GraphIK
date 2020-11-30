import numpy as np
import itertools
import time
import sys
import pandas as pd
from numpy.linalg import norm
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib import rc

rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 18})
rc("text", usetex=True)
from progress.bar import ShadyBar as Bar

from graphik.solvers.local_solver import LocalSolver
from graphik.solvers.solver_fabrik import solver_fabrik
from graphik.solvers.geometric_jacobian import jacobian_ik
from graphik.solvers.riemannian_solver import RiemannianSolver
from graphik.graphs.graph_base import Graph
from graphik.robots.robot_base import RobotRevolute, RobotSpherical, RobotPlanar
from graphik.utils.utils import (
    list_to_variable_dict,
    list_to_variable_dict_spherical,
    variable_dict_to_list,
    best_fit_transform,
    bounds_to_spherical_bounds,
    trans_axis,
    safe_arccos,
    wraptopi,
    bernoulli_confidence_jeffreys,
    # bernoulli_confidence_normal_approximation
)


# Colours and styles used throughout
linestyles = ["-", "--", "-."]
line_colors = ["#000000", "#990000", "#294772", "#F6AA1C"]  # , '#cccccc']
line_markers = ["s", "o", "d", "*"]


def run_multiple_experiments(
    graph,
    n_goals: int,
    n_init: int,
    zero_init: bool,
    local_solver_params: dict,
    riemann_params,
    jacobian_params,
    use_limits,
    verbosity,
    bound_smoothing,
    local_algorithms,
    riemannian_algorithms,
    fabrik_max_iter,
    trigsimp=False,
    do_fabrik=True,
    do_jacobian=True,
    fabrik_tol=1e-6,
    use_symbolic=True,
    use_hess=True,
    local_graph=None,
    local_graph_map=None,
    pose_goals=False,
):
    results_list = []

    bar = Bar("", max=n_goals, check_tty=False, hide_cursor=False)
    for idx in range(n_goals):
        q_goal = graph.robot.random_configuration()
        if zero_init:
            if graph.robot.spherical:
                init = n_init * [graph.robot.n * 2 * [0.0]]
            else:
                init = n_init * [graph.robot.n * [0.0]]
        else:
            init = [graph.robot.random_configuration() for _ in range(n_init)]
            init = variable_dict_to_list(init)
        res = run_full_experiment(
            graph,
            local_solver_params,
            riemann_params,
            jacobian_params,
            q_goal,
            init,
            use_limits,
            verbosity,
            bound_smoothing,
            local_algorithms,
            riemannian_algorithms,
            trigsimp,
            fabrik_max_iter,
            fabrik_tol=fabrik_tol,
            do_fabrik=do_fabrik,
            do_jacobian=do_jacobian,
            use_symbolic=use_symbolic,
            use_hess=use_hess,
            local_graph=local_graph,
            local_graph_map=local_graph_map,
            pose_goals=pose_goals,
        )
        results_list.append(res)
        bar.next()
    bar.finish()
    return pd.concat(results_list, sort=True)


def run_full_experiment(
    graph: Graph,
    solver_params: dict,
    riemannian_params: dict,
    jacobian_params: dict,
    q_goal: dict,
    init: list,
    use_limits: bool = False,
    verbosity: int = 2,
    bound_smoothing: bool = False,
    local_algorithms: list = None,
    riemannian_algorithms: list = None,
    trigsimp: bool = False,
    fabrik_max_iter: int = 200,
    fabrik_tol: float = 1e-6,
    do_fabrik: bool = True,
    do_jacobian: bool = True,
    use_symbolic: bool = True,
    use_hess: bool = True,
    local_graph=None,
    local_graph_map=None,
    pose_goals=False,
) -> pd.DataFrame:
    """
    Run an experiment with a variety of solvers for a single goal specified by ee_goals.

    :param graph: instance of Graph describing our robot
    :param solver_params: dictionary with local solver parameters
    :param riemannian_params: dictionary with Riemannian solver parameters
    :param jacobian_params: dictionary with Jacobian solver parameters
    :param q_goal: angular configuration specifying end-effector goals in dictionary form
    :param init: a list of angular configurations to try as initial points
    :param use_limits: boolean indicating whether to use the angular limits in graph.robot
    :param verbosity: integer representing solver verbosity (0 to 2)
    :param bound_smoothing: boolean indicating whether to initialize the Riemannian solver with bound smoothing (only
                            makes sense for a comparison with a single init)
    :param local_algorithms: list of local algorithms to use with local_solver (e.g., "L-BFGS-B", "TNC")
    :param riemannian_algorithms: list of algorithms to use with the riemannian solver
                                  (e.g., "TrustRegions", "ConjugateGradient")
    :param trigsimp: boolean indicating whether to use sympy.trigsimp on the local solver's cost function.
    :param fabrik_max_iter: number of iterations for the FABRIK solver
    :param fabrik_tol: tolerance for fabrik
    :param do_fabrik: whether FABRIK should be run
    :param do_jacobian: whether the Jacobian-based method should be run
    :return: pd.DataFrame with all results
    """
    if local_algorithms is None:
        if use_limits:
            local_algorithms = ["L-BFGS-B", "TNC", "SLSQP", "trust-constr"]
        else:
            local_algorithms = ["BFGS", "CG", "Newton-CG", "trust-exact"]

    if riemannian_algorithms is None:
        riemannian_algorithms = ["TrustRegions", "ConjugateGradient"]
    results_list = []  # Stores all data frames to be eventually merged

    # is_revolute3d = type(graph.robot) in (Revolute3dChain, Revolute3dTree)
    is_revolute3d = type(graph.robot) is RobotRevolute
    is_spherical = type(graph.robot) is RobotSpherical
    is_planar = not (is_revolute3d or is_spherical)

    # Set end effector goals from q_goal
    ee_goals = {}
    for ee in graph.robot.end_effectors:
        if is_revolute3d:
            ee_p = ee[0] if "p" in ee[0] else ee[1]
            ee_goals[ee_p] = graph.robot.get_pose(q_goal, ee_p)
        else:
            ee_goals[ee[0]] = graph.robot.get_pose(q_goal, ee[0])

    # Used by FABRIK
    ee_goals_points = {}
    for ee in graph.robot.end_effectors:
        ee_goals_points[ee[0]] = graph.robot.get_pose(q_goal, ee[0]).trans
        if pose_goals and not is_revolute3d:
            ee_goals_points[ee[1]] = graph.robot.get_pose(q_goal, ee[1]).trans
        elif pose_goals and is_revolute3d:
            ee_goals_points[ee[1]] = (
                graph.robot.get_pose(q_goal, ee[0])
                .dot(trans_axis(graph.robot.axis_length, "z"))
                .trans
            )
    # Deal with the local graph object (for spherical case)
    if local_graph is None:
        local_graph = graph
        if pose_goals and is_planar:
            ee_goals_local = {
                key: graph.robot.get_pose(q_goal, key) for key in ee_goals_points
            }
        else:
            ee_goals_local = ee_goals
        ee_goals_local_eval = ee_goals
        spherical_to_revolute_case = False
    else:
        if pose_goals:
            ee_goals_local = {
                local_graph_map[key]: graph.robot.get_pose(q_goal, key)
                for key in ee_goals_points
            }
            ee_goals_local_eval = {
                local_graph_map[ee[0]]: graph.robot.get_pose(q_goal, ee[0])
                for ee in graph.robot.end_effectors
            }
        spherical_to_revolute_case = True

    if len(local_algorithms) != 0:
        use_q_in_cost = pose_goals and not spherical_to_revolute_case
        local_solver = LocalSolver(solver_params)
        if is_revolute3d or spherical_to_revolute_case:
            local_solver.set_revolute_cost_function(
                local_graph.robot,
                ee_goals_local,
                local_graph.robot.lb.keys(),
                pose_cost=use_q_in_cost,
            )
        elif use_symbolic:
            local_solver.set_symbolic_cost_function(
                local_graph.robot,
                ee_goals_local,
                local_graph.robot.lb.keys(),
                use_trigsimp=trigsimp,
                use_hess=use_hess,
                pose_cost=use_q_in_cost,
            )
        elif is_planar:
            local_solver.set_procedural_cost_function(
                local_graph.robot,
                ee_goals_local,
                pose_cost=False,
                do_grad_and_hess=True,
            )
        else:
            local_solver.set_procedural_cost_function(
                local_graph.robot, ee_goals_local, pose_cost=use_q_in_cost
            )

    for algorithm in local_algorithms:
        if is_revolute3d or spherical_to_revolute_case:
            solve_fn = run_local_revolute_experiment
        elif is_planar:
            solve_fn = run_local_planar_experiment
        local_solver.params["solver"] = algorithm
        res_df = solve_fn(
            local_graph,
            ee_goals_local_eval,
            local_solver,
            -1,
            init[0],
            use_limits=use_limits,
            use_hess=use_hess,
            pose_goals=use_q_in_cost,
        )
        res_df["Solver"] = algorithm
        results_list.append(res_df)

    # Set up Riemannian solver sweep inputs
    if len(riemannian_algorithms) != 0:
        G_goal = graph.realization(q_goal)
        X_goal = graph.pos_from_graph(G_goal)
        D_goal = graph.distance_matrix(q_goal)
        if pose_goals:
            T_goal = ee_goals
        else:
            T_goal = graph.robot.get_pose(q_goal, f"p{graph.robot.n}")

    for algorithm in riemannian_algorithms:
        if is_revolute3d:
            solve_fn = run_riemannian_revolute_experiment
        elif is_spherical:
            solve_fn = run_riemannian_spherical_experiment
        elif is_planar:
            solve_fn = run_riemannian_planar_experiment
        # print("Running Riemannian {:} solver...".format(algorithm))
        riemannian_params["solver"] = algorithm
        riemannian_solver = RiemannianSolver(graph, riemannian_params)

        res_df = solve_fn(
            graph,
            riemannian_solver,
            -1,
            D_goal,
            X_goal,
            ee_goals,
            init[0],
            T_goal=T_goal,
            use_limits=use_limits,
            verbosity=verbosity,
            bound_smoothing=False,
        )
        res_df["Solver"] = "Riemannian " + algorithm
        results_list.append(res_df)

        if bound_smoothing:
            # print("Running Riemannian {:} solver with BS...".format(algorithm))
            riemannian_params["solver"] = algorithm
            riemannian_solver = RiemannianSolver(graph, riemannian_params)

            res_df_bs = solve_fn(
                graph,
                riemannian_solver,
                -1,
                D_goal,
                X_goal,
                ee_goals,
                init[0],
                T_goal=T_goal,
                use_limits=use_limits,
                verbosity=verbosity,
                bound_smoothing=True,
                pose_goals=use_q_in_cost,
            )
            res_df_bs["Solver"] = "Riemannian " + algorithm + " + BS"
            results_list.append(res_df_bs)

    # Run FABRIK solver
    if do_fabrik:
        goals_fabrik, goals_index_fabrik = retrieve_goals(ee_goals_points)
        res_fabrik = run_full_fabrik_sweep_experiment(
            graph,
            goals_fabrik,
            goals_index_fabrik,
            ee_goals=ee_goals,
            initial=init,
            use_limits=use_limits,
            max_iteration=fabrik_max_iter,
            verbosity=0,
            tol=fabrik_tol,
        )
        res_fabrik["Solver"] = "FABRIK"
        results_list.append(res_fabrik)

    # Jacobian-based solver (dls-inverse)
    if do_jacobian:
        if is_revolute3d:
            solve_fn = run_jacobian_revolute_experiment
        elif is_spherical:
            solve_fn = run_jacobian_spherical_experiment
        elif is_planar:
            solve_fn = run_jacobian_planar_experiment
        local_solver_jac = LocalSolverJac(graph.robot)
        res_jacobian = solve_fn(
            graph, local_solver_jac, init[0], q_goal, use_limits=use_limits
        )
        # if not spherical_to_revolute_case:  # Use the revolute formulation if a revolute equivalent is provided
        #     res_jacobian = run_full_jacobian_sweep_experiment(
        #         graph, ee_goals, init, q_goal, params=jacobian_params, use_limits=use_limits
        #     )
        # else:
        #     q_goal_revolute = list_to_variable_dict(flatten(q_goal.values()))
        #     res_jacobian = run_full_jacobian_sweep_experiment(
        #         local_graph, ee_goals_local, init, q_goal_revolute, params=jacobian_params, use_limits=use_limits
        #     )
        res_jacobian["Solver"] = "Jacobian"
        results_list.append(res_jacobian)

    # Joint it all together
    results = pd.concat(results_list, sort=True)
    results["Goal"] = str(q_goal)  # TODO: is there a better way to store the goal?
    return results


def process_experiment(data: pd.DataFrame, pos_threshold=0.01, rot_threshold=0.01):
    # Summarize angular constraint violation and squared end-effector error
    for algorithm in data["Solver"].unique():
        data_alg = data[data["Solver"] == algorithm]

        successes = (
            (data_alg["Pos Error"] < pos_threshold)
            & (data_alg["Rot Error"] < rot_threshold)
            & (data_alg["Limits Violated"] == False)
        )

        print("Solver: {:}".format(algorithm))
        print(data_alg[successes]["Pos Error"].describe())
        print(data_alg[successes]["Rot Error"].describe())
        print("Success rate over {:} runs: ".format(data_alg["Pos Error"].count()))
        print(
            100
            * data_alg[successes]["Pos Error"].count()
            / data_alg["Pos Error"].count()
        )
        print(data_alg[successes]["Runtime"].mean())


def run_riemannian_revolute_experiment(
    graph: Graph,
    solver: RiemannianSolver,
    n_per_dim: int,
    D_goal,
    Y_goal,
    ee_goals: dict,
    Y_init: dict,
    T_goal=None,
    use_limits: bool = False,
    verbosity=2,
    bound_smoothing: bool = False,
    pose_goals: bool = False,
) -> pd.DataFrame:
    """
    :param graph:
    :param solver:
    :param n_per_dim:
    :param D_goal:
    :param X_goal:
    :param ee_goals:
    :param T_goal:
    :param init:
    :param use_limits:
    :param verbosity:
    :param bound_smoothing:
    :return:
    """

    # axis length determines q node distances
    axis_len = graph.robot.axis_length

    # Determine align indices
    align_ind = list(np.arange(graph.dim + 1))
    for name in ee_goals.keys():
        align_ind.append(graph.node_ids.index(name))

    # Set known positions
    goals = {}
    for key in ee_goals:
        goals[key] = ee_goals[key].trans
        goals["q" + key[1:]] = ee_goals[key].dot(trans_axis(axis_len, "z")).trans
    G = graph.complete_from_pos(goals)

    # Adjacency matrix
    omega = graph.adjacency_matrix(G)

    init_angles = list_to_variable_dict(Y_init)
    G_init = graph.realization(init_angles)
    Y_init = graph.pos_from_graph(G_init)

    # Set bounds if using bound smoothing
    bounds = None
    if bound_smoothing:
        lb, ub = graph.distance_bounds(G)  # will take goals and jli
        bounds = (lb, ub)

    # Solve problem
    Y_opt, optlog = solver.solve_experiment_wrapper(
        D_goal,
        omega,
        bounds=bounds,
        X=Y_init,
        use_limits=use_limits,
        verbosity=verbosity,
    )

    f_x = optlog["final_values"]["f(x)"]
    grad_norm = optlog["final_values"]["gradnorm"]
    runtime = optlog["final_values"]["time"]
    num_iters = optlog["final_values"]["iterations"]

    # Check for linear/planar solutions in the 3D case, pad with zeros to fix
    if Y_opt.shape[1] < graph.dim:
        Y_opt = np.hstack(
            [Y_opt, np.zeros((Y_opt.shape[0], graph.dim - Y_opt.shape[1]))]
        )

    # Get solution config
    R, t = best_fit_transform(Y_opt[align_ind, :], Y_goal[align_ind, :])
    P_e = (R @ Y_opt.T + t.reshape(graph.dim, 1)).T
    G_e = graph.graph_from_pos(P_e)
    q_sol = graph.robot.joint_angles_from_graph(G_e, T_goal)

    # If limits are used check for angle violations
    limit_violations = list_to_variable_dict(graph.robot.n * [0])
    limits_violated = False
    if use_limits:
        for key in graph.robot.limited_joints:
            limit_violations[key] = max(
                graph.robot.lb[key] - q_sol[key], q_sol[key] - graph.robot.ub[key]
            )

            if limit_violations[key] > 0.01 * graph.robot.ub[key]:
                limits_violated = True

    # if limits_violated:
    #     print("--------------------------")
    #     print("Method: Riemannian")
    #     print(
    #         "Angle violated! \n Lower bounds: {:} \n Upper bounds: {:}".format(
    #             graph.robot.lb, graph.robot.ub
    #         )
    #     )
    #     print("q_sol: {:}".format(q_sol))
    #     print("--------------------------")

    # Calculate final error
    D_sol = graph.distance_matrix(q_sol)
    e_D = omega * (np.sqrt(D_sol) - np.sqrt(D_goal))
    max_dist_error = abs(max(e_D.min(), e_D.max(), key=abs))
    err_pos = 0.0
    err_rot = 0.0
    for key in ee_goals:
        T_sol = graph.robot.get_pose(list_to_variable_dict(q_sol), key)
        T_sol.rot.as_matrix()[0:3, 0:2] = ee_goals[key].rot.as_matrix()[0:3, 0:2]
        err_pos += norm(ee_goals[key].trans - T_sol.trans)
        # err_rot += norm((ee_goals[key].rot.dot(T_sol.rot.inv())).log())
        z1 = ee_goals[key].rot.as_matrix()[0:3, -1]
        z2 = T_sol.rot.as_matrix()[0:3, -1]
        err_rot += safe_arccos(z1.dot(z2))

    columns = [
        "Init.",
        "Goals",
        "f(x)",
        "Gradient Norm",
        "Iterations",
        "Runtime",
        "Solution",
        "Solution Config",
        "Pos Error",
        "Rot Error",
        "Limit Violations",
        "Limits Violated",
        "Max Dist Error",
    ]

    data = dict(
        zip(
            columns,
            [
                [Y_init],
                [ee_goals],
                [f_x],
                [grad_norm],
                [num_iters],
                [runtime],
                [Y_opt],
                [q_sol],
                [err_pos],
                [err_rot],
                [limit_violations],
                [limits_violated],
                [max_dist_error],
            ],
        )
    )

    results = pd.DataFrame(data)
    results["Bound Smoothing"] = bound_smoothing
    return results


def run_local_revolute_experiment(
    graph: Graph,
    ee_goals: dict,
    solver: LocalSolver,
    n_per_dim: int,
    init: list,
    use_limits=False,
    use_hess=True,
    pose_goals=False,
) -> pd.DataFrame:
    """

    :param graph:
    :param solver: LocalSolver object with cost-function pre-set
    :param n_per_dim:
    :param init: list of specific angle combinations to try
    :param angle_tol: tolerance on angle
    :return:
    """

    problem_params = {}
    if use_limits:
        problem_params["angular_limits"] = graph.robot.ub  # Assumes symmetrical limits

    problem_params["initial_guess"] = list_to_variable_dict(init)
    results = solver.solve(graph, problem_params)
    solutions = results.x

    # Wrap to within [-pi, pi]
    solutions = [wraptopi(val) for val in solutions]
    q_sol = list_to_variable_dict(solutions)

    # If limits are used check for angle violations
    limit_violations = list_to_variable_dict(graph.robot.n * [0])
    limits_violated = False
    if use_limits:
        for key in graph.robot.limited_joints:
            limit_violations[key] = max(
                graph.robot.lb[key] - q_sol[key], q_sol[key] - graph.robot.ub[key]
            )

            if limit_violations[key] > 0.01 * graph.robot.ub[key]:
                limits_violated = True

    # if limits_violated:
    #     print("--------------------------")
    #     print("Method: Local")
    #     print(
    #         "Angle violated! \n Lower bounds: {:} \n Upper bounds: {:}".format(
    #             graph.robot.lb, graph.robot.ub
    #         )
    #     )
    #     print("q_sol: {:}".format(q_sol))
    #     print("--------------------------")

    err_pos = 0.0
    err_rot = 0.0
    for key in ee_goals:
        T_sol = graph.robot.get_pose(list_to_variable_dict(q_sol), key)
        err_pos += norm(ee_goals[key].trans - T_sol.trans)
        z1 = ee_goals[key].rot.as_matrix()[0:3, -1]
        z2 = T_sol.rot.as_matrix()[0:3, -1]
        err_rot += safe_arccos(z1.dot(z2))

        # T_sol.rot.as_matrix()[0:3,0:2] = ee_goals[key].rot.as_matrix()[0:3,0:2]
        # err_rot += norm((ee_goals[key].rot.dot(T_sol.rot.inv())).log())

    # print(ee_goals)
    # print(T_sol)
    # print("\n")

    grad_norm = norm(solver.grad(results.x))
    if use_hess:
        smallest_eig = min(np.linalg.eigvalsh(solver.hess(results.x).astype(float)))
    runtime = results.runtime
    num_iters = results.nit

    columns = [
        "Init.",
        "Goals",
        "Iterations",
        "Runtime",
        "Solution Config",
        "Pos Error",
        "Rot Error",
        "Limit Violations",
        "Limits Violated",
    ]

    data = dict(
        zip(
            columns,
            [
                [init],
                [ee_goals],
                [num_iters],
                [runtime],
                [q_sol],
                [err_pos],
                [err_rot],
                [limit_violations],
                [limits_violated],
            ],
        )
    )

    return pd.DataFrame(data)


def run_riemannian_planar_experiment(
    graph: Graph,
    solver: RiemannianSolver,
    n_per_dim: int,
    D_goal,
    Y_goal,
    ee_goals: dict,
    q_init: dict,
    T_goal=None,
    use_limits: bool = False,
    verbosity=2,
    bound_smoothing: bool = False,
    pose_goals: bool = False,
) -> pd.DataFrame:
    """
    :param graph:
    :param solver:
    :param n_per_dim:
    :param D_goal:
    :param X_goal:
    :param ee_goals:
    :param T_goal:
    :param init:
    :param use_limits:
    :param verbosity:
    :param bound_smoothing:
    :return:
    """

    # Determine align indices
    align_ind = list(np.arange(graph.dim + 1))
    for name in ee_goals.keys():
        align_ind.append(graph.node_ids.index(name))

    # Set known positions
    # G = graph.complete_from_pos(ee_goals)
    G = graph.complete_from_pose_goal(ee_goals)

    # Adjacency matrix
    omega = graph.adjacency_matrix(G)

    q_init = list_to_variable_dict(q_init)
    G_init = graph.realization(q_init)
    Y_init = graph.pos_from_graph(G_init)

    # Set bounds if using bound smoothing
    bounds = None
    if bound_smoothing:
        lb, ub = graph.distance_bounds(G)  # will take goals and jli
        bounds = (lb, ub)

    # Solve problem
    Y_opt, optlog = solver.solve_experiment_wrapper(
        D_goal,
        omega,
        bounds=bounds,
        X=Y_init,
        use_limits=use_limits,
        verbosity=verbosity,
    )

    f_x = optlog["final_values"]["f(x)"]
    grad_norm = optlog["final_values"]["gradnorm"]
    runtime = optlog["final_values"]["time"]
    num_iters = optlog["final_values"]["iterations"]

    # Check for linear/planar solutions in the 3D case, pad with zeros to fix
    if Y_opt.shape[1] < graph.dim:
        Y_opt = np.hstack(
            [Y_opt, np.zeros((Y_opt.shape[0], graph.dim - Y_opt.shape[1]))]
        )

    # Get solution config
    R, t = best_fit_transform(Y_opt[align_ind, :], Y_goal[align_ind, :])
    P_e = (R @ Y_opt.T + t.reshape(graph.dim, 1)).T
    G_e = graph.graph_from_pos(P_e)
    q_sol = graph.robot.joint_variables(G_e)

    # If limits are used check for angle violations
    limit_violations = list_to_variable_dict(graph.robot.n * [0])
    limits_violated = False
    if use_limits:
        for key in q_sol:
            limit_violations[key] = max(
                graph.robot.lb[key] - q_sol[key], q_sol[key] - graph.robot.ub[key]
            )

            if limit_violations[key] > 0.01 * graph.robot.ub[key]:
                limits_violated = True

    # Calculate final error
    D_sol = graph.distance_matrix(q_sol)
    e_D = omega * (np.sqrt(D_sol) - np.sqrt(D_goal))
    max_dist_error = abs(max(e_D.min(), e_D.max(), key=abs))
    err_pos = 0
    err_rot = 0
    for key in ee_goals:
        T_sol = graph.robot.get_pose(list_to_variable_dict(q_sol), key)
        err_pos += norm(ee_goals[key].trans - T_sol.trans)
        err_rot += safe_arccos(
            (ee_goals[key].rot.dot(T_sol.rot.inv())).as_matrix()[0, 0]
        )

    data = dict(
        [
            ("Init.", [Y_init]),
            ("Goals", [ee_goals]),
            ("f(x)", [f_x]),
            ("Gradient Norm", [grad_norm]),
            ("Iterations", [num_iters]),
            ("Runtime", [runtime]),
            ("Solution", [Y_opt]),
            ("Solution Config", [q_sol]),
            ("Pos Error", [err_pos]),
            ("Rot Error", [err_rot]),
            ("Limit Violations", [limit_violations]),
            ("Limits Violated", [limits_violated]),
            ("Max Dist Error", [max_dist_error]),
        ]
    )

    results = pd.DataFrame(data)
    results["Bound Smoothing"] = bound_smoothing
    return results


def run_riemannian_spherical_experiment(
    graph: Graph,
    solver: RiemannianSolver,
    n_per_dim: int,
    D_goal,
    Y_goal,
    ee_goals: dict,
    q_init: dict,
    T_goal=None,
    use_limits: bool = False,
    verbosity=2,
    bound_smoothing: bool = False,
    pose_goals: bool = False,
) -> pd.DataFrame:
    """
    :param graph:
    :param solver:
    :param n_per_dim:
    :param D_goal:
    :param X_goal:
    :param ee_goals:
    :param T_goal:
    :param init:
    :param use_limits:
    :param verbosity:
    :param bound_smoothing:
    :return:
    """

    # Determine align indices
    align_ind = list(np.arange(graph.dim + 1))
    for name in ee_goals.keys():
        align_ind.append(graph.node_ids.index(name))

    # Set known positions
    # G = graph.complete_from_pos(ee_goals)
    G = graph.complete_from_pose_goal(ee_goals)

    # Adjacency matrix
    omega = graph.adjacency_matrix(G)

    q_init = list_to_variable_dict_spherical(q_init, in_pairs=True)
    G_init = graph.realization(q_init)
    Y_init = graph.pos_from_graph(G_init)

    # Set bounds if using bound smoothing
    bounds = None
    if bound_smoothing:
        lb, ub = graph.distance_bounds(G)  # will take goals and jli
        bounds = (lb, ub)

    # Solve problem
    Y_opt, optlog = solver.solve_experiment_wrapper(
        D_goal,
        omega,
        bounds=bounds,
        X=Y_init,
        use_limits=use_limits,
        verbosity=verbosity,
    )

    f_x = optlog["final_values"]["f(x)"]
    grad_norm = optlog["final_values"]["gradnorm"]
    runtime = optlog["final_values"]["time"]
    num_iters = optlog["final_values"]["iterations"]

    # Check for linear/planar solutions in the 3D case, pad with zeros to fix
    if Y_opt.shape[1] < graph.dim:
        Y_opt = np.hstack(
            [Y_opt, np.zeros((Y_opt.shape[0], graph.dim - Y_opt.shape[1]))]
        )

    # Get solution config
    R, t = best_fit_transform(Y_opt[align_ind, :], Y_goal[align_ind, :])
    P_e = (R @ Y_opt.T + t.reshape(graph.dim, 1)).T
    G_e = graph.graph_from_pos(P_e)
    q_sol = graph.robot.joint_variables(G_e)

    # If limits are used check for angle violations
    limit_violations = list_to_variable_dict(graph.robot.n * [0])
    limits_violated = False
    if use_limits:
        for key in q_sol:
            limit_violations[key] = max(
                graph.robot.lb[key] - q_sol[key][1], q_sol[key][1] - graph.robot.ub[key]
            )
            if limit_violations[key] > 0.01 * graph.robot.ub[key]:
                limits_violated = True

    # Calculate final error
    D_sol = graph.distance_matrix(q_sol)
    e_D = omega * (np.sqrt(D_sol) - np.sqrt(D_goal))
    max_dist_error = abs(max(e_D.min(), e_D.max(), key=abs))
    err_pos = 0
    err_rot = 0
    for key in ee_goals:
        T_sol = graph.robot.get_pose(q_sol, key)
        err_pos += norm(ee_goals[key].trans - T_sol.trans)
        z1 = ee_goals[key].rot.as_matrix()[0:3, -1]
        z2 = graph.robot.get_pose(q_sol, key).rot.as_matrix()[0:3, -1]
        err_rot += safe_arccos(z1.dot(z2))

    data = dict(
        [
            ("Init.", [Y_init]),
            ("Goals", [ee_goals]),
            ("f(x)", [f_x]),
            ("Gradient Norm", [grad_norm]),
            ("Iterations", [num_iters]),
            ("Runtime", [runtime]),
            ("Solution", [Y_opt]),
            ("Solution Config", [q_sol]),
            ("Pos Error", [err_pos]),
            ("Rot Error", [err_rot]),
            ("Limit Violations", [limit_violations]),
            ("Limits Violated", [limits_violated]),
            ("Max Dist Error", [max_dist_error]),
        ]
    )

    results = pd.DataFrame(data)
    results["Bound Smoothing"] = bound_smoothing
    return results


def run_local_planar_experiment(
    graph: Graph,
    ee_goals: dict,
    solver: LocalSolver,
    n_per_dim: int,
    init: list,
    use_limits=False,
    use_hess=False,
    pose_goals=True,
) -> pd.DataFrame:
    """

    :param graph:
    :param solver: LocalSolver object with cost-function pre-set
    :param n_per_dim:
    :param init: list of specific angle combinations to try
    :param angle_tol: tolerance on angle
    :return:
    """

    problem_params = {}
    if use_limits:
        problem_params["angular_limits"] = graph.robot.ub  # Assumes symmetrical limits

    problem_params["initial_guess"] = list_to_variable_dict(init)
    results = solver.solve(graph, problem_params)
    q_sol = list_to_variable_dict(results.x)

    # If limits are used check for angle violations
    limit_violations = list_to_variable_dict(graph.robot.n * [0])
    limits_violated = False
    if use_limits:
        for key in q_sol:
            limit_violations[key] = max(
                graph.robot.lb[key] - q_sol[key], q_sol[key] - graph.robot.ub[key]
            )
            if limit_violations[key] > 0.01 * graph.robot.ub[key]:
                limits_violated = True

    err_pos = 0.0
    err_rot = 0.0
    for key in ee_goals:
        T_sol = graph.robot.get_pose(q_sol, key)
        err_pos += norm(ee_goals[key].trans - T_sol.trans)
        # Do arccos for an angle instead
        err_rot += safe_arccos(
            (ee_goals[key].rot.dot(T_sol.rot.inv())).as_matrix()[0, 0]
        )

    runtime = results.runtime
    num_iters = results.nit

    data = dict(
        [
            ("Init.", [init]),
            ("Goals", [ee_goals]),
            ("Iterations", [num_iters]),
            ("Runtime", [runtime]),
            ("Solution Config", [q_sol]),
            ("Pos Error", [err_pos]),
            ("Rot Error", [err_rot]),
            ("Limit Violations", [limit_violations]),
            ("Limits Violated", [limits_violated]),
        ]
    )

    return pd.DataFrame(data)


def retrieve_goals(end_effector_assignment):
    end_effectors = list(end_effector_assignment.keys())
    goals = []
    goal_index = []
    for i in range(len(end_effectors)):
        goals += [list(end_effector_assignment[end_effectors[i]])]
        goal_index += [int(end_effectors[i][1:])]

    return goals, goal_index


def run_full_fabrik_sweep_experiment(
    graph,
    goals: list,
    goal_index: list,
    ee_goals: dict,
    initial: list = None,
    use_limits: bool = True,
    max_iteration=65,
    tol=0.01,
    verbosity=0,
) -> pd.DataFrame:
    robot = graph.robot
    N = robot.n + 1
    # converting 2d goals into 3d goals for solver_fabrik
    if robot.dim == 2:
        goals = list(goals)
        for i in range(len(goals)):
            goals[i] = list(goals[i]) + [0]

    # retrieving parents indices in the format of solver_fabrik
    if (type(robot).__name__ == "RobotPlanar") or (
        type(robot).__name__ == "RobotSpherical"
    ):
        parents_index = [-1] * len(robot.parents)
        for key in list(robot.parents.keys()):
            for child in robot.parents[key]:
                parents_index[int(child[1:])] = int(key[1:])
    else:
        parents_index = [-1] + list(range(N - 1))

    # Retrieving the position for the initial guess
    initial_guess = []
    guess = np.zeros((N, 3))

    if initial is not None:
        for j in range(len(initial)):
            for i in range(N):
                if robot.spherical:
                    init = list_to_variable_dict_spherical(initial[j], in_pairs=True)
                else:
                    init = list_to_variable_dict(initial[j])
                pos = list(robot.get_pose(init, "p" + str(i)).trans)
                if robot.dim == 2:
                    pos += [0]
                guess[i, :] = pos
                initial_guess += [guess]

    # solver_fabrik params
    dim = robot.dim
    parents = parents_index

    r = [0] * robot.n
    for i in range(robot.n):
        if dim == 2:
            r[i] = robot.a["p" + str(i + 1)]
        elif dim == 3:
            r[i] = robot.d["p" + str(i + 1)]

    angle_limit = [0] * N
    for i in range(1, N):
        if use_limits:
            angle_limit[i] = robot.ub["p" + str(i)]
        else:
            angle_limit[i] = np.pi
    goal_position = goals

    params = {
        "N": N,
        "r": r,
        "parents": parents,
        "angle_limit": angle_limit,
        "goal_index": goal_index,
        "goal_position": goal_position,
        "dim": dim,
    }

    final_errors = []
    max_ee_errors = []
    runtimes = []
    iterations = []
    solutions = []
    successes = []
    status = []
    angles = []
    # max_angle_violation = []
    limits_violated = []
    limit_violations = []
    pos_error = []
    rot_error = []
    pos_error_check = []

    if initial == None:
        initial = [-1]

    for i in range(len(initial)):
        # Initializing solver
        solver = solver_fabrik(params)

        # Solving
        if type(initial[0]).__name__ == "int":
            result = solver.solve(
                max_iteration=max_iteration,
                error_threshold=tol,
                sensitivity=0.00001,
                sensitivity_range=50,
            )
        else:
            result = solver.solve(
                initial_guess=initial_guess[i],
                max_iteration=max_iteration,
                error_threshold=tol,
                sensitivity=0.00001,
                sensitivity_range=50,
            )

        # Extract q_sol
        if dim == 2:
            P_e = result["positions"][:, 0:2]
            P_e = np.insert(P_e, 1, np.eye(2), 0)
        else:
            P_e = np.insert(result["positions"], 1, np.eye(3), 0)
        G_e = graph.graph_from_pos(P_e)
        try:
            q_sol = graph.robot.joint_variables(G_e)
        except np.linalg.LinAlgError:
            print("Breakpoint for lstsq error")
        # Determine rot_error with q_sol
        err_rot_ind = 0.0
        err_pos_check_ind = 0.0
        for key in ee_goals:
            T_sol = graph.robot.get_pose(q_sol, key)
            err_pos_check_ind += norm(ee_goals[key].trans - T_sol.trans)
            if dim == 2:
                err_rot_ind += safe_arccos(
                    (ee_goals[key].rot.dot(T_sol.rot.inv())).as_matrix()[0, 0]
                )
            else:
                z1 = ee_goals[key].rot.as_matrix()[0:3, -1]
                z2 = graph.robot.get_pose(q_sol, key).rot.as_matrix()[0:3, -1]
                err_rot_ind += safe_arccos(z1.dot(z2))

        limit_violations_ind = {}
        violated_ind = False
        for key in q_sol:
            if dim == 2:
                limit_violations_ind[key] = max(
                    graph.robot.lb[key] - q_sol[key], q_sol[key] - graph.robot.ub[key]
                )
            else:
                limit_violations_ind[key] = max(
                    graph.robot.lb[key] - q_sol[key][1],
                    q_sol[key][1] - graph.robot.ub[key],
                )
            if limit_violations_ind[key] > 0.01 * graph.robot.ub[key]:
                violated_ind = True
        limit_violations += [limit_violations_ind]
        limits_violated += [violated_ind]

        # max_violation = 0
        # for counter in range(1, solver.N):
        #     if result["angles"][counter] - solver.angle_limit[counter] > max_violation:
        #         max_violation = result["angles"][counter] - solver.angle_limit[counter]
        # if max_violation > 0.01*solver.angle_limit[counter]:
        #     limits_violated += [True]
        # else:
        #     limits_violated += [False]

        if verbosity == 1:
            print("Initialization %d:" % i)
            solver.print_solution(result, print_enabled=True, show=False)
        elif verbosity == 2:
            print("Initialization %d:" % i)
            solver.print_solution(result, print_enabled=True, show=True)

        # max_angle_violation += [max_violation]
        final_errors += [result["final_error"] ** 2]  # Need the squared error
        runtimes += [result["runtime"]]
        iterations += [result["iterations"]]
        solutions += [result["positions"]]
        successes += [result["success"]]
        status += [result["status"]]
        angles += [result["angles"]]
        max_ee_errors += [result["max_error_per_iteration"][-1]]
        pos_error += [result["final_error"]]
        rot_error += [err_rot_ind]
        pos_error_check += [err_pos_check_ind]

    data = dict(
        [
            ("Init.", initial),
            ("Error", final_errors),
            ("Runtime", runtimes),
            ("Iterations", iterations),
            ("Solution", solutions),
            ("Success", successes),
            ("Status", status),
            ("Limit Violations", limit_violations),
            ("Max End-Effector Error", max_ee_errors),
            ("Limits Violated", limits_violated),
            ("Pos Error", pos_error),
            ("Pos Error Check", pos_error_check),
            ("Rot Error", rot_error),
        ]
    )

    return pd.DataFrame(data)


def run_full_jacobian_sweep_experiment(
    graph: Graph,
    ee_goals: dict,
    init: list,
    q_goal: dict,
    params: dict = None,
    use_limits=False,
) -> pd.DataFrame:

    runtime = len(init) * [-1]
    ee_error = len(init) * [-1]
    max_ee_error = len(init) * [-1]
    max_angle_violation = len(init) * [-1]
    iterations = len(init) * [-1]
    ind = 0
    for init_angles in init:
        t1 = time.perf_counter()

        if graph.robot.spherical:
            init_angles = list_to_variable_dict_spherical(init_angles, in_pairs=True)
        else:
            init_angles = list_to_variable_dict(init_angles)

        q_sol, n_iter = jacobian_ik(
            graph.robot,
            init_angles,
            q_goal,
            params=params,
            use_limits=use_limits,
        )
        iterations[ind] = n_iter
        runtime[ind] = time.perf_counter() - t1

        q_sol = list_to_variable_dict(q_sol)
        for key in ee_goals:
            p_sol = graph.robot.get_pose(q_sol, key).trans
            # print(ee_goals[key] - p_sol)
            err_key = norm(ee_goals[key] - p_sol)
            ee_error[ind] += err_key ** 2
            max_ee_error[ind] = max(max_ee_error[ind], err_key)

        if use_limits:
            angle_violation = 0.0
            for key in q_sol:
                if graph.robot.spherical:
                    q_angle = q_sol[key][1]
                else:
                    q_angle = q_sol[key]
                angle_violation = max(
                    max(
                        graph.robot.lb[key] - q_angle,
                        q_angle - graph.robot.ub[key],
                    ),
                    angle_violation,
                )
            if angle_violation > 1e-6:
                print("--------------------------")
                print(
                    "Angle violated! \n Lower bounds: {:} \n Upper bounds: {:}".format(
                        graph.robot.lb, graph.robot.ub
                    )
                )
                print("q_sol: {:}".format(q_sol))
                print("--------------------------")
            max_angle_violation[ind] = angle_violation

        ind += 1
    # Form the dataframe with results
    columns = ["Init.", "Error", "Runtime", "Max Angle Violation", "Iterations"]
    data = dict(
        zip(
            columns,
            [
                init,
                ee_error,
                runtime,
                max_angle_violation,
                iterations,
            ],
        )
    )
    return pd.DataFrame(data)


def run_full_sweep_experiment(
    graph: Graph,
    ee_goals: dict,
    solver: LocalSolver,
    n_per_dim: int,
    init: list = None,
    use_limits=False,
    use_hess=True,
    pose_goals=False,
) -> pd.DataFrame:
    """

    :param graph:
    :param solver: LocalSolver object with cost-function pre-set
    :param n_per_dim:
    :param init: list of specific angle combinations to try
    :param angle_tol: tolerance on angle
    :return:
    """
    # TODO: include the fixed params as dataframe columns here? Solver, robot as a whole?
    # if init is None:
    #     angles = np.linspace(-np.pi, np.pi, n_per_dim)
    #     init = list(itertools.product(angles, repeat=graph.robot.n))
    if init is None:
        if n_per_dim <= 3:
            angles = np.linspace(-np.pi / 2.0, np.pi / 2.0, n_per_dim)
            init = list(itertools.product(angles, repeat=graph.robot.n))
        else:
            angles_sets = []
            for lower, upper in zip(graph.robot.lb.values(), graph.robot.ub.values()):
                angles_sets.append(np.linspace(lower, upper, n_per_dim))
            init = list(itertools.product(*angles_sets))

    error = len(init) * [0.0]
    max_ee_error = len(init) * [0.0]
    grad_norm = len(init) * [0.0]
    smallest_eig = len(init) * [np.nan]
    runtime = len(init) * [0.0]
    num_iters = len(init) * [0]
    solutions = len(init) * [None]
    max_angle_violation = len(init) * [0.0]
    # max_constraint_violation = len(init)*[0.]
    success = len(init) * [False]
    status = len(init) * [0]
    message = len(init) * [""]
    max_angle_error = len(init) * [0.0]
    ind = 0
    problem_params = {}
    if use_limits:
        problem_params["angular_limits"] = graph.robot.ub  # Assumes symmetrical limits
    # bar = Bar(
    #     "Local " + solver.params["solver"] + " solver",
    #     max=len(init),
    #     check_tty=False,
    #     hide_cursor=False,
    # )
    for init_angles in init:
        if graph.robot.spherical:
            problem_params["initial_guess"] = list_to_variable_dict_spherical(
                init_angles, in_pairs=True
            )
        else:
            problem_params["initial_guess"] = list_to_variable_dict(init_angles)
        results = solver.solve(graph, problem_params)
        if graph.robot.spherical:
            solutions[ind] = np.reshape(results.x, (-1, 2))
        else:
            solutions[ind] = results.x

        q_sol = list_to_variable_dict(solutions[ind])
        # error[ind] = results.fun
        for key in ee_goals:
            p_sol = graph.robot.get_pose(q_sol, key).trans
            err_key = (
                norm(ee_goals[key] - p_sol)
                if not pose_goals
                else norm(ee_goals[key].trans - p_sol)
            )
            error[ind] += err_key ** 2
            max_ee_error[ind] = max(max_ee_error[ind], err_key)
            if pose_goals:
                z1 = ee_goals[key].rot.as_matrix()[0:3, -1]
                z2 = graph.robot.get_pose(q_sol, key).rot.as_matrix()[0:3, -1]
                err_rot = safe_arccos(z1.dot(z2))
                # rot_error_so3 = (ee_goals[key].rot.dot(graph.robot.get_pose(q_sol, key).rot.inv())).log()
                # print("ROT ERROR so3: {:}".format(rot_error_so3))
                # err_rot = norm(rot_error_so3[0:2])
                max_angle_error[ind] = max(max_angle_error[ind], err_rot)

        grad_norm[ind] = norm(solver.grad(results.x))
        if use_hess:
            smallest_eig[ind] = min(
                np.linalg.eigvalsh(solver.hess(results.x).astype(float))
            )
        runtime[ind] = results.runtime
        num_iters[ind] = results.nit

        # print(results.keys())
        if use_limits:
            lb = np.array(list(graph.robot.lb.values()))
            ub = np.array(list(graph.robot.ub.values()))
            if graph.robot.spherical:
                ub = np.array(bounds_to_spherical_bounds(ub))
                lb = -ub
            angle_violation = max(
                max(np.max(lb - results.x), np.max(results.x - ub)), 0.0
            )
            if angle_violation > 1e-6:
                print("--------------------------")
                print(
                    "Angle violated! \n Lower bounds: {:} \n Upper bounds: {:}".format(
                        graph.robot.lb, graph.robot.ub
                    )
                )
                print("q_sol: {:}".format(results.x))
                print("--------------------------")
            # max_constraint_violation[ind] = results.maxcv
            max_angle_violation[ind] = angle_violation
        success[ind] = results.success
        status[ind] = results.status
        message[ind] = results.message
        ind += 1
    #     bar.next()
    # bar.finish()

    columns = [
        "Init.",
        "Error",
        "Gradient Norm",
        "Smallest Eig.",
        "Runtime",
        "Iterations",
        "Solution",
        "Max Angle Violation",
        "Success",
        "Status",
        "Message",
        "Max End-Effector Error",
        "Max Angular Error",
    ]
    data = dict(
        zip(
            columns,
            [
                init,
                error,
                grad_norm,
                smallest_eig,
                runtime,
                num_iters,
                solutions,
                max_angle_violation,
                success,
                status,
                message,
                max_ee_error,
                max_angle_error,
            ],
        )
    )

    return pd.DataFrame(data)


def limit_init_to_ball(init, target, r):
    return target + (init - target) * r / norm(init - target)


def scatter_error_between_solvers(
    data, solver1="Riemannian TrustRegions", solver2="L-BFGS-B", attribute="Error"
):
    plt.figure()
    plt.scatter(
        data[data["Solver"] == solver1][attribute],
        data[data["Solver"] == solver2][attribute],
        s=60,
        c="b",
    )
    plt.title("Riemannian Algorithm")
    plt.xlabel(solver1)
    plt.ylabel(solver2)
    plt.grid()
    plt.show()


def compute_successes(
    data,
    solver,
    attribute1="Pos Error",
    attribute2="Rot Error",
    tol_ee=1e-2,
    tol_ang=1e-2,
    use_limits=True,
):
    data_alg = data[data["Solver"] == solver]
    if use_limits:
        successes = (
            (data_alg[attribute1] < tol_ee)
            & (data_alg[attribute2] < tol_ang)
            & (data_alg["Limits Violated"] == False)
        )
    else:
        successes = (data_alg[attribute1] < tol_ee) & (data_alg[attribute2] < tol_ang)
    return successes  # data_alg[successes][attribute1].count()


def make_latex_results_table_old(data, tol_ee=1e-2, tol_ang=1e-2, use_limits=True):
    solvers = list(data["Solver"].unique())
    success_rates = []
    mean_iters = []
    mean_runtime = []
    mean_pos_errors = []
    mean_rot_errors = []

    for solver in solvers:
        data_alg = data[data["Solver"] == solver]
        successes = compute_successes(
            data, solver, tol_ee=tol_ee, tol_ang=tol_ang, use_limits=use_limits
        )
        success_rates.append(
            data_alg[successes]["Iterations"].count()
            / data[data["Solver"] == solver].shape[0]
        )
        if "Riemann" in solver:
            mean_iters.append(data_alg["Iterations"].mean() + 1)
        else:
            mean_iters.append(data_alg["Iterations"].mean())
        mean_runtime.append(data_alg["Runtime"].mean())

        # Only use non-constraint violating results
        mean_pos_errors.append(data_alg[successes]["Pos Error"].mean())
        mean_rot_errors.append(data_alg[successes]["Rot Error"].mean())

    table = pd.DataFrame(
        {
            "Solver": solvers,
            "Success Rate": success_rates,
            "Mean No. of Iterations": mean_iters,
            "Mean Runtime (s)": mean_runtime,
            "Mean Pos. Error (m)": mean_pos_errors,
            "Mean Rot. Error (rad)": mean_rot_errors,
        }
    )

    table = table.to_latex(
        index=False, float_format="%.3f"
    )  # , label=label, caption=caption)
    table_processed = ""
    our_results = ""
    for line in table.split("\n"):
        if "Riemannian" in line:
            our_results += "& " + line + "\n"
        elif (
            not any([s in line for s in ["rule", "tabular", "Iterations"]])
            and len(line) > 4
        ):
            table_processed += "& " + line + "\n"
    table_processed += our_results  # Places ours at the bottom
    local_algorithms_unbounded = [" BFGS &", " CG &", " Newton-CG &", " trust-exact &"]
    local_algorithms_bounded = [" L-BFGS-B &", " TNC &", " SLSQP &", " trust-constr &"]
    other_algs = [" FABRIK &", " Jacobian &"]
    rename_algs = [
        " Riemannian TrustRegions &",
        " Riemannian TrustRegions + BS &",
        " Riemannian ConjugateGradient &",
        " Riemannian ConjugateGradient + BS &",
    ]
    algs = (
        local_algorithms_unbounded + local_algorithms_bounded + other_algs + rename_algs
    )
    for alg in algs:
        table_processed = table_processed.replace(alg, " \\texttt{" + alg[1:-2] + "} &")

    new_alg_names_map = {
        "Riemannian TrustRegions}": "Riem. TR}",
        "Riemannian TrustRegions + BS}": "Riem. TR+BS}",
        "Riemannian ConjugateGradient}": "Riem. CG}",
        "Riemannian ConjugateGradient + BS}": "Riem. CG+BS}",
        "Jacobian}": "Jacobian DLS}",
    }
    for old_name in new_alg_names_map:
        table_processed = table_processed.replace(old_name, new_alg_names_map[old_name])

    return table_processed


def make_latex_results_table(data, tol_ee=1e-2, tol_ang=1e-2, use_limits=True):
    # solvers_in = list(data["Solver"].unique())
    if use_limits:
        solvers = [
            "trust-constr",
            "FABRIK",
            "Riemannian TrustRegions",
            "Riemannian TrustRegions + BS",
        ]
    else:
        solvers = [
            "trust-exact",
            "FABRIK",
            "Riemannian TrustRegions",
            "Riemannian TrustRegions + BS",
        ]

    output_string = ""

    for solver in solvers:
        data_alg = data[data["Solver"] == solver]
        successes = compute_successes(
            data, solver, tol_ee=tol_ee, tol_ang=tol_ang, use_limits=use_limits
        )
        n_successes = data_alg[successes]["Iterations"].count()
        n = data[data["Solver"] == solver].shape[0]
        success_rate, success_rad = bernoulli_confidence_jeffreys(
            n, n_successes, confidence=0.95
        )
        success_rate_pct = np.round(success_rate * 100)
        success_rad_pct = success_rad * 100
        output_string += (
            " & $"
            + "{:.1f}".format(success_rate_pct)
            + " \pm "
            + "{:.1f}".format(success_rad_pct)
            + "$"
        )
        if "Riemann" in solver:
            output_string += " & " + "{:.0f}".format(
                np.round(data_alg["Iterations"].mean() + 1)
            )
        else:
            output_string += " & " + "{:.0f}".format(
                np.round(data_alg["Iterations"].mean())
            )
        output_string += " ({:.0f})".format(np.round(data_alg["Iterations"].std()))

    return output_string + " \\\\"


def plot_waterfall_curve(
    data,
    attribute1="Pos Error",
    attribute2="Rot Error",
    tols_ee: list = None,
    tols_ang: list = None,
    use_limits=True,
    solver_list=None,
    solver_names=None,
    save_file=None,
    plot_y_label=True,
    make_legend=True,
    plot_confidence=True,
):

    if solver_list is None:
        solver_list = [
            "trust-constr",
            "FABRIK",
            "Riemannian TrustRegions",
            "Riemannian TrustRegions + BS",
        ]
    if solver_names is None:
        solver_names = [
            "\\texttt{trust-constr}",
            "\\texttt{FABRIK}",
            "\\texttt{Riem. TR}",
            "\\texttt{Riem. TR+BS}",
        ]

    if tols_ee is None and tols_ang is None:
        tols_ee = list(np.linspace(-6, 1, 8))
        tols_ang = [0.01]

    results_by_solver = {solver: [] for solver in solver_list}
    if plot_confidence:
        lower_bounds = {solver: [] for solver in solver_list}
        upper_bounds = {solver: [] for solver in solver_list}

    for tol_ee in tols_ee:
        for tol_ang in tols_ang:
            for solver in results_by_solver:
                data_alg = data[data["Solver"] == solver]
                successes = compute_successes(
                    data,
                    solver,
                    attribute1,
                    attribute2,
                    tol_ee=10 ** tol_ee,
                    tol_ang=tol_ang,
                    use_limits=use_limits,
                )
                n_successes = data_alg[successes]["Iterations"].count()
                results_by_solver[solver].append(n_successes / data_alg.shape[0])

                if plot_confidence:
                    n = data[data["Solver"] == solver].shape[0]
                    success_rate, success_rad = bernoulli_confidence_jeffreys(
                        n, n_successes, confidence=0.95
                    )
                    lower_bounds[solver].append(success_rate - success_rad)
                    upper_bounds[solver].append(success_rate + success_rad)

    # Plotting
    if len(tols_ee) > len(tols_ang):
        x_values = tols_ee
        x_label = "log$_{10}$ Pos. Error Tolerance (m)"
    else:
        x_values = tols_ang
        x_label = "Rot. Error Tolerance (rad)"

    linewidth = 1.3
    markersize = 5
    fig = plt.figure()
    plt.grid()

    if len(solver_list) == 3:
        line_colors_local = [line_colors[0]] + line_colors[2:]
        line_markers_local = [line_markers[0]] + line_markers[2:]
        linestyles_local = [linestyles[0]] + linestyles[2:]
    else:
        line_colors_local = line_colors
        line_markers_local = line_markers
        linestyles_local = linestyles

    for idx, solver in enumerate(results_by_solver):
        color = line_colors_local[idx % len(line_colors_local)]
        marker = line_markers_local[idx % len(line_markers_local)]
        style = linestyles_local[idx % len(linestyles_local)]
        plt.plot(
            x_values,
            results_by_solver[solver],
            color=color,
            marker=marker,
            linestyle=style,
            linewidth=linewidth,
            markersize=markersize + 2 * (marker in ("d", "*")),
            alpha=1.0,
        )
        if plot_confidence:
            plt.fill_between(
                x_values,
                lower_bounds[solver],
                upper_bounds[solver],
                facecolor=color,
                edgecolor=None,
                alpha=0.29,
            )

    plt.xlabel(x_label)

    if plot_y_label:
        plt.ylabel("Success Rate")
    plt.ylim([-0.05, 1.15])
    if make_legend:
        plt.legend(solver_names, loc="lower left")
    fig.set_size_inches(8, 5)
    if save_file is not None:
        fig.tight_layout()
        fig.savefig(save_file, bbox_inches="tight")
    return plt.gcf(), plt.gca()


def create_box_plot_from_data(
    data,
    attribute="Pos Error",
    solver_list=None,
    solver_names=None,
    xlabel=None,
    ylabel=None,
    save_file=None,
):
    if solver_list is None:
        solver_list = [
            "trust-constr",
            "FABRIK",
            "Riemannian TrustRegions",
            "Riemannian TrustRegions + BS",
        ]
    if solver_names is None:
        solver_names = [
            "\\texttt{trust-constr}",
            "\\texttt{FABRIK}",
            "\\texttt{Riem. TR}",
            "\\texttt{Riem. TR+BS}",
        ]

    fig, axes = plt.subplots()
    lw = 0.25
    widths = 0.6
    boxprops = dict(linewidth=0.4)
    whiskerprops = dict(linewidth=2 * lw)
    flierprops = dict(marker="o", markersize=5, markeredgewidth=lw)
    medianprops = dict(linewidth=lw, color="k")
    plot_data = [
        np.log10(data[data["Solver"] == solver][attribute]) for solver in solver_list
    ]
    bp = axes.boxplot(
        plot_data,
        widths=widths,
        patch_artist=True,
        notch=True,
        flierprops=flierprops,
        whiskerprops=whiskerprops,
        medianprops=medianprops,
    )
    axes.set(xticklabels=solver_names)  # , xlabel=xlabel)

    if len(solver_list) == 3:
        line_colors_local = [line_colors[0]] + line_colors[2:]
    else:
        line_colors_local = line_colors

    for p_i in range(len(bp["boxes"])):
        # print("Box index: {:}".format(p_i))
        bp["boxes"][p_i].set(
            facecolor=to_rgba(line_colors_local[p_i], 0.4),
            edgecolor=line_colors_local[p_i],
        )
        bp["medians"][p_i].set(color=line_colors_local[p_i])
        bp["fliers"][p_i].set(markeredgecolor=line_colors_local[p_i])

    for p_i in range(len(bp["whiskers"])):
        bp["whiskers"][p_i].set(color=line_colors_local[p_i // 2])
        bp["caps"][p_i].set(color=line_colors_local[p_i // 2])

    # axes[a_i].set_yscale('log')
    # axes.set_yticks(np.linspace(-25, 1, 27), minor=True)
    axes.grid(True, which="both", color="k", linestyle="--", alpha=0.5, linewidth=0.25)
    axes.set_ylabel(ylabel)
    fig.set_size_inches(8, 5)

    if save_file is not None:
        fig.tight_layout()
        fig.savefig(save_file, bbox_inches="tight")

    return fig, axes


if __name__ == "__main__":
    import pickle

    data = pickle.load(
        open(
            "/Users/mattgiamou/stars/code/graphik/experiments/tro2020/results/planar_tree_dof_6_bounded_True_tol_1e-06_maxiter_100_n_goals_100_bound_smoothing_False_full_results.p",
            "rb",
        )
    )
    table = make_latex_results_table(data, tol=1e-4)
    print(table)
