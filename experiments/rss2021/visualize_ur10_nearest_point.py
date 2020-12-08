import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import mosek
from cvxik.utils.utils import list_to_variable_dict, constraint_violations
from cvxik.solvers.constraints import constraints_from_graph, nearest_neighbour_cost, get_full_revolute_nearest_point
from cvxik.solvers.sdp_formulations import SdpSolverParams
from cvxik.solvers.solver_base import SdpRelaxationSolver
from cvxik.utils.robot_visualization import plot_heatmap
from cvxik.graphs.graph_base import load_ur10

from matplotlib import rc
rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 18})
rc("text", usetex=True)
from progress.bar import ShadyBar as Bar


if __name__ == '__main__':
    # Experiment params
    # Perturbation from zero cost case and matrices for storing results
    force_dense = False  # Work with sparsity later (it should be equivalent but slower)
    perturbed_variable = 'p2'
    n_steps_perturb = 40  # 40
    max_perturb = 2.0  # 0.75 # 5. #20
    perturb_x = np.linspace(-max_perturb, max_perturb, n_steps_perturb)
    perturb_y = perturb_x
    perturb_X, perturb_Y = np.meshgrid(perturb_x, perturb_y)
    eigval_ratio = np.zeros(perturb_X.shape)
    second_eigval_mag = np.zeros(perturb_X.shape)
    sdp_cost = np.zeros(perturb_X.shape)
    primal_cost = np.zeros(perturb_X.shape)
    gap = np.zeros(perturb_X.shape)
    max_constraint_violation = np.zeros(perturb_X.shape)
    runtime = np.zeros(perturb_X.shape)

    robot, graph = load_ur10()
    n = robot.n
    q_goal = list_to_variable_dict(np.zeros(n))
    # Just use the final position goal (not pose involving 'q6' as well) since there's no angular limit
    ee_goals = {ee[0]: robot.get_pose(q_goal, ee[0]).trans for ee in robot.end_effectors}
    cons = constraints_from_graph(graph, ee_goals)

    # Use exact nearest points (only on ps for now, figure out how to get qs)
    # nearest_points = {joint: robot.get_pose(q_goal, joint).trans for joint in ['p'+str(idx) for idx in range(1, n)]}
    nearest_points = get_full_revolute_nearest_point(graph, q_goal)
    cost = nearest_neighbour_cost(graph, nearest_points)

    # Try SDP solver
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
    solver = SdpRelaxationSolver(
        params=sdp_solver_params, verbose=False, force_dense=force_dense
    )
    bar = Bar("", max=n_steps_perturb**2, check_tty=False, hide_cursor=False)
    for idx, dx in enumerate(perturb_x):
        for jdx, dy in enumerate(perturb_y):
            print("dx: {:}, dy: {:}".format(dx, dy))
            nearest_points_ij = get_full_revolute_nearest_point(graph, q_goal)
            nearest_points_ij[perturbed_variable] = nearest_points_ij[perturbed_variable] + np.array([dx, dy, 0.])
            solver.cost = nearest_neighbour_cost(graph, nearest_points_ij)
            prob_params = {"end_effector_assignment": ee_goals}
            try:
                solution_dict, ranks, prob, constraints_ij = solver.solve(graph, prob_params)
            except cp.error.SolverError as e:
                print("---------------")
                print("SOLVER ERROR at dx={:}, dy={:}".format(dx, dy))
                print("---------------")
                second_eigval_mag[idx, jdx] = 1e6
                eigval_ratio[idx, jdx] = 1.0
                gap[idx, jdx] = 1e6
                runtime[idx, jdx] = prob.solver_stats.solve_time
                max_constraint_violation[idx, jdx] = 1e6
                bar.next()
                continue

            runtime[idx, jdx] = prob.solver_stats.solve_time
            # Compute eigval parameters
            Z = prob.variables()[0].value
            Z_eigvals = np.linalg.eigvalsh(Z)
            second_eigval_mag[idx, jdx] = np.abs(Z_eigvals[-2])
            eigval_ratio[idx, jdx] = np.abs(Z_eigvals[-1] / Z_eigvals[-2])
            # Compute relaxation gap
            sdp_cost[idx, jdx] = prob.value
            primal_cost[idx, jdx] = solver.cost.subs(solution_dict)
            gap[idx, jdx] = primal_cost[idx, jdx] - sdp_cost[idx, jdx]
            # Compute max constraint violation
            violations = constraint_violations(constraints_ij, solution_dict)
            # robot_graph.plot_solution(solution_dict)
            eq_resid = [resid for (resid, is_eq) in violations if is_eq]
            ineq_resid = [resid for (resid, is_eq) in violations if not is_eq]
            eq_resid_max = np.max(np.abs(eq_resid))
            max_constraint_violation[idx, jdx] = eq_resid_max

            bar.next()

    bar.finish()

    # Plot all the heatmaps (linear scale)
    plot_heatmap(
        perturb_X, perturb_Y, np.log10(np.abs(gap)), title="Relaxation Gap (log$_{10}$)", xlabel="x", ylabel="y"
    )
    plt.show()
    plot_heatmap(
        perturb_X, perturb_Y, sdp_cost, title="SDP Cost", xlabel="x", ylabel="y"
    )
    plt.show()
    plot_heatmap(
        perturb_X, perturb_Y, primal_cost, title="Primal Cost", xlabel="x", ylabel="y"
    )
    plt.show()
    # plot_heatmap(
    #     perturb_X,
    #     perturb_Y,
    #     max_constraint_violation,
    #     title="Max Constraint Violation",
    #     xlabel="x",
    #     ylabel="y",
    # )
    # plt.show()
    # plot_heatmap(
    #     perturb_X,
    #     perturb_Y,
    #     second_eigval_mag,
    #     title="Second Largest Eigenvalue Magnitude",
    #     xlabel="x",
    #     ylabel="y",
    # )
    # plt.show()
    # plot_heatmap(
    #     perturb_X,
    #     perturb_Y,
    #     eigval_ratio,
    #     title="First and Second Eigenvalue Magn. Ratio",
    #     xlabel="x",
    #     ylabel="y",
    # )
    # plt.show()
    # Now log scale
    # plot_heatmap(perturb_X, perturb_Y, np.log10(gap), title='Relaxation Gap (Log 10)', xlabel='x', ylabel='y')
    plot_heatmap(
        perturb_X,
        perturb_Y,
        np.log10(max_constraint_violation),
        title="Max Constraint Violation (Log 10)",
        xlabel="x",
        ylabel="y",
    )
    plt.show()
    plot_heatmap(
        perturb_X,
        perturb_Y,
        np.log10(second_eigval_mag),
        title="Second Largest Eigenvalue Magnitude (Log 10)",
        xlabel="x",
        ylabel="y",
    )
    plt.show()