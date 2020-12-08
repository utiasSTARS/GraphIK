import numpy as np
import cvxpy as cp
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import mosek
from graphik.utils.utils import list_to_variable_dict, measure_perturbation, constraint_violations
from graphik.solvers.constraints import constraints_from_graph, nearest_neighbour_cost, get_full_revolute_nearest_point
from graphik.solvers.sdp_formulations import SdpSolverParams
from graphik.solvers.solver_generic_sdp import SdpRelaxationSolver
from graphik.utils.robot_visualization import plot_heatmap
from graphik.utils.roboturdf import load_ur10

from progress.bar import ShadyBar as Bar
from matplotlib import rc
rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 18})
rc("text", usetex=True)


def ball_perturb(points: dict, r_max=1.0):
    dim = len(list(points.values())[0])
    n = len(points)*dim
    perturb_vals = 2.*r_max*np.random.rand(n) - r_max
    if np.linalg.norm(perturb_vals) > r_max:
        perturb_vals = r_max*perturb_vals/np.linalg.norm(perturb_vals)
    perturb_index = 0
    perturbed_points = {}
    for idx, key in enumerate(points.keys()):
        perturbed_points[key] = points[key] + perturb_vals[idx*dim:(idx+1)*dim]
        perturb_index += 1
    return perturbed_points


def box_perturb(points, perturb_max=1.0):
    dim = len(list(points.values())[0])
    n = len(points) * dim
    perturb_vals = 2.*perturb_max*np.random.rand(n) - perturb_max
    perturb_index = 0
    perturbed_points = {}
    for idx, key in enumerate(points.keys()):
        perturbed_points[key] = points[key] + perturb_vals[idx * dim:(idx + 1) * dim]
        perturb_index += 1
    return perturbed_points


if __name__ == '__main__':
    # Experiment params
    # Perturbation from zero cost case and matrices for storing results
    seed = 8675309
    np.random.seed(seed)
    force_dense = False  # Work with sparsity later (it should be equivalent but slower)
    n_runs = 10  # 40
    perturb_in_ball = True
    max_perturb = 0.1
    save_string = 'results/ur10_perturbed_reachable_goals_' + f'n_runs_{n_runs}_' + f'sparse_{not force_dense}_' \
                  + f'ball_{perturb_in_ball}_' + f'max_perturb_{max_perturb}_'
    run_experiment = True

    if run_experiment:
        eigval_ratio = np.zeros(n_runs)
        second_eigval_mag = np.zeros(n_runs)
        sdp_cost = np.zeros(n_runs)
        primal_cost = np.zeros(n_runs)
        gap = np.zeros(n_runs)
        max_constraint_violation = np.zeros(n_runs)
        max_nearest_points_dist = np.zeros(n_runs)
        total_nearest_points_dist = np.zeros(n_runs)
        runtime = np.zeros(n_runs)
        max_perturbation = np.zeros(n_runs)
        total_perturbation = np.zeros(n_runs)
        n_failed_runs = 0

        robot, graph = load_ur10()
        n = robot.n
        # q_goal = list_to_variable_dict(np.zeros(n))
        q_goal = list_to_variable_dict(np.random.rand(n)*2.*np.pi - np.pi)
        true_points = get_full_revolute_nearest_point(graph, q_goal)
        # Just use the final position goal (not pose involving 'q6' as well) since there's no angular limit
        ee_goals = {ee[0]: robot.get_pose(q_goal, ee[0]).trans for ee in robot.end_effectors}
        prob_params = {"end_effector_assignment": ee_goals}

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
        bar = Bar("", max=n_runs, check_tty=False, hide_cursor=False)
        for idx in range(n_runs):
            solver_worked = False
            while not solver_worked:
                if perturb_in_ball:
                    nearest_points = ball_perturb(true_points, max_perturb)
                else:
                    nearest_points = box_perturb(true_points, max_perturb)
                total_perturb_idx, max_perturb_idx = measure_perturbation(true_points, nearest_points)
                total_perturbation[idx] = total_perturb_idx
                max_perturbation[idx] = max_perturb_idx
                solver.cost = nearest_neighbour_cost(graph, nearest_points)
                try:
                    solution_dict, ranks, prob, constraints_ij = solver.solve(graph, prob_params)
                    solver_worked = True
                except cp.error.SolverError as e:
                    n_failed_runs += 1
                    print("---------------")
                    print("SOLVER ERROR at {:}".format(nearest_points))
                    print("---------------")


            runtime[idx] = prob.solver_stats.solve_time
            # Compute eigval parameters
            Z = prob.variables()[0].value
            Z_eigvals = np.linalg.eigvalsh(Z)
            second_eigval_mag[idx] = np.abs(Z_eigvals[-2])
            eigval_ratio[idx] = np.abs(Z_eigvals[-1] / Z_eigvals[-2])
            # Compute relaxation gap
            sdp_cost[idx] = prob.value
            primal_cost[idx] = solver.cost.subs(solution_dict)
            gap[idx] = primal_cost[idx] - sdp_cost[idx]
            # Compute max constraint violation
            violations = constraint_violations(constraints_ij, solution_dict)
            # robot_graph.plot_solution(solution_dict)
            eq_resid = [resid for (resid, is_eq) in violations if is_eq]
            ineq_resid = [resid for (resid, is_eq) in violations if not is_eq]
            eq_resid_max = np.max(np.abs(eq_resid))
            max_constraint_violation[idx] = eq_resid_max

            bar.next()
        bar.finish()

        # Form dataframe
        data = [('Primal Cost', primal_cost),
                ('SDP Cost', sdp_cost),
                ('Relaxation Gap', gap),
                ('Max. Constraint Violation', max_constraint_violation),
                ('Runtime', runtime),
                ('Eigenvalue Ratio', eigval_ratio),
                ('Second Eigenvalue Magnitude', second_eigval_mag),
                ('Max. Perturbation', max_perturbation),
                ('Total Perturbation', total_perturbation)]
        results = pd.DataFrame(dict(data))

        # Pickle results
        pickle.dump(results, open(save_string + "full_results.p", "wb"))

    else:
        results = pickle.load(open(save_string + 'full_results.p', "rb"))

    # Do some visualizations
    plt.figure()
    plt.scatter(
        results['Relaxation Gap'],
        results['Max. Constraint Violation'],
        s=60,
        c="b",
    )
    plt.xlabel('Relaxation Gap (m$^2$)')
    plt.ylabel('Max. Constr. Violation (m$^2$)')
    plt.grid()
    plt.show()

    plt.figure()
    plt.scatter(
        results['Eigenvalue Ratio'],
        results['Max. Constraint Violation'],
        s=60,
        c="b"
    )
    plt.xscale('log')
    plt.xlabel('Eigenvalue Ratio')
    plt.ylabel('Max. Constr. Violation (m$^2$)')
    plt.grid()
    plt.show()

    # Output correlation table
    print(results.drop(['Eigenvalue Ratio', 'SDP Cost'], axis=1).corr().to_markdown(floatfmt=".3f"))  # Can use to_latex() as well