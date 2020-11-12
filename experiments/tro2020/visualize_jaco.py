import pickle
import numpy as np
from cvxik.utils.experiments import make_latex_results_table, plot_waterfall_curve, scatter_error_between_solvers, \
                                    create_box_plot_from_data


if __name__ == '__main__':
    # File loading params
    use_limits = False
    tol = 1e-9
    save = False
    n_goals = 100
    # Visualization params
    tol_ee = 1e-2
    tol_ang = np.pi/180./10.
    tols_ee_waterfall = list(np.linspace(-8, 0, 9))

    jaco_file = "results/jaco_bounded_" + str(use_limits) + "_tol_" + str(tol) + "_maxiter_200_n_goals_" + str(n_goals) + \
                "_n_init_1_zero_init_True_full_results.p"

    data = pickle.load(open(jaco_file, "rb"))
    table = make_latex_results_table(data, tol_ee=tol_ee, tol_ang=tol_ang, use_limits=use_limits)
    print(table)

    if use_limits:
        solver_list = ["trust-constr", "Jacobian", "Riemannian TrustRegions", "Riemannian TrustRegions + BS"]
        solver_names = ["\\texttt{trust-constr}", "\\texttt{DLS}", "\\texttt{Riem. TR}", "\\texttt{Riem. TR+BS}"]
    else:
        solver_list = ["trust-exact", "Jacobian", "Riemannian TrustRegions", "Riemannian TrustRegions + BS"]
        solver_names = ["\\texttt{trust-exact}", "\\texttt{DLS}", "\\texttt{Riem. TR}", "\\texttt{Riem. TR+BS}"]

    if save:
        save_waterfall = "results/figs/waterfall_jaco_bounded_{:}.pdf".format(use_limits)
        save_boxplot = "results/figs/boxplot_jaco_bounded_{:}.pdf".format(use_limits)
    else:
        save_waterfall = None
        save_boxplot = None

    fig, ax = plot_waterfall_curve(data, use_limits=use_limits, save_file=save_waterfall,
                                   solver_list=solver_list, solver_names=solver_names,
                                   tols_ang=[tol_ang], tols_ee=tols_ee_waterfall, plot_y_label=True)
    xlabel = "Solver"
    ylabel = " log$_{10}$ Max. Error (m)"

    create_box_plot_from_data(data, attribute="Max End-Effector Error",
                              solver_list=solver_list, solver_names=solver_names,
                              xlabel=xlabel, ylabel=ylabel, save_file=save_boxplot)
