import pickle
import numpy as np
from cvxik.utils.experiments import make_latex_results_table, plot_waterfall_curve, scatter_error_between_solvers, \
                                    create_box_plot_from_data


if __name__ == '__main__':
    # File loading params
    use_limits = True
    tol = 1e-9
    save = True
    n_goals = 2000
    # Visualization params
    tol_ee = 0.01
    tol_ang = 0.01
    tols_ee_waterfall = list(np.linspace(-6, 1, 8))

    ur10_file = "results/ur10_bounded_" + str(use_limits) + "_tol_" + str(tol) + "_maxiter_2000_n_goals_" + str(n_goals) + \
                "_n_init_1_zero_init_True_full_results.p"

    data = pickle.load(open(ur10_file, "rb"))
    table = make_latex_results_table(data, tol_ee=tol_ee, tol_ang=tol_ang, use_limits=use_limits)
    print(table)

    if use_limits:
        solver_list = ["trust-constr", "Riemannian TrustRegions", "Riemannian TrustRegions + BS"]
        solver_names = ["\\texttt{trust-constr}", "\\texttt{Riem. TR}", "\\texttt{Riem. TR+BS}"]
    else:
        solver_list = ["trust-exact", "Riemannian TrustRegions", "Riemannian TrustRegions + BS"]
        solver_names = ["\\texttt{trust-exact}", "\\texttt{Riem. TR}", "\\texttt{Riem. TR+BS}"]

    if save:
        save_waterfall = "results/figs/waterfall_ur10_bounded_{:}.pdf".format(use_limits)
        save_boxplot = "results/figs/boxplot_ur10_bounded_{:}.pdf".format(use_limits)
        save_boxplot_rot = "results/figs/boxplot_rot_ur10_bounded_{:}.pdf".format(use_limits)
    else:
        save_waterfall = None
        save_boxplot = None
        save_boxplot_rot = None

    fig, ax = plot_waterfall_curve(data, use_limits=use_limits, save_file=save_waterfall,
                                   solver_list=solver_list, solver_names=solver_names,
                                   tols_ang=[tol_ang], tols_ee=tols_ee_waterfall, plot_y_label=True)
    xlabel = "Solver"
    ylabel = " log$_{10}$ Pos. Error (m)"

    create_box_plot_from_data(data, attribute="Pos Error",
                              solver_list=solver_list, solver_names=solver_names,
                              xlabel=xlabel, ylabel=ylabel, save_file=save_boxplot)

    ylabel_rot = " log$_{10}$ Rot. Error (rad)"
    create_box_plot_from_data(data, attribute="Rot Error",
                              solver_list=solver_list, solver_names=solver_names,
                              xlabel=xlabel, ylabel=ylabel_rot, save_file=save_boxplot_rot)
