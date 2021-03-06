import pickle
import numpy as np
from cvxik.utils.experiments import make_latex_results_table, plot_waterfall_curve, scatter_error_between_solvers, \
                                    create_box_plot_from_data


if __name__ == '__main__':
    # File loading params
    use_limits = False
    tol = 1e-9
    dofs = [14, 30] #, 14, 30] #, 30]
    save = True

    # Visualization params
    tol_ee = 1e-2
    tol_ang = 1e-2
    tols_ee_waterfall = list(np.linspace(-6, 1, 8))
    for dof in dofs:
        n_goals = 1000
        spherical_tree_file = "results/spherical_tree_dof_"+str(dof)+"_bounded_" + str(use_limits) + \
                           "_tol_"+str(tol)+"_maxiter_2000_n_goals_"+str(n_goals)+"_n_init_1_zero_init_True_full_results.p"
        data = pickle.load(open(spherical_tree_file, "rb"))
        table = make_latex_results_table(data, tol_ee=tol_ee, tol_ang=tol_ang, use_limits=use_limits)
        print(table)

        if use_limits:
            solver_list = ["trust-constr", "FABRIK", "Riemannian TrustRegions", "Riemannian TrustRegions + BS"]
            solver_names = ["\\texttt{trust-constr}", "\\texttt{FABRIK}", "\\texttt{Riem. TR}", "\\texttt{Riem. TR+BS}"]
        else:
            solver_list = ["trust-exact", "FABRIK", "Riemannian TrustRegions", "Riemannian TrustRegions + BS"]
            solver_names = ["\\texttt{trust-exact}", "\\texttt{FABRIK}", "\\texttt{Riem. TR}", "\\texttt{Riem. TR+BS}"]

        if save:
            save_waterfall = "results/figs/waterfall_spherical_tree_dof_{:}_bounded_{:}.pdf".format(dof, use_limits)
            save_boxplot = "results/figs/boxplot_spherical_tree_dof_{:}_bounded_{:}.pdf".format(dof, use_limits)
        else:
            save_waterfall = None
            save_boxplot = None

        plot_y_label = (dof == dofs[0])
        fig, ax = plot_waterfall_curve(data, use_limits=use_limits, save_file=save_waterfall,
                                       solver_list=solver_list, solver_names=solver_names,
                                       tols_ang=[tol_ang], tols_ee=tols_ee_waterfall, plot_y_label=plot_y_label)
        xlabel = "Solver"
        if dof == dofs[0]:
            ylabel = " log$_{10}$ Pos. Error (m)"
        else:
            ylabel = None
        create_box_plot_from_data(data, attribute="Pos Error",
                                  solver_list=solver_list, solver_names=solver_names,
                                  xlabel=xlabel, ylabel=ylabel, save_file=save_boxplot)

