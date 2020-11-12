import pickle
import pandas as pd
import numpy as np


if __name__ == '__main__':
    chain_dofs = [6, 10]
    tree_dofs = [14, 30]
    tol = 1e-9

    # spherical chains
    spherical_chain = []
    spherical_chain_bounded = []
    for dof in chain_dofs:
        spherical_chain_file = "results/spherical_chain_dof_" + str(dof) + "_bounded_" + str(False) + \
                           "_tol_" + str(tol) + "_maxiter_2000_n_goals_1000_n_init_1_zero_init_True_full_results.p"
        spherical_chain_data = pickle.load(open(spherical_chain_file, "rb"))

        spherical_chain_bounded_file = "results/spherical_chain_dof_" + str(dof) + "_bounded_" + str(True) + \
                                   "_tol_" + str(tol) + "_maxiter_2000_n_goals_1000_n_init_1_zero_init_True_full_results.p"
        spherical_chain_bounded_data = pickle.load(open(spherical_chain_bounded_file, "rb"))
        spherical_chain.append(spherical_chain_data)
        spherical_chain_bounded.append(spherical_chain_bounded_data)

    # spherical trees
    spherical_tree = []
    spherical_tree_bounded = []
    for dof in tree_dofs:
        spherical_tree_file = "results/spherical_tree_dof_"+str(dof)+"_bounded_" + str(False) + \
                           "_tol_"+str(tol)+"_maxiter_2000_n_goals_1000_n_init_1_zero_init_True_full_results.p"
        spherical_tree_data = pickle.load(open(spherical_tree_file, "rb"))

        spherical_tree_bounded_file = "results/spherical_tree_dof_" + str(dof) + "_bounded_" + str(True) + \
                           "_tol_" + str(tol) + "_maxiter_2000_n_goals_1000_n_init_1_zero_init_True_full_results.p"
        spherical_tree_bounded_data = pickle.load(open(spherical_tree_bounded_file, "rb"))
        spherical_tree.append(spherical_tree_data)
        spherical_tree_bounded.append(spherical_tree_bounded_data)


    spherical_chain_data = pd.concat(spherical_chain)
    spherical_chain_bounded_data = pd.concat(spherical_chain_bounded)
    spherical_tree_bounded_data = pd.concat(spherical_tree_bounded)
    spherical_tree_data = pd.concat(spherical_tree)

    # Combine spherical results
    fabrik_results_list = [spherical_chain_data[spherical_chain_data["Solver"] == "FABRIK"]["Runtime"],
                           spherical_chain_bounded_data[spherical_chain_bounded_data["Solver"] == "FABRIK"]["Runtime"],
                           spherical_tree_data[spherical_tree_data["Solver"] == "FABRIK"]["Runtime"],
                           spherical_tree_bounded_data[spherical_tree_bounded_data["Solver"] == "FABRIK"]["Runtime"]]
    fabrik_results = pd.concat(fabrik_results_list)

    local_results_list = [spherical_chain_data[spherical_chain_data["Solver"] == "trust-exact"]["Runtime"],
                           spherical_chain_bounded_data[spherical_chain_bounded_data["Solver"] == "trust-constr"]["Runtime"],
                           spherical_tree_data[spherical_tree_data["Solver"] == "trust-exact"]["Runtime"],
                           spherical_tree_bounded_data[spherical_tree_bounded_data["Solver"] == "trust-constr"]["Runtime"]]
    local_results = pd.concat(local_results_list)

    riem_results_list = [spherical_chain_data[spherical_chain_data["Solver"] == "Riemannian TrustRegions"]["Runtime"],
                          spherical_chain_bounded_data[spherical_chain_bounded_data["Solver"] == "Riemannian TrustRegions"]["Runtime"],
                          spherical_tree_data[spherical_tree_data["Solver"] == "Riemannian TrustRegions"]["Runtime"],
                          spherical_tree_bounded_data[spherical_tree_bounded_data["Solver"] == "Riemannian TrustRegions"]["Runtime"]]

    riem_bs_results_list = [spherical_chain_data[spherical_chain_data["Solver"] == "Riemannian TrustRegions + BS"]["Runtime"],
                         spherical_chain_bounded_data[spherical_chain_bounded_data["Solver"] ==
                                                   "Riemannian TrustRegions + BS"]["Runtime"],
                         spherical_tree_data[spherical_tree_data["Solver"] == "Riemannian TrustRegions + BS"]["Runtime"],
                         spherical_tree_bounded_data[spherical_tree_bounded_data["Solver"] ==
                                                  "Riemannian TrustRegions + BS"]["Runtime"]]
    riem_results = pd.concat(riem_results_list + riem_bs_results_list)

    ### Print spherical results
    print("FABRIK results: ")
    print(fabrik_results.describe())

    print("Local results: ")
    print(local_results.describe())

    print("Riem. TR (+ BS) results: ")
    print(riem_results.describe())

