import pickle
import pandas as pd
import numpy as np


if __name__ == '__main__':
    chain_dofs = [6, 10]
    tree_dofs = [14, 30]
    tol = 1e-9

    # Planar chains
    planar_chain = []
    planar_chain_bounded = []
    for dof in chain_dofs:
        planar_chain_file = "results/planar_chain_dof_" + str(dof) + "_bounded_" + str(False) + \
                           "_tol_" + str(tol) + "_maxiter_2000_n_goals_1000_n_init_1_zero_init_True_full_results.p"
        planar_chain_data = pickle.load(open(planar_chain_file, "rb"))

        planar_chain_bounded_file = "results/planar_chain_dof_" + str(dof) + "_bounded_" + str(True) + \
                                   "_tol_" + str(tol) + "_maxiter_2000_n_goals_1000_n_init_1_zero_init_True_full_results.p"
        planar_chain_bounded_data = pickle.load(open(planar_chain_bounded_file, "rb"))
        planar_chain.append(planar_chain_data)
        planar_chain_bounded.append(planar_chain_bounded_data)

    # Planar trees
    planar_tree = []
    planar_tree_bounded = []
    for dof in tree_dofs:
        planar_tree_file = "results/planar_tree_dof_"+str(dof)+"_bounded_" + str(False) + \
                           "_tol_"+str(tol)+"_maxiter_2000_n_goals_1000_n_init_1_zero_init_True_full_results.p"
        planar_tree_data = pickle.load(open(planar_tree_file, "rb"))

        planar_tree_bounded_file = "results/planar_tree_dof_" + str(dof) + "_bounded_" + str(True) + \
                           "_tol_" + str(tol) + "_maxiter_2000_n_goals_1000_n_init_1_zero_init_True_full_results.p"
        planar_tree_bounded_data = pickle.load(open(planar_tree_bounded_file, "rb"))
        planar_tree.append(planar_tree_data)
        planar_tree_bounded.append(planar_tree_bounded_data)


    planar_chain_data = pd.concat(planar_chain)
    planar_chain_bounded_data = pd.concat(planar_chain_bounded)
    planar_tree_bounded_data = pd.concat(planar_tree_bounded)
    planar_tree_data = pd.concat(planar_tree)

    # Combine planar results
    fabrik_results_list = [planar_chain_data[planar_chain_data["Solver"] == "FABRIK"]["Runtime"],
                           planar_chain_bounded_data[planar_chain_bounded_data["Solver"] == "FABRIK"]["Runtime"],
                           planar_tree_data[planar_tree_data["Solver"] == "FABRIK"]["Runtime"],
                           planar_tree_bounded_data[planar_tree_bounded_data["Solver"] == "FABRIK"]["Runtime"]]
    fabrik_results = pd.concat(fabrik_results_list)

    local_results_list = [planar_chain_data[planar_chain_data["Solver"] == "trust-exact"]["Runtime"],
                           planar_chain_bounded_data[planar_chain_bounded_data["Solver"] == "trust-constr"]["Runtime"],
                           planar_tree_data[planar_tree_data["Solver"] == "trust-exact"]["Runtime"],
                           planar_tree_bounded_data[planar_tree_bounded_data["Solver"] == "trust-constr"]["Runtime"]]
    local_results = pd.concat(local_results_list)

    riem_results_list = [planar_chain_data[planar_chain_data["Solver"] == "Riemannian TrustRegions"]["Runtime"],
                          planar_chain_bounded_data[planar_chain_bounded_data["Solver"] == "Riemannian TrustRegions"]["Runtime"],
                          planar_tree_data[planar_tree_data["Solver"] == "Riemannian TrustRegions"]["Runtime"],
                          planar_tree_bounded_data[planar_tree_bounded_data["Solver"] == "Riemannian TrustRegions"]["Runtime"]]

    riem_bs_results_list = [planar_chain_data[planar_chain_data["Solver"] == "Riemannian TrustRegions + BS"]["Runtime"],
                         planar_chain_bounded_data[planar_chain_bounded_data["Solver"] ==
                                                   "Riemannian TrustRegions + BS"]["Runtime"],
                         planar_tree_data[planar_tree_data["Solver"] == "Riemannian TrustRegions + BS"]["Runtime"],
                         planar_tree_bounded_data[planar_tree_bounded_data["Solver"] ==
                                                  "Riemannian TrustRegions + BS"]["Runtime"]]
    riem_results = pd.concat(riem_results_list + riem_bs_results_list)

    ### Print planar results
    print("FABRIK results: ")
    print(fabrik_results.describe())

    print("Local results: ")
    print(local_results.describe())

    print("Riem. TR (+ BS) results: ")
    print(riem_results.describe())

