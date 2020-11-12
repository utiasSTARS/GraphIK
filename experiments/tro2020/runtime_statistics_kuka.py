import pickle
import pandas as pd
import numpy as np


if __name__ == '__main__':
    # File loading params
    tol = 1e-9
    save = True
    n_goals = 2000
    # Visualization params
    tol_ee = 0.01
    tol_ang = 0.01
    tols_ee_waterfall = list(np.linspace(-6, 1, 8))

    robot_file = "results/lwa4d_bounded_" + "False" + "_tol_" + str(tol) + "_maxiter_2000_n_goals_" + str(n_goals) + \
                "_n_init_1_zero_init_True_full_results.p"

    data = pickle.load(open(robot_file, "rb"))

    robot_limits_file = "results/lwa4d_bounded_" + "True" + "_tol_" + str(tol) + "_maxiter_2000_n_goals_" + str(n_goals) + \
                "_n_init_1_zero_init_True_full_results.p"

    data_limits = pickle.load(open(robot_limits_file, "rb"))

    riem_results_list = [data[data["Solver"] == "Riemannian TrustRegions"]["Runtime"],
                          data_limits[data_limits["Solver"] == "Riemannian TrustRegions"]["Runtime"],]
    riem_bs_results_list = [data[data["Solver"] == "Riemannian TrustRegions + BS"]["Runtime"],
                         data_limits[data_limits["Solver"] == "Riemannian TrustRegions + BS"]["Runtime"],]
    # riem_results = pd.concat(riem_results_list + riem_bs_results_list)
    riem_results = pd.concat(riem_results_list)

    local_results_list = [data[data["Solver"] == "trust-exact"]["Runtime"],
                           data_limits[data_limits["Solver"] == "trust-constr"]["Runtime"],]
    local_results = pd.concat(local_results_list)
    # ### Print planar results
    # print("FABRIK results: ")
    # print(fabrik_results.describe())

    print("Local results: ")
    print(local_results.describe())

    print("Riem. TR (+ BS) results: ")
    print(riem_results.describe())
