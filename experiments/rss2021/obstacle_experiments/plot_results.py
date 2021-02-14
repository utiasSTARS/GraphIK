import os, sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc
from mpl_toolkits import mplot3d

if __name__ == "__main__":
    # data = pickle.load(
    #     open(
    #         os.path.dirname(os.path.realpath(__file__))
    #         + "/results/"
    #         + str(sys.argv[1]),
    #         "rb",
    #     )
    # )
    #
    rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 20})
    rc("text", usetex=True)
    boxprops = dict(linestyle="-", linewidth=2)
    medianprops = dict(linestyle="-", linewidth=2)
    files = [
        # "kuka_20210212-224415.p",
        # "lwa4d_20210212-223157.p",
        # "ur10_20210212-221855.p",
        # "kuka_20210213-005344_no_obs.p",
        # "lwa4d_20210213-005117_no_obs.p",
        # "ur10_20210213-000822_no_obs.p",
        "kuka_exp_obs.p",
        "lwa4d_exp_obs.p",
        "ur10_exp_obs.p",
    ]
    name = ["kuka", "schunk", "UR10"]
    data = {}
    for idx, f in enumerate(files):
        data[name[idx]] = pickle.load(
            open(
                os.path.dirname(os.path.realpath(__file__)) + "/results/" + f,
                "rb",
            )
        )

    problem_kuka = data["kuka"]["Problem"]
    solution_kuka = data["kuka"]["Solution"]
    cvl_kuka = data["kuka"]["Constraint Violations"]

    problem_UR10 = data["UR10"]["Problem"]
    solution_UR10 = data["UR10"]["Solution"]
    cvl_UR10 = data["UR10"]["Constraint Violations"]

    problem_schunk = data["schunk"]["Problem"]
    solution_schunk = data["schunk"]["Solution"]
    cvl_schunk = data["schunk"]["Constraint Violations"]

    rot_error = pd.DataFrame(
        [
            solution_kuka["Rot. Error"],
            solution_schunk["Rot. Error"],
            solution_UR10["Rot. Error"],
        ],
        index=name,
    ).transpose()

    ax_rot = np.log10(rot_error[:200]).plot.box(
        title="Rot. Error",
        grid=True,
        showfliers=False,
        boxprops=dict(linestyle="-", linewidth=2),
        flierprops=dict(linestyle="-", linewidth=2),
        medianprops=dict(linestyle="-", linewidth=2),
        whiskerprops=dict(linestyle="-", linewidth=2),
        capprops=dict(linestyle="-", linewidth=2),
    )
    ax_rot.set_ylabel("log$_{10}$ Rot. Error [rad]")

    pos_error = pd.DataFrame(
        [
            solution_kuka["Pos. Error"],
            solution_schunk["Pos. Error"],
            solution_UR10["Pos. Error"],
        ],
        index=name,
    ).transpose()

    ax_pos = np.log10(pos_error[:200]).plot.box(
        title="Pos. Error",
        grid=True,
        showfliers=False,
        boxprops=dict(linestyle="-", linewidth=2),
        flierprops=dict(linestyle="-", linewidth=2),
        medianprops=dict(linestyle="-", linewidth=2),
        whiskerprops=dict(linestyle="-", linewidth=2),
        capprops=dict(linestyle="-", linewidth=2),
    )
    ax_pos.set_ylabel("log$_{10}$ Pos. Error [m]")

    sol_time = pd.DataFrame(
        [
            solution_kuka["Fantope Time"] + solution_kuka["Primal Time"],
            solution_schunk["Fantope Time"] + solution_schunk["Primal Time"],
            solution_UR10["Fantope Time"] + solution_UR10["Primal Time"],
        ],
        index=name,
    ).transpose()

    ax_t = sol_time[:200].plot.box(
        title="Sol. Time",
        grid=True,
        showfliers=False,
        boxprops=dict(linestyle="-", linewidth=2),
        flierprops=dict(linestyle="-", linewidth=2),
        medianprops=dict(linestyle="-", linewidth=2),
        whiskerprops=dict(linestyle="-", linewidth=2),
        capprops=dict(linestyle="-", linewidth=2),
    )
    ax_t.set_ylabel("Sol. Time [s]")

    plt.show()

    # success rate
    print(
        len(
            solution_kuka["Pos. Error"][
                (solution_kuka["Pos. Error"] > 0.01)
                & (solution_kuka["Rot. Error"] > 0.01)
            ]
        )
    )
    print(
        len(
            solution_schunk["Pos. Error"][
                (solution_schunk["Pos. Error"] > 0.01)
                & (solution_schunk["Rot. Error"] > 0.01)
            ]
        )
    )
    print(
        len(
            solution_UR10["Pos. Error"][
                (solution_UR10["Pos. Error"] > 0.01)
                & (solution_UR10["Rot. Error"] > 0.01)
            ]
        )
    )
    # print(len(solution_schunk["Pos. Error"][solution_schunk["Pos. Error"] > 0.01]))
    # print(len(solution_ur10["Pos. Error"][solution_ur10["Pos. Error"] > 0.01]))
