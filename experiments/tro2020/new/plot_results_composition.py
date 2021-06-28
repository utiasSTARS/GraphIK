import os, sys, argparse
import tikzplotlib
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc
from matplotlib.patches import PathPatch
from mpl_toolkits import mplot3d

# PLOT_DIR = "/home/filipmrc/Documents/Latex/2021-giamou-semidefinite-rss/figures/results/experiment_1/"


def adjust_box_widths(axes, fac):
    """
    Adjust the widths of a seaborn-generated boxplot.
    """

    # iterating through Axes instances
    for ax in axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5 * (xmin + xmax)
                xhalf = 0.5 * (xmax - xmin)

                # setting new width of box
                xmin_new = xmid - fac * xhalf
                xmax_new = xmid + fac * xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])


def change_width(ax, new_value):
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * 0.5)


POS_TOL = 1e-2
ROT_TOL = 1e-2
COL_TOL = 1e-2


def get_success_rate(
    solution_data: pd.DataFrame,
    true_feasible: pd.Series,
    collision_data=None,
    pos_tol: float = POS_TOL,
    rot_tol: float = ROT_TOL,
    col_tol=COL_TOL,
):
    if collision_data is not None:
        collision = collision_data.abs()
        collision = collision.groupby(collision.index).max()
        collision = collision.reindex(
            list(range(solution_data.index.min(), solution_data.index.max() + 1)),
            fill_value=0.00001,
        )
    else:
        # collision = False
        collision = 0  # NOTE WERE LOOKING AT CONSTRAINT VIOLATIONS

    collision_out = collision
    collision = collision > col_tol
    reached_goal_pos = solution_data["Pos. Error"] < pos_tol
    reached_goal_rot = solution_data["Rot. Error"] < rot_tol
    feas = solution_data["True Feasible"] == True

    # solutions to feasible problems that reach the goal AND don't collide
    success_solution_data = solution_data[
        # (reached_goal_pos) & (reached_goal_rot) & (true_feasible) & (~collision)
        (reached_goal_pos) & (reached_goal_rot) & (feas) & (~collision)
    ]

    # solutions to feasible problems that don't reach the goal OR collide
    # fail_solution_data = solution_data[
    #     (true_feasible) & ((~reached_goal_pos) | (~reached_goal_rot) | (collision))
    # ]

    success = len(success_solution_data) / len(solution_data)

    feasiblity = len(true_feasible) / len(solution_data)

    return (
        success_solution_data,
        0,
        # fail_solution_data,
        success,
        feasiblity,
        collision_out,
    )


if __name__ == "__main__":
    sns.set_theme()
    palette = sns.color_palette("deep", 8)
    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--robots",
        nargs="*",
        type=str,
        default=["ur10", "kuka", "schunk"],
    )
    CLI.add_argument(
        "--methods",
        nargs="*",
        type=str,
        default=["joint", "RI"],
    )
    CLI.add_argument(
        "--envs",
        nargs="*",
        type=str,
        default=["", "octahedron", "cube", "icosahedron"],
    )
    args = CLI.parse_args()
    robots = args.robots
    methods = args.methods
    envs = args.envs

    # data = {}
    # problem_data = {}
    solution_data = {}
    # cvl_data = {}
    sucess_rates = []
    success_solution_data = {}
    # fail_solution_data = {}
    names = []
    feas = []
    for env in envs:
        for method in methods:
            for robot in robots:
                print(robot + " " + method + " " + env + ":")
                name = robot + "_" + method + "_" + env
                names += [name]

                f = open(
                    os.path.dirname(os.path.realpath(__file__))
                    + "/results/"
                    + name
                    + ".p",
                    "rb",
                )
                data = pickle.load(f)
                problem_data = data["Problem"]
                solution_data[name] = data["Solution"]
                cvl_data = data["Constraint Violations"]

                solution_data[name]["Robot"] = robot
                solution_data[name]["Method"] = method
                solution_data[name]["Environment"] = env

                true_feasible = problem_data["True Feasible"][
                    problem_data["True Feasible"] == True
                ].copy()
                solution_data[name]["True Feasible"] = problem_data["True Feasible"]

                if "value" in cvl_data:
                    col_data = cvl_data["value"].copy()
                else:
                    col_data = None

                del problem_data
                del cvl_data
                del data
                # success rate
                (
                    success_solution_data[name],
                    _,
                    # fail_solution_data[name],
                    success,
                    feasibility,
                    collision,
                ) = get_success_rate(
                    solution_data[name], true_feasible, collision_data=col_data
                )

                solution_data[name]["Max. Viol."] = collision
                solution_data[name]["Success"] = success
                feas += [feasibility]

                print("The success rate is: " + str(100 * success) + "%")
                print(
                    "The lowest possible percentage of feasible problems is: "
                    + str(100 * feasibility)
                    + "%"
                )
                print("---------------------------------------------------")
                f.close()

    all_sol_concat = pd.concat(
        list(solution_data.values())
    )  # concatenate all solution data

    # colerr = sns.catplot(
    #     y="Max. Viol.",
    #     x="Robot",
    #     col="Environment",
    #     kind="box",
    #     sharey=True,
    #     data=all_sol_concat[
    #         # ((all_sol_concat["Max. Viol."] > 0) | (all_sol_concat["Environment"] == ""))
    #         # & (all_sol_concat["Sol. Feasible"] == True)
    #         (all_sol_concat["True Feasible"] == True)
    #     ],
    #     palette=palette,
    #     hue="Method",
    #     legend_out=False,
    #     showfliers=False,
    # )

    # adjust_box_widths(colerr.axes[0], 0.7)
    # for ax in colerr.axes[0]:
    #     ax.axhline(COL_TOL, ls="--", c="black")
    #     ax.set(title="")
    #     ax.set(yscale="log")
    #     ax.tick_params(left=False)
    #     ax.set(xlabel="")

    # tikzplotlib.save(
    #     PLOT_DIR + "exp1_colerr" + ".tex", figure=colerr.fig, textsize=14.0
    # )
    # plt.show()

    srate = sns.catplot(
        data=all_sol_concat,
        kind="bar",
        x="Robot",
        y="Success",
        col="Environment",
        hue="Method",
        palette=[palette[0], palette[1], palette[7]],
        hue_order=["joint", "RI"],
        legend_out=False,
        edgecolor="black",
        # height=1,
    )
    srate.axes[0, 0].axhline(feas[0], xmin=0.050, xmax=0.3, ls="--", c="black")
    srate.axes[0, 0].axhline(feas[1], xmin=0.385, xmax=0.635, ls="--", c="black")
    srate.axes[0, 0].axhline(feas[2], xmin=0.720, xmax=0.960, ls="--", c="black")
    srate.axes[0, 1].axhline(feas[6], xmin=0.050, xmax=0.3, ls="--", c="black")
    srate.axes[0, 1].axhline(feas[7], xmin=0.385, xmax=0.635, ls="--", c="black")
    srate.axes[0, 1].axhline(feas[8], xmin=0.720, xmax=0.960, ls="--", c="black")
    srate.axes[0, 2].axhline(feas[12], xmin=0.050, xmax=0.3, ls="--", c="black")
    srate.axes[0, 2].axhline(feas[13], xmin=0.385, xmax=0.635, ls="--", c="black")
    srate.axes[0, 2].axhline(feas[14], xmin=0.720, xmax=0.960, ls="--", c="black")
    srate.axes[0, 3].axhline(feas[18], xmin=0.050, xmax=0.3, ls="--", c="black")
    srate.axes[0, 3].axhline(feas[19], xmin=0.385, xmax=0.635, ls="--", c="black")
    srate.axes[0, 3].axhline(feas[20], xmin=0.720, xmax=0.960, ls="--", c="black")
    for ax in srate.axes[0]:
        change_width(ax, 0.3)
        ax.set_xlabel("")
        # ax.set_title("")

    # tikzplotlib.save(PLOT_DIR + "exp1_srate" + ".tex", figure=srate.fig, textsize=14.0)
    plt.show()

    sol_concat = pd.concat(
        list(success_solution_data.values())
    )  # concatenate solution data

    # print(all_sol_concat[all_sol_concat["Environment"] == "icosahedron"])
    poserr = sns.catplot(
        y="Pos. Error",
        x="Robot",
        col="Environment",
        kind="box",
        sharey=True,
        data=all_sol_concat[all_sol_concat["True Feasible"] == True],
        # data=all_sol_concat,
        palette=palette,
        hue="Method",
        hue_order=["joint", "RI"],
        legend_out=False,
        showfliers=False,
    )
    # adjust_box_widths(poserr.axes[0], 0.7)
    # for ax in poserr.axes[0]:
    #     ax.axhline(POS_TOL, ls="--", c="black")
    #     ax.set(title="")
    #     ax.set(yscale="log")
    #     ax.tick_params(left=False)
    #     ax.set(xlabel="")

    # tikzplotlib.save(
    #     PLOT_DIR + "exp1_poserr" + ".tex", figure=poserr.fig, textsize=14.0
    # )
    plt.show()

    roterr = sns.catplot(
        y="Rot. Error",
        x="Robot",
        col="Environment",
        kind="box",
        sharey=True,
        data=all_sol_concat[all_sol_concat["True Feasible"] == True],
        palette=palette,
        hue="Method",
        showfliers=False,
        legend_out=False,
    )

    adjust_box_widths(roterr.axes[0], 0.7)
    for ax in roterr.axes[0]:
        # print(ax.get_yticklabels())
        # ax.set_yscale("log")
        ax.axhline(ROT_TOL, ls="--", c="black")
        # ax.set_ylabel("Rot. Error")
        ax.set(title="")
        ax.set(yscale="log")
        ax.tick_params(left=False)
        ax.set(xlabel="")

    # tikzplotlib.save(
    #     PLOT_DIR + "exp1_roterr" + ".tex", figure=roterr.fig, textsize=14.0
    # )
    # plt.show()

    time = sns.catplot(
        y="Sol. Time",
        x="Robot",
        col="Environment",
        kind="box",
        sharey=True,
        # data=all_sol_concat[all_sol_concat["Sol. Feasible"] == True],
        data=all_sol_concat,
        palette=palette,
        hue="Method",
        showfliers=False,
        legend_out=False,
    )

    adjust_box_widths(time.axes[0], 0.7)
    for ax in time.axes[0]:
        # ax.set_ylabel("Sol. Time")
        ax.set_xlabel("")
        ax.set_title("")

    # tikzplotlib.save(PLOT_DIR + "exp1_soltime" + ".tex", figure=time.fig, textsize=14.0)
    plt.show()
