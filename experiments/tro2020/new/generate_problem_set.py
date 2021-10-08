import time
import sys
import os
import argparse
import numpy as np
import pandas as pd
import pickle
from progress.bar import ShadyBar as Bar

from graphik.graphs import ProblemGraphRevolute
from graphik.utils import *
from graphik.utils.roboturdf import load_ur10, load_kuka, load_schunk_lwa4d
from numpy import pi

def generate_revolute_problem(graph: ProblemGraphRevolute, obstacles = False):

    robot = graph.robot
    n = robot.n
    axis_len = robot.axis_length

    q_goal = graph.robot.random_configuration()
    G_goal = graph.realization(q_goal)

    T_goal = robot.get_pose(q_goal, f"p{n}")

    goals = {
        f"p{n}": T_goal.trans,
        f"q{n}": T_goal.dot(trans_axis(axis_len, "z")).trans,
    }

    G_partial = graph.from_pos(goals)

    true_feasibility = False
    if len(graph.check_distance_limits(G_goal, tol=1e-10)) == 0:
        true_feasibility = True
    return T_goal, G_partial, true_feasibility

if __name__ == "__main__":

    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--robot",
        nargs="*",
        type=str,
        default=["ur10", "kuka", "schunk"],
    )
    CLI.add_argument(
        "--num_prob",
        nargs="*",
        type=int,
        default=[100],
    )

    CLI.add_argument(
        "--env",
        nargs="*",
        type=str,
        default=[""],
    )
    args = CLI.parse_args()
    robots = args.robot
    num_prob = args.num_prob[0]
    envs = args.env
    np.random.seed(21)

    for env_name in envs:
        obstacles = []
        if env_name == "icosahedron":
            phi = (1 + np.sqrt(5)) / 2
            scale = 0.5
            radius = 0.5
            obstacles = [
                (scale * np.asarray([0, 1, phi]), radius),
                (scale * np.asarray([0, 1, -phi]), radius),
                (scale * np.asarray([0, -1, -phi]), radius),
                (scale * np.asarray([0, -1, phi]), radius),
                (scale * np.asarray([1, phi, 0]), radius),
                (scale * np.asarray([1, -phi, 0]), radius),
                (scale * np.asarray([-1, -phi, 0]), radius),
                (scale * np.asarray([-1, phi, 0]), radius),
                (scale * np.asarray([phi, 0, 1]), radius),
                (scale * np.asarray([-phi, 0, 1]), radius),
                (scale * np.asarray([-phi, 0, -1]), radius),
                (scale * np.asarray([phi, 0, -1]), radius),
            ]

        if env_name == "cube":
            scale = 0.5
            radius = 0.45
            obstacles = [
                (scale * np.asarray([-1, 1, 1]), radius),
                (scale * np.asarray([-1, 1, -1]), radius),
                (scale * np.asarray([1, -1, 1]), radius),
                (scale * np.asarray([1, -1, -1]), radius),
                (scale * np.asarray([1, 1, 1]), radius),
                (scale * np.asarray([1, 1, -1]), radius),
                (scale * np.asarray([-1, -1, 1]), radius),
                (scale * np.asarray([-1, -1, -1]), radius),
            ]
        if env_name == "octahedron":
            scale = 0.75
            radius = 0.4
            obstacles = [
                (scale * np.asarray([1, 1, 0]), radius),
                (scale * np.asarray([1, -1, 0]), radius),
                (scale * np.asarray([-1, 1, 0]), radius),
                (scale * np.asarray([-1, -1, 0]), radius),
                (scale * np.asarray([0, 0, 1]), radius),
                (scale * np.asarray([0, 0, -1]), radius),
            ]
        for robot_name in robots:
            if robot_name == "ur10":
                robot, graph = load_ur10(limits=None)
            elif robot_name == "kuka":
                robot, graph = load_kuka(limits=None)
            elif robot_name == "schunk":
                robot, graph = load_schunk_lwa4d(limits=None)
            else:
                raise NotImplementedError

            for idx, obs in enumerate(obstacles):
                graph.add_spherical_obstacle(f"o{idx}", obs[0], obs[1])
            problems = []
            for _ in range(num_prob):
                T_goal, G_partial, feas = generate_revolute_problem(graph, obstacles=True)
                problems += [
                    {
                        "T_goal": T_goal,
                        "G_partial": G_partial,
                        "Feasibility": feas,
                        "Robot": robot_name,
                        "Environment": env_name,
                    }
                ]

            save_data = pd.DataFrame(problems)
            storage_base_path = os.path.dirname(os.path.realpath(__file__)) + "/generated_problems/"
            path = storage_base_path + robot_name + "_" + env_name + ".p"

            if not os.path.exists(storage_base_path):
                os.makedirs(storage_base_path)

            with open(path, "wb") as f:
                pickle.dump(save_data, f)
