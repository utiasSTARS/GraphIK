import numpy as np
import networkx as nx
import time

from numpy.linalg.linalg import norm
from numpy import pi
from graphik.solvers.convex_iteration import (
    convex_iterate_sdp_snl_graph,
)
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from graphik.graphs.graph_base import RobotRevoluteGraph
from graphik.utils.utils import safe_arccos
from graphik.utils.dgp import graph_from_pos_dict, pos_from_graph
from graphik.utils.constants import *
from graphik.solvers.sdp_snl import (
    extract_solution,
)
from graphik.solvers.constraints import get_full_revolute_nearest_point
from graphik.utils.roboturdf import load_ur10


def plot_robot(G: nx.DiGraph, pos: dict):

    ax = plt.gca(projection="3d")
    ax.cla()

    typ = nx.get_node_attributes(G, name=TYPE)
    edg = nx.get_edge_attributes(G, name=DIST)

    # 3D network plot
    for u, v in edg.keys():
        if typ[u] == "robot" and typ[v] == "robot":

            # plot nodes
            pos_u = pos[u]
            pos_v = pos[v]
            ax.scatter(pos_u[0], pos_u[1], pos_u[2], c="black")
            ax.scatter(pos_v[0], pos_v[1], pos_v[2], c="black")

            # plot edges
            x = np.array((pos_u[0], pos_v[0]))
            y = np.array((pos_u[1], pos_v[1]))
            z = np.array((pos_u[2], pos_v[2]))

            ax.plot(x, y, z, c="black", alpha=0.5)

    for node in G:
        if typ[node] == "obstacle":
            center, radius = G.nodes[node][POS], G["p1"][node][LOWER]
            u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
            x = radius * np.cos(u) * np.sin(v) + center[0]
            y = radius * np.sin(u) * np.sin(v) + center[1]
            z = radius * np.cos(v) + center[2]
            ax.plot_wireframe(x, y, z, color="r")

    ax.set_xlim((-3, 3))
    ax.set_ylim((-3, 3))
    ax.set_zlim((-3, 3))


# def check_collision(graph: RobotRevoluteGraph, sol: nx.DiGraph):
#     typ = nx.get_node_attributes(graph.directed, name=TYPE)
#     for node in graph.directed:
#         if typ[node] == "obstacle":
#             center = graph.directed.nodes[node][POS]
#             radius = graph.directed["p1"][node][LOWER]

#             for pnt in sol:
#                 if typ[pnt] == "robot" and pnt[0] == "p":
#                     pos = sol.nodes[pnt][POS]
#                     if (pos - center) @ np.identity(len(center)) @ (
#                         pos - center
#                     ).T <= radius ** 2:
#                         return True
#     return False


def solve_random_problem(graph: RobotRevoluteGraph):
    robot = graph.robot
    n = robot.n
    t_sol = 0

    q_goal = robot.random_configuration()
    G_goal = graph.realization(q_goal)
    T_goal = robot.get_pose(q_goal, f"p{n}")

    full_points = [f"p{idx}" for idx in range(0, n + 1)] + [
        f"q{idx}" for idx in range(0, n + 1)
    ]

    input_vals = get_full_revolute_nearest_point(graph, q_goal, full_points)
    anchors = {key: input_vals[key] for key in ["p0", "q0", f"p{n}", f"q{n}"]}

    t_sol = time.perf_counter()
    (
        _,
        constraint_clique_dict,
        sdp_variable_map,
        _,
        _,
        _,
    ) = convex_iterate_sdp_snl_graph(graph, anchors, ranges=True)
    t_sol = time.perf_counter() - t_sol

    solution = extract_solution(constraint_clique_dict, sdp_variable_map, robot.dim)

    # expand solution to include all the points
    for node in G_goal:
        if node not in solution.keys():
            solution[node] = G_goal.nodes[node][POS]

    G_sol = graph_from_pos_dict(solution)

    q_sol = robot.joint_variables(G_sol, {f"p{n}": T_goal})
    T_riemannian = robot.get_pose(q_sol, f"p{n}")

    err_riemannian_pos = norm(T_goal.trans - T_riemannian.trans)
    z_goal = T_goal.as_matrix()[:3, 2]
    z_riemannian = T_riemannian.as_matrix()[:3, 2]
    err_riemannian_rot = abs(safe_arccos(z_riemannian.dot(z_goal)))

    # check for collisons
    col = False
    if len(graph.check_collision(G_sol)) > 0:
        col = True

    not_reach = False
    if err_riemannian_pos > 0.01 or err_riemannian_rot > 0.01:
        not_reach = True

    broken_limits = {}
    limit_violations = False
    for key in robot.limited_joints:
        if abs(q_sol[key]) > (graph.robot.ub[key] * 1.01):
            limit_violations = True
            broken_limits[key] = abs(q_sol[key]) - (graph.robot.ub[key])
            print(key, broken_limits[key])

    fail = False
    if limit_violations or col or not_reach:
        print(
            col * "collision"
            # + infeas * "+ infeasible goal"
            + not_reach * "+ didn't reach"
            + limit_violations * "+ limit violations"
        )
        fail = True
    print(
        f"Pos. error: {err_riemannian_pos}\nRot. error: {err_riemannian_rot}\nSolution time: {t_sol}"
    )
    print("------------------------------------")
    return err_riemannian_pos, err_riemannian_rot, t_sol, fail, G_sol


if __name__ == "__main__":
    ub = np.minimum(np.random.rand(6) * (pi / 2) + pi / 2, pi)
    lb = -ub
    limits = (lb, ub)

    robot, graph = load_ur10(limits=None)
    obstacles = [
        (np.array([0, 1, 1]), 0.75),
    ]
    for idx, obs in enumerate(obstacles):
        graph.add_spherical_obstacle(f"o{idx}", obs[0], obs[1])

    num_tests = 100
    e_pos = []
    e_rot = []
    t = []
    fails = []

    for _ in range(num_tests):
        e_r_pos, e_r_rot, t_sol, fail, G_sol = solve_random_problem(graph)
        plot_robot(graph.directed, nx.get_node_attributes(G_sol, POS))
        plt.pause(0.5)

        e_pos += [e_r_pos]
        e_rot += [e_r_rot]
        t += [t_sol]
        fails += [fail]

    t = np.array(t)
    t = t[abs(t - np.mean(t)) < 2 * np.std(t)]

    print("Average solution time {:}".format(np.average(t)))
    print("Standard deviation of solution time {:}".format(np.std(np.array(t))))
    print("Average pos error {:}".format(np.average(np.array(e_pos))))
    print("Average rot error {:}".format(np.average(np.array(e_rot))))
    print("Number of fails {:}".format(sum(fails)))
