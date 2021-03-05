from copy import deepcopy
from graphik.graphs.graph_base import (
    RobotGraph,
)
from graphik.graphs import RobotRevoluteGraph
import networkx as nx
import numpy as np
import sympy as sp
import numpy.linalg as la
from numpy import pi
from graphik.utils.utils import norm_sq
from graphik.utils.dgp import pos_from_graph
from graphik.robots import RobotRevolute


def apply_angular_offset_2d(joint_angle_offset, z0):
    """
    :param joint_angle_offset:
    :param z0:
    :return:
    """
    R = np.zeros((2, 2))
    R[0, 0] = np.cos(joint_angle_offset)
    R[0, 1] = np.sin(joint_angle_offset)
    R[1, 0] = -np.sin(joint_angle_offset)
    R[1, 1] = R[0, 0]
    return R * z0


def archimedean_constraint(variables, M_bounds):
    return [var ** 2 <= M for var, M in zip(variables, M_bounds)]


def all_symbols(constraints):
    all_vars = set()
    for cons in constraints:
        all_vars = all_vars.union(set(cons.free_symbols))
    return list(all_vars)


def generate_symbols(
    robot_graph: RobotGraph, nodes_with_angular_vars: dict = None
) -> nx.DiGraph:
    """
    Make a copy of robot_graph's such that the nodes have "sym" fields populated.
    Also populate the angular_variables field if angular residual constraint variables are needed.

    :param robot_graph:
    :param nodes_with_angular_vars: dictionary whose keys are the nodes we want to have angular vars
    :return:
    """
    G = deepcopy(robot_graph.directed)  # Make a copy of the problem graph

    # Assign symbolic variables where needed
    for node_id in G:
        if robot_graph.dim == 3:
            G.nodes[node_id]["sym"] = np.array(
                sp.symbols(f"{node_id}_x, {node_id}_y, {node_id}_z")
            )
        elif robot_graph.dim == 2:
            G.nodes[node_id]["sym"] = np.array(sp.symbols(f"{node_id}_x, {node_id}_y"))
        if node_id in list(robot_graph.base.nodes()):
            G.nodes[node_id]["root"] = True

    if nodes_with_angular_vars is not None:
        G.angular_variables = {}
        for node_id in nodes_with_angular_vars:
            G.angular_variables[node_id] = sp.symbols("s_" + node_id)
        # G.angular_variables = np.array(sp.symbols("s:"+str(n_angular)))
    # else:
    #     angular_variables =
    return G


def constraints_from_graph(
    robot_graph: RobotGraph, end_effector_assignments: dict, options: dict = None
) -> list:
    """

    :param robot_graph:
    :param end_effector_assignments:
    :param options:
    :return:
    """
    G = generate_symbols(robot_graph)
    # Anchor end effectors
    for node_id in end_effector_assignments:
        G.nodes[node_id]["pos"] = end_effector_assignments[node_id]

    constraints = []

    # define structural constraints (this excludes end-effector distances)
    for u, v, weight in G.edges.data("weight"):
        if weight:
            # Ugly, but we don't need these (TODO: Porta case check?)
            if (
                u not in ("x", "y")
                and v not in ("x", "y")
                and set((u, v)) != set(("p0", "q0"))
                and ("bounded" not in G[u][v] or (G[u][v]["bounded"]))
            ):
                p_u = (
                    G.nodes[u]["pos"]
                    if u in end_effector_assignments or u in ("p0", "q0")
                    else G.nodes[u]["sym"]
                )
                p_v = (
                    G.nodes[v]["pos"]
                    if v in end_effector_assignments or v in ("p0", "q0")
                    else G.nodes[v]["sym"]
                )
                equality = sp.Eq(norm_sq(p_u - p_v), weight ** 2)
                if (
                    type(equality) == sp.Equality
                ):  # This will be a boolean expression if both values are set
                    constraints.append(equality)

    return constraints


def angular_constraints(
    robot_graph: RobotGraph,
    angular_limits: dict,
    end_effector_assignments: dict,
    angular_offsets: dict = None,
    as_equality: bool = False,
) -> list:
    if as_equality:
        G = generate_symbols(robot_graph, angular_limits)
    else:
        G = generate_symbols(robot_graph)
    constraints = []
    for node in angular_limits:
        limit = angular_limits[node]
        if node in end_effector_assignments:
            x2 = end_effector_assignments[node]
        elif node in robot_graph.base.nodes:
            x2 = G.nodes[node]["pos"]
        else:
            x2 = G.nodes[node]["sym"]
        parent = robot_graph.parent_node_id(node)
        if parent in end_effector_assignments:
            x1 = end_effector_assignments[parent]
        elif parent in robot_graph.base.nodes:
            x1 = G.nodes[parent]["pos"]
        else:
            x1 = G.nodes[parent]["sym"]

        # TODO: devise a better system for this? p0 is no longer the root of the DiGraph (it's not a tree anymore)
        if parent == "p0":
            x0 = np.zeros((robot_graph.dim,))
            x0[0] = 1.0  # np.array([1., 0., 0.])
            x0 = G.nodes[parent]["pos"] - x0
        else:
            grandparent = robot_graph.parent_node_id(parent)
            if grandparent in end_effector_assignments:
                x0 = end_effector_assignments[grandparent]
            elif grandparent in robot_graph.base.nodes:
                x0 = G.nodes[grandparent]["pos"]
            else:
                x0 = G.nodes[grandparent]["sym"]

        l1 = 1.0 if parent == "p0" else G.edges[grandparent, parent]["weight"]
        l2 = G.edges[parent, node]["weight"]
        z0 = (x1 - x0) / l1
        z1 = (x2 - x1) / l2

        if angular_offsets is not None:
            z0 = apply_angular_offset_2d(angular_offsets[node], z0)

        if as_equality:
            constraints.append(
                sp.Eq(
                    norm_sq(z1 - z0) + G.angular_variables[node] ** 2,
                    2.0 * (1.0 - np.cos(limit)),
                )
            )
        else:
            constraints.append(norm_sq(z1 - z0) <= 2.0 * (1.0 - np.cos(limit)))

    return constraints


def nearest_neighbour_cost(
    robot_graph: RobotGraph,
    nearest_neighbour_points: dict,
    nearest_angular_residuals: dict = None,
):
    """
    Produce a nearest-neighbour cost function.
    TODO: add assertions to help with checking this. For graph case, need a (6,) ndarray with p before q.

    :param robot_graph: Graph object describing our robot
    :param nearest_neighbour_points: dictionary mapping node ids (strings) to ndarrays
    :param nearest_angular_residuals: list of angular residuals (floats)
    :return: symbolic expression of cost
    """
    if nearest_angular_residuals is None:
        G = generate_symbols(robot_graph)  # augment the nodes with a symbol attribute
    else:
        G = generate_symbols(robot_graph, nearest_angular_residuals)
    cost = 0.0

    for node_id in nearest_neighbour_points:
        cost += norm_sq(G.nodes[node_id]["sym"] - nearest_neighbour_points[node_id])

    if nearest_angular_residuals is not None:
        for node_id in nearest_angular_residuals:
            cost += (
                G.angular_variables[node_id] - nearest_angular_residuals[node_id]
            ) ** 2
    # if self.as_equality and nearest_angular_residuals is not None:
    #     for idx, node in enumerate(self.nodes[1:]):
    #         cost += (node.angle_variable - nearest_angular_residuals[idx]) ** 2
    return cost


def get_full_revolute_nearest_points_pose(graph, q):
    full_points = [f"p{idx}" for idx in range(1, graph.robot.n)] + [
        f"q{idx}" for idx in range(1, graph.robot.n)
    ]
    return get_full_revolute_nearest_point(graph, q, full_points)


def get_full_revolute_nearest_point(graph, q, full_points=None):
    nearest_points = {}
    if full_points is None:
        # Assume only x, y, p0, q0, and the final p (p{n+1}) is constrained
        full_points = [f"p{idx}" for idx in range(1, graph.robot.n)] + [
            f"q{idx}" for idx in range(1, graph.robot.n + 1)
        ]
    G = graph.realization(q)
    P = pos_from_graph(G)
    for idx, key in enumerate(graph.node_ids):
        if key in full_points:
            nearest_points[key] = P[idx, :]
    return nearest_points


if __name__ == "__main__":

    # n = 5

    # a = np.ones(n)
    # th = np.zeros(n)

    # params = {"a": a, "theta": th}

    # robot = RobotPlanar(params)
    # graph = PlanarRobotGraph(robot)
    # constraints = constraints_from_graph(graph)

    n = 5
    # Generate random DH parameters
    a = np.random.rand(n)
    d = np.random.rand(n + 1)
    al = np.random.rand(n) * pi / 2 - 2 * np.random.rand(n) * pi / 2
    th = 0 * np.ones(n + 1)
    ub = np.concatenate([np.ones(n) * pi - 2 * pi * np.random.rand(n), [0]])
    lb = -ub

    params = {"a": a, "alpha": al, "d": d, "theta": th, "lb": lb, "ub": ub}
    robot = RobotRevolute(params)  # instantiate robot
    end_effector_assignments = {"p6": np.array([1.0, 1.0, 1.0])}
    graph = RobotRevoluteGraph(robot)
    constraints = constraints_from_graph(graph, end_effector_assignments)
    print(graph.directed.edges())
    print(constraints)
