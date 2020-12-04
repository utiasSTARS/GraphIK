from copy import deepcopy
from graphik.graphs.graph_base import Graph, SphericalRobotGraph, Revolute3dRobotGraph
import networkx as nx
import numpy as np
import sympy as sp
import numpy.linalg as la
from numpy import pi
from graphik.utils.utils import norm_sq
from graphik.robots.robot_base import RobotRevolute


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
    robot_graph: Graph, nodes_with_angular_vars: dict = None
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
    robot_graph: Graph, end_effector_assignments: dict, options: dict = None
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
                constraints.append(sp.Eq(norm_sq(p_u - p_v), weight ** 2))

    return constraints


def angular_constraints(
    robot_graph: Graph,
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


def edm_angular_constraints(node0, node1, node2, as_equality=False) -> list:
    """
    TODO: as_equality? Not sure it's needed yet
    TODO: angular offset? Not sure it's possible or easy
    :param node0:
    :param node1:
    :param node2:
    :param as_equality:
    :return:
    """
    if node0 is None:
        assert node1.is_root(), "Node1 needs to be root for node0 to be None."
        v0 = np.array([-1.0, 0.0, 0.0])
    if node0.value is not None:
        v0 = node0["pos"]
    else:
        v0 = node0["sym"]
    if node2.value is not None:
        v2 = node2["pos"]
    else:
        v2 = node2["sym"]
    l1 = node1.link_length
    l2 = node2.link_length
    constraints = []
    angle_max = node2.angular_limit
    if not as_equality:
        constraints.append(
            norm_sq(v2 - v0) >= l1 ** 2 + l2 ** 2 + 2.0 * l1 * l2 * np.cos(angle_max)
        )
        constraints.append(norm_sq(v2 - v0) <= (l1 + l2) ** 2)
    else:
        constraints.append(
            sp.Eq(
                norm_sq(v2 - v0) - node2.angle_variable ** 2,
                l1 ** 2 + l2 ** 2 + 2.0 * l2 * l2 * np.cos(angle_max),
            )
        )
        constraints.append(
            sp.Eq(norm_sq(v2 - v0) + node2.angle_variable ** 2, (l1 + l2) ** 2)
        )
    return constraints


def convex_angular_constraints(node0, node1, node2, as_equality=False):
    """
            2
           /
          /
    0----1

    :param node0:
    :param node1:
    :param node2:
    :param as_equality:
    :return:
    """
    if node0 is None:
        assert node1.is_root, "Node1 needs to be root for node0 to be None."
        v_base = np.zeros((node2.dim,))
        v_base[0] = 1.0  # np.array([1., 0., 0.])
        v0 = node1.value - v_base
    elif node0.value is not None:
        v0 = node0.value
    else:
        v0 = node0.variable
    if node1.value is not None:
        v1 = node1.value
    else:
        v1 = node1.variable
    if node2.value is not None:
        v2 = node2.value
    else:
        v2 = node2.variable
    l1 = node1.link_length
    l2 = node2.link_length
    z0 = (v1 - v0) / l1
    z1 = (v2 - v1) / l2
    if node2.angular_offset != 0:
        z0 = apply_angular_offset_2d(node2.angular_offset, z0)
    angle_max = node2.angular_limit
    if as_equality:
        constraints = [
            sp.Eq(
                norm_sq(z1 - z0) + node2.angle_variable ** 2,
                2.0 * (1.0 - np.cos(angle_max)),
            )
        ]
    else:
        constraints = [norm_sq(z1 - z0) <= 2.0 * (1.0 - np.cos(angle_max))]

    return constraints


def nearest_neighbour_cost(
    robot_graph: Graph,
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


# TODO: move get_nearest_neighbour_cost/linear cost into the Solver/QcqpToSdpRelaxation classes

# def symbolic_constraints(self) -> list:
#     """
#     Assumes all the parameters (value vs. variable) in the nodes have been set.
#     :param params: dictionary of parameters
#     :return:
#     """
#     constraints = []
#     for node in self.nodes[1:]:  # Root does not need to be handled
#         node_id = node.node_id
#         parent_node = self.parent_node(node_id)
#         if parent_node.is_root:
#             grandparent_node = None
#         else:
#             grandparent_node = self.parent_node(parent_node.node_id)

#         constraints += node.distance_constraints(parent_node)

#         if self.angle_constraints_are_convex:
#             constraints += PlanarNode.convex_angular_constraints(
#                 grandparent_node, parent_node, node, as_equality=self.as_equality
#             )
#         else:
#             constraints += PlanarNode.edm_angular_constraints(
#                 grandparent_node, parent_node, node, as_equality=self.as_equality
#             )

#     return constraints


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
    graph = Revolute3dRobotGraph(robot)
    constraints = constraints_from_graph(graph, end_effector_assignments)
    print(graph.directed.edges())
    print(constraints)
