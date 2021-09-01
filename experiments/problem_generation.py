from graphik.utils import *
from graphik.graphs.graph_revolute import RobotRevoluteGraph

def generate_revolute_problem(graph: RobotRevoluteGraph, obstacles = False):

    robot = graph.robot
    n = robot.n
    axis_len = graph.axis_length

    if obstacles:
        feasible = False
        while not feasible:
            q_goal = robot.random_configuration()
            G_goal = graph.realization(q_goal)
            T_goal = robot.pose(q_goal, f"p{n}")
            broken_limits = graph.check_distance_limits(G_goal)
            if len(broken_limits) == 0:
                feasible = True
    else:
        q_goal = graph.robot.random_configuration()

    G_goal = graph.realization(q_goal)
    # D_goal = graph.distance_matrix_from_joints(q_goal)
    T_goal = robot.pose(q_goal, f"p{n}")

    goals = {
        f"p{n}": T_goal.trans,
        f"q{n}": T_goal.dot(trans_axis(axis_len, "z")).trans,
    }

    X_goal = pos_from_graph(G_goal)
    X_goal = normalize_positions(X_goal)
    D_goal = distance_matrix_from_pos(X_goal)

    G = graph.complete_from_pos(goals)

    return G, T_goal, D_goal, X_goal
