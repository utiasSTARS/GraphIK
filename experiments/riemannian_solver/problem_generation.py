from graphik.utils.geometry import trans_axis
from graphik.utils.utils import list_to_variable_dict
from graphik.utils.dgp import pos_from_graph
from graphik.graphs.graph_revolute import RobotRevoluteGraph


def generate_revolute_problem(graph: RobotRevoluteGraph):

    robot = graph.robot
    n = robot.n
    axis_len = robot.axis_length
    q_goal = graph.robot.random_configuration()
    G_goal = graph.realization(q_goal)
    X_goal = pos_from_graph(G_goal)
    D_goal = graph.distance_matrix_from_joints(q_goal)
    T_goal = robot.get_pose(q_goal, f"p{n}")

    goals = {
        f"p{n}": T_goal.trans,
        f"q{n}": T_goal.dot(trans_axis(axis_len, "z")).trans,
    }
    G = graph.complete_from_pos(goals)
    return G, T_goal, D_goal, X_goal
