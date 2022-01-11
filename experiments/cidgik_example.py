from liegroups.numpy import SE3

from graphik.graphs.graph_revolute import ProblemGraphRevolute
from graphik.utils.utils import table_environment
from graphik.utils.constants import *
from graphik.solvers.convex_iteration import convex_iterate_sdp_snl_graph
from graphik.solvers.sdp_snl import extract_solution

# Multiple robot models to try out, or implement your own
from graphik.utils.roboturdf import load_ur10, load_kuka, load_schunk_lwa4d, load_9_dof


def solve_with_cidgik(graph: ProblemGraphRevolute, T_goal: SE3) -> (dict, dict):
    robot = graph.robot
    n = robot.n

    # Set up the anchors needed as input to CIDGIK
    anchors = {
        "p0": graph.nodes["p0"][POS],
        "q0": graph.nodes["q0"][POS],
        f"p{n}": T_goal.trans,
        f"q{n}": T_goal.trans + T_goal.rot.as_matrix()[:, 2]
    }

    # Solve with CIDGIK
    _, constraint_clique_dict, sdp_variable_map, _, _, _, _, _, feasible = \
        convex_iterate_sdp_snl_graph(
            graph,
            anchors,
            ranges=True,
            sparse=False,
            closed_form=True,
            scs=False
        )

    # Extract the angular configuration
    if feasible is FEASIBLE:
        solution = extract_solution(constraint_clique_dict, sdp_variable_map, robot.dim)

        # Add the end-effector goal points to the solution
        solution[f"p{robot.n}"] = anchors[f"p{robot.n}"]
        solution[f"q{robot.n}"] = anchors[f"q{robot.n}"]

        # Add the base points to the solution
        base_nodes = ["p0", "x", "y", "q0"]
        for node in base_nodes:
            solution[node] = graph.nodes[node][POS]
        G_sol = graph.from_pos(solution)
        q_sol = graph.joint_variables(G_sol, {f"p{n}": T_goal})

        return q_sol, solution
    else:
        return None, None


if __name__ == "__main__":
    # Load an example robot
    robot, graph = load_ur10()  # load_9_dof()

    # Load an example obstacle environment, or construct your own (see implementation of table_environment())
    obstacles = table_environment()

    # Initialize the graph object with obstacles from the chosen environment
    for idx, obs in enumerate(obstacles):
        graph.add_spherical_obstacle(f"o{idx}", obs[0], obs[1])

    # Generate anchor nodes representing a pose goal the end-effector
    q_goal = robot.random_configuration()
    T_goal = robot.pose(q_goal, f"p{robot.n}")  # Can be any desired pose, this is just a simple example

    # Run CIDGIK with the anchor nodes as a goal
    q_sol, solution_points = solve_with_cidgik(graph, T_goal)  # Returns None if infeasible or didn't solve

    # Compare the solution's end effector pose to the goal.
    # Don't be surprised if the configurations are different, even for the UR10!
    # Each pose has up to 16 unique solutions for 6-DOF manipulators.
    print("Target pose: ")
    print(T_goal)
    print("Target configuration: ")
    print(q_goal)
    print("--------------------------------------------")
    if q_sol:
        print("CIDGIK solution's pose: ")
        print(robot.pose(q_sol, f"p{robot.n}"))
        print("CIDGIK configuration: ")
        print(q_sol)
    else:
        print("CIDGIK did not return a feasible solution.")
