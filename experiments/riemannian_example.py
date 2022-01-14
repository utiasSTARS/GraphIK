from graphik.utils.utils import table_environment
from graphik.solvers.riemannian_solver import solve_with_riemannian

# Multiple robot models to try out, or you can implement your own
from graphik.utils.roboturdf import load_ur10


if __name__ == "__main__":
    # Load an example robot
    robot, graph = load_ur10()

    # Load an example obstacle environment, or construct your own (see implementation of table_environment())
    obstacles = table_environment()

    # Initialize the graph object with obstacles from the chosen environment
    for idx, obs in enumerate(obstacles):
        graph.add_spherical_obstacle(f"o{idx}", obs[0], obs[1])

    # Generate anchor nodes representing a pose goal the end-effector
    q_goal = robot.random_configuration()
    T_goal = robot.pose(q_goal, f"p{robot.n}")  # Can be any desired pose, this is just a simple example

    # Run Riemannian solver
    q_sol, solution_points = solve_with_riemannian(graph, T_goal)  # Returns None if infeasible or didn't solve

    # Compare the solution's end effector pose to the goal.
    # Don't be surprised if the configurations are different, even for the UR10!
    # Each pose has up to 16 unique solutions for 6-DOF manipulators.
    print("Target pose: ")
    print(T_goal)
    print("Target configuration: ")
    print(q_goal)
    print("--------------------------------------------")
    if q_sol:
        print("Riemannian solution's pose: ")
        print(robot.pose(q_sol, f"p{robot.n}"))
        print("Riemannian configuration: ")
        print(q_sol)
    else:
        print("Riemannian did not return a feasible solution.")
