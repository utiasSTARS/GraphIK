import numpy as np
from graphik.solvers.sdp_snl import (
    cvxpy_inequality_constraints,
    distance_inequality_constraint,
    distance_range_constraints,
    solve_nearest_point_sdp,
    distance_constraints,
)
from graphik.utils.roboturdf import load_ur10
from graphik.solvers.sdp_formulations import SdpSolverParams
from graphik.solvers.constraints import get_full_revolute_nearest_point

if __name__ == "__main__":

    sparse = False  # Whether to exploit chordal sparsity in the SDP formulation
    ee_cost = False  # Whether to treat the end-effectors as variables with targets in the cost.
    # If False, end-effectors are NOT variables (they are baked in to constraints as parameters)
    conic_solver = "MOSEK"  # One of "MOSEK", "CVXOPT" for now
    solver_params = (
        SdpSolverParams()
    )  # Use MOSEK settings that worked well for us before

    robot, graph = load_ur10()
    n = robot.n

    # Generate a random feasible target
    q = robot.random_configuration()

    # Extract the positions of the points
    full_points = [f"p{idx}" for idx in range(0, graph.robot.n + 1)] + [
        f"q{idx}" for idx in range(0, graph.robot.n + 1)
    ]
    input_vals = get_full_revolute_nearest_point(graph, q, full_points)

    # End-effectors are 'generalized' to include the base pair ('p0', 'q0')
    end_effectors = {
        key: input_vals[key] for key in ["p0", "q0", f"p{robot.n}", f"q{robot.n}"]
    }  # NOTE not sure if using the term end-effector for the base is sensible

    # NOTE both functions should really take in the output of graph.complete_from_pos() and parse that
    # Form the equality constraints
    constraint_clique_dict = distance_constraints(graph, end_effectors, sparse, ee_cost)

    # Form the inequality constraints
    inequality_map = distance_range_constraints(
        graph.structure, constraint_clique_dict, end_effectors
    )  # this will only add joint limits

    # Nuclear norm cost (simple)
    nearest_points_nuclear = {
        key: np.zeros(robot.dim)
        for key in input_vals
        if key not in ["p0", "q0", f"p{robot.n}", f"q{robot.n}"]
    }

    solution, prob, constraint_clique_dict, sdp_variable_map = solve_nearest_point_sdp(
        nearest_points_nuclear,
        end_effectors,
        graph,
        sparse=False,
        solver_params=None,
        verbose=True,
        inequality_constraints_map=inequality_map,
    )
