import numpy as np
from numpy.linalg import norm
from numpy import pi
from graphik.utils.roboturdf import load_ur10, load_kuka, load_schunk_lwa4d, load_schunk_lwa4p
from graphik.utils import *
from graphik.graphs import RobotRevoluteGraph
from graphik.solvers.joint_angle_solver import JointAngleSolver


def solve_random_problem(graph: RobotRevoluteGraph, solver: JointAngleSolver):
    robot = graph.robot
    n = graph.robot.n

    feasible = False
    while not feasible:
        q_goal = robot.random_configuration()
        G_goal = graph.realization(q_goal)
        T_goal = robot.pose(q_goal, f"p{n}")
        broken_limits = graph.check_distance_limits(G_goal)
        if len(broken_limits) == 0:
            feasible = True

    goals = {f"p{n}": T_goal}
    q_sol, t_sol, nit = solver.solve(goals, robot.zero_configuration())

    T_local = robot.pose(q_sol, "p" + str(n))
    err_pos = norm(T_goal.trans - T_local.trans)

    z_goal = T_goal.as_matrix()[:3, 2]
    z_local = T_local.as_matrix()[:3, 2]
    err_rot = abs(safe_arccos(z_local.dot(z_goal)))

    fail = False
    if err_pos > 0.01 or err_rot > 0.01:
        fail = True

    broken_limits = graph.check_distance_limits(graph.realization(q_sol), tol=1e-6)
    print(broken_limits)
    print(f"Pos. error: {err_pos}\nRot. error: {err_rot}\nSolution time: {t_sol}.")
    print("------------------------------------")
    return err_pos, err_rot, t_sol, nit, fail

def main():
    np.random.seed(21)

    ### UR10 DH
    robot, graph = load_ur10()

    phi = (1 + np.sqrt(5)) / 2
    scale = 0.5
    radius = 0.5
    obstacles = [
        # (scale * np.asarray([0, 1, phi]), radius),
        # (scale * np.asarray([0, 1, -phi]), radius),
        # (scale * np.asarray([0, -1, -phi]), radius),
        # (scale * np.asarray([0, -1, phi]), radius),
        # (scale * np.asarray([1, phi, 0]), radius),
        # (scale * np.asarray([1, -phi, 0]), radius),
        # (scale * np.asarray([-1, -phi, 0]), radius),
        # (scale * np.asarray([-1, phi, 0]), radius),
        # (scale * np.asarray([phi, 0, 1]), radius),
        # (scale * np.asarray([-phi, 0, 1]), radius),
        # (scale * np.asarray([-phi, 0, -1]), radius),
        # (scale * np.asarray([phi, 0, -1]), radius),
    ]
    for idx, obs in enumerate(obstacles):
        graph.add_spherical_obstacle(f"o{idx}", obs[0], obs[1])

    params = {"W": np.eye(robot.n)}
    solver = JointAngleSolver(graph, params)
    num_tests = 100
    e_pos = []
    e_rot = []
    t = []
    fails = []
    nit = []
    for _ in range(num_tests):
        e_r_pos, e_r_rot, t_sol, num_iter, fail = solve_random_problem(graph, solver)
        e_pos += [e_r_pos]
        e_rot += [e_r_rot]
        t += [t_sol]
        fails += [fail]
        nit += [num_iter]

    t = np.asarray(t)
    t = t[abs(t - np.mean(t)) < 2 * np.std(t)]
    print("Average solution time {:}".format(np.average(t)))
    print("Median solution time {:}".format(np.median(t)))
    print("Standard deviation of solution time {:}".format(np.std(t)))
    print("Average iterations {:}".format(np.average(nit)))
    print("Median iterations {:}".format(np.median(nit)))
    print("Average pos error {:}".format(np.average(np.asarray(e_pos))))
    print("Average rot error {:}".format(np.average(np.asarray(e_rot))))
    print("Number of fails {:}".format(sum(fails)))

if __name__ == "__main__":
    main()
