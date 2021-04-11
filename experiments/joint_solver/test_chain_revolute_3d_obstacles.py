import graphik
import time
import numpy as np
from numpy.linalg import norm
from numpy import pi
from graphik.utils.roboturdf import RobotURDF
from graphik.utils.constants import *
from graphik.graphs.graph_base import RobotGraph, RobotRevoluteGraph
from graphik.solvers.jacobian_solver import LocalSolver
from graphik.utils.utils import list_to_variable_dict, safe_arccos


def solve_random_problem(graph: RobotGraph, solver: LocalSolver):
    n = graph.robot.n

    feasible = False
    while not feasible:
        q_goal = robot.random_configuration()
        G_goal = graph.realization(q_goal)
        T_goal = robot.get_pose(q_goal, f"p{n}")
        broken_limits = graph.check_distance_limits(G_goal)
        if len(broken_limits) == 0:
            feasible = True

    goals = {f"p{n}": T_goal}
    t_sol = time.perf_counter()
    res = solver.solve(goals, robot.random_configuration())
    t_sol = time.perf_counter() - t_sol
    q_sol = list_to_variable_dict(res.x)

    T_local = robot.get_pose(q_sol, "p" + str(n))
    err_pos = norm(T_goal.trans - T_local.trans)

    z_goal = T_goal.as_matrix()[:3, 2]
    z_local = T_local.as_matrix()[:3, 2]
    err_rot = abs(safe_arccos(z_local.dot(z_goal)))

    fail = False
    if err_pos > 0.01 or err_rot > 0.01:
        fail = True

    broken_limits = graph.check_distance_limits(graph.realization(q_sol))
    print(broken_limits)
    print(f"Pos. error: {err_pos}\nRot. error: {err_rot}\nSolution time: {t_sol}.")
    print("------------------------------------")
    return err_pos, err_rot, t_sol, fail


if __name__ == "__main__":

    np.random.seed(21)

    ### UR10 DH
    n = 6
    ub = (pi) * np.ones(n)
    lb = -ub
    modified_dh = False

    fname = graphik.__path__[0] + "/robots/urdfs/ur10_mod.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/lwa4p.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/lwa4d.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/panda_arm.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/kuka_iiwr.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/kuka_lwr.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/jaco2arm6DOF_no_hand.urdf"

    urdf_robot = RobotURDF(fname)
    robot = urdf_robot.make_Revolute3d(ub, lb)  # make the Revolute class from a URDF

    graph = RobotRevoluteGraph(robot)

    obstacles = [
        (np.asarray([0, 1, 0.75]), 0.75),
        (np.asarray([0, 1, -0.75]), 0.75),
        (np.asarray([0, -1, 0.75]), 0.75),
        (np.asarray([0, -1, -0.75]), 0.75),
        (np.asarray([1, 0, 0.75]), 0.75),
        (np.asarray([1, 0, -0.75]), 0.75),
        (np.asarray([-1, 0, 0.75]), 0.75),
        (np.asarray([-1, 0, -0.75]), 0.75),
    ]

    for idx, obs in enumerate(obstacles):
        graph.add_spherical_obstacle(f"o{idx}", obs[0], obs[1])

    params = {"W": np.eye(n)}
    solver = LocalSolver(graph, params)
    num_tests = 100
    e_pos = []
    e_rot = []
    t = []
    fails = []
    for _ in range(num_tests):
        e_r_pos, e_r_rot, t_sol, fail = solve_random_problem(graph, solver)
        e_pos += [e_r_pos]
        e_rot += [e_r_rot]
        t += [t_sol]
        fails += [fail]

    t = np.asarray(t)
    t = t[abs(t - np.mean(t)) < 2 * np.std(t)]
    print("Average solution time {:}".format(np.average(t)))
    print("Standard deviation of solution time {:}".format(np.std(t)))
    print("Average pos error {:}".format(np.average(np.asarray(e_pos))))
    print("Average rot error {:}".format(np.average(np.asarray(e_rot))))
    print("Number of fails {:}".format(sum(fails)))
