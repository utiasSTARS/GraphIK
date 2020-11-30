#!/usr/bin/env python3
import numpy as np
from graphik.robots.robot_base import RobotPlanar
import matplotlib.pyplot as plt
from graphik.utils.utils import list_to_variable_dict, variable_dict_to_list
from graphik.utils.robot_visualization import plot_planar_manipulator
from scipy.linalg import block_diag


def get_close_config(q: list, max=0.1, min=-0.1):
    n = len(q)
    q_deltas = np.random.random(n) * (max - min) + min
    q = np.array(q) + q_deltas
    return q.tolist()


def random_problem_2d_chain():
    """
    Create a random 2D chain
    """
    n = 3

    a = list_to_variable_dict(np.ones(n))
    th = list_to_variable_dict(np.zeros(n))
    lim_u = list_to_variable_dict(4 * np.pi / 6 * np.ones(n))
    lim_l = list_to_variable_dict(-4 * np.pi / 6 * np.ones(n))
    params = {
        "a": a,
        "theta": th,
        "joint_limits_upper": lim_u,
        "joint_limits_lower": lim_l,
    }
    # print(params[a])

    robot = RobotPlanar(params)

    return robot


def random_joint_angles(robot):
    """
    Create random joint angles
    """
    n = len(robot.a)
    ub = np.array(list(robot.ub.values()))
    lb = np.array(list(robot.lb.values()))

    ja = np.random.random(n) * (ub - lb) + lb
    ja = list_to_variable_dict(ja)
    return ja


def planar_jacobian(robot: RobotPlanar, q: list, ee: str):
    """
    Calculate a geometric Jacobian. See Chapter 3.1 from
    "Robot: Modelling, Planning, and Control" - Sicilliano
    """
    # assume all joints are revolute
    n = robot.n

    Jp = np.zeros((3, n))
    Jo = np.zeros((3, n))

    if type(robot) == Revolute3dChain:
        path_names = [f"p{i}" for i in range(0, robot.n + 1)]
    else:
        path_names = robot.kinematic_map["p0"][ee]

    if type(robot) == RobotPlanar:
        edges = list(robot.tree_graph().edges)  # for Revolute2dtree
    elif type(robot) == RobotPlanar:
        edges = list(robot.chain_graph().edges)  # for Revolute2dchain
    elif type(robot) == Revolute3dChain or type(robot) == Revolute3dTree:
        edges = [
            (node, path_names[p_ind + 1]) for p_ind, node in enumerate(path_names[0:-1])
        ]
    # elif type(robot) == Revolute3dTree:
    #     edges = [
    #
    #     ]

    # Ts = robot.get_full_pose_fast_lambdify(list_to_variable_dict(q))
    Ts = robot.get_all_poses(list_to_variable_dict(q))
    # Ts["p0"] = np.eye(4)

    T_0_ee = Ts[ee]
    pe = T_0_ee.as_matrix()[0:3, -1]

    for i_path, joint in enumerate(path_names[:-1]):
        T_0_i = Ts[joint].as_matrix()
        z_hat_i = T_0_i[0:3, 2]
        p_i = T_0_i[0:3, -1]
        edge = (joint, path_names[i_path + 1])
        j_idx = edges.index(edge)  # get joint column number
        Jp[:, j_idx] = np.cross(z_hat_i, pe - p_i)

        # Euler error jacobian as in eqn 3.88
        Jo[:, j_idx] = z_hat_i

    J = np.vstack([Jp, Jo])
    return J


def stacked_jacobians(robot: RobotPlanar, q: dict):
    """
    Creates large jacobian, stacked for each end effector, and it's pseudoinverse.
    """
    t = 0
    jacobians = []
    for ee in robot.end_effectors:
        J = planar_jacobian(robot, q, ee[0])
        jacobians.append(J)

    # stack them into numpy arraays
    jacobians = np.vstack(jacobians)

    # compute inverses
    jacobians_star = np.linalg.pinv(jacobians)  # slightly faster than individual pinv
    return jacobians, jacobians_star


def jacobian_ik(robot, q_init: dict, q_goal: dict, params=None, use_limits=True):
    """
    Jacobian based Inverse Kinematics
    Use axis angle represntation of orientation error
    p.139 in "Robotics: Modelling, Planning and Control" - Sicilliano
    """
    if params is None:
        tol = 1e-6
        maxiter = 5000
        dt = 1e-3
        method = "dls_inverse"
    else:
        tol = params["tol"]
        maxiter = params["maxiter"]
        dt = params["dt"]
        method = params["method"]

    n = robot.n
    ub = np.array(variable_dict_to_list(robot.ub))
    lb = np.array(variable_dict_to_list(robot.lb))
    q_bar = (ub + lb) / 2.0
    q = np.array(variable_dict_to_list(q_init))

    N_ee = len(robot.end_effectors)

    k = 0.01  # DLS jacobian inverse damping factor
    k0 = 20  # joint limit gain

    # gains
    K_p = np.eye(3) * 1000  # position gain
    K_o = np.eye(3) * 1000  # orientation gain

    K = np.eye(6)
    K[:3, :3] = K_p
    K[3:, 3:] = K_o
    K = np.kron(np.eye(N_ee), K)

    count = 0

    # Initialize system
    e = error(robot, q, q_goal)
    J, J_star = stacked_jacobians(robot, q)
    ll, llinv = stacked_L(robot, q, q_goal)
    q_dot = np.dot(J_star, np.dot(K, np.dot(llinv, e)))
    # loop unitl error is converged AND all joint angles are within bounds.
    while (
        np.linalg.norm(e) > tol or (any((q > ub) | (q < lb)) and use_limits)
    ) and count < maxiter:

        J, J_star = stacked_jacobians(robot, q)  # get jacobians

        e = error(robot, q, q_goal)  # Error to goal

        ll, llinv = stacked_L(
            robot, q, q_goal
        )  # Accounting for Euler Error (see eqn. 387 on p. 139)

        if use_limits:
            q_dot = (
                -k0 / n * (q - q_bar) / (ub - lb) * q_dot
            )  # Joint angle avoidance using eqn. 3.57 on p. 126
        q_dot = np.dot(J_star, np.dot(K, np.dot(llinv, e))) + np.dot(
            (np.eye(n) - np.dot(J_star, J)), q_dot
        )

        q = q + q_dot * dt  # update joint angles
        q = (q + np.pi) % (2 * np.pi) - np.pi  # wrap angles to -pi to pi

        if count % 100 == 0:
            print("count: %s" % count)
            print("error: %s" % e)
            print("q_dot: %s", q_dot)
            U, S, V = np.linalg.svd(J)
            cond = np.min(S) / np.max(S)
            print("Jacobian condition: %s" % cond)

            print("q: %s" % q)
        count += 1

    if count >= maxiter:
        print("Did not find config!")
        print("iterations: %s" % count)
        print("error: %s" % e)
        ja_violations = (q > ub) | (q < lb)
        print("Violations: %s" % ja_violations)
        return q, count
    else:

        print("Finished")
        print("iterations: %s" % count)
        print("error: %s" % e)
        print("Joint Angles: %s" % q)
        ja_violations = (q > ub) | (q < lb)
        print("Violations: %s" % ja_violations)
        return q, count


def error(robot, q, q_goal):
    """
    Compute the error using the euclidean distance and angular error
    of the end effector frame using a euler representation. Errors are
    stacked into an array
    """
    Ts_ee = robot.get_full_pose_fast_lambdify(list_to_variable_dict(q))
    Ts_goal = robot.get_full_pose_fast_lambdify(list_to_variable_dict(q_goal))

    T_0_ee = Ts_ee[robot.end_effectors[0][0]]
    T_0_goal = Ts_goal[robot.end_effectors[0][0]]
    err = error_raw(T_0_ee, T_0_goal)
    for ee in robot.end_effectors[1:]:
        T_0_ee = Ts_ee[ee[0]]
        T_0_goal = Ts_goal[ee[0]]
        err = np.concatenate([err, error_raw(T_0_ee, T_0_goal)])
    return err


def violations(robot, q):
    """
    True is there are any joint angle violations
    """
    ub = np.array(variable_dict_to_list(robot.ub))
    lb = np.array(variable_dict_to_list(robot.lb))
    ja_violations = (q > ub) | (q < lb)
    return any(ja_violations)


def L(Rd: np.array, Re: np.array):
    """
    eq. 3.85
    Accounts for using Geometric Jacobian and Euler error formulation
    """
    nd = Rd[:, 0]
    sd = Rd[:, 1]
    ad = Rd[:, 2]

    ne = Re[:, 0]
    se = Re[:, 1]
    ae = Re[:, 2]

    ll = -0.5 * (
        np.dot(skew(nd), skew(ne))
        + np.dot(skew(sd), skew(se))
        + np.dot(skew(ad), skew(ae))
    )

    llinv = np.linalg.pinv(ll)

    return ll, llinv


def stacked_L(robot: RobotPlanar, q: list, q_goal: list):
    """
    Stacks the L matrices for conviencne
    """
    LL = []
    LLinv = []

    Ts_ee = robot.get_full_pose_fast_lambdify(list_to_variable_dict(q))
    Ts_goal = robot.get_full_pose_fast_lambdify(list_to_variable_dict(q_goal))

    for ee in robot.end_effectors:

        T_0_ee = SE2_to_SE3(Ts_ee[ee[0]])
        Re = T_0_ee[0:3, 0:3]
        T_0_goal = SE2_to_SE3(Ts_goal[ee[0]])
        Rd = T_0_goal[0:3, 0:3]
        ll, llinv = L(Rd, Re)
        LL.append(np.eye(3))
        LLinv.append(np.eye(3))

        LL.append(ll)
        LLinv.append(llinv)

    LL = block_diag(*LL)
    LLinv = block_diag(*LLinv)

    return LL, LLinv


def error_raw(T_0_ee: np.array, T_0_goal: np.array):
    """
    Euler error for a single end effector
    """

    R_e = T_0_ee[0:3, 0:3]
    R_d = T_0_goal[0:3, 0:3]

    p_e = T_0_ee[0:3, 3]
    p_d = T_0_goal[0:3, 3]

    e_p = p_d - p_e
    e_o = euler_error(R_d, R_e)

    err = np.concatenate((e_p, e_o))

    return err


def euler_error(Rd, Re):
    """
    Euler orientation error, See eqn. 3.85
    """

    nd = Rd[:, 0]
    sd = Rd[:, 1]
    ad = Rd[:, 2]

    ne = Re[:, 0]
    se = Re[:, 1]
    ae = Re[:, 2]

    return 0.5 * (np.cross(ne, nd) + np.cross(se, sd) + np.cross(ae, ad))  # eqn 3.85


def skew(x):
    """
    Creates a skew symmetric matrix from vector x
    """
    X = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    return X


def draw_planar_frame(T, colour):
    p = np.array([T.trans[0], T.trans[1], 0])
    x_hat = T.rot.as_matrix()[:, 0]
    y_hat = T.rot.as_matrix()[:, 1]

    dx = x_hat
    dy = y_hat

    plt.plot(p[0], p[1], colour)
    plt.arrow(p[0], p[1], dx[0], dx[1], color="r")
    plt.arrow(p[0], p[1], dy[0], dy[1], color="g")


def plot_2D_chain(robot, q: list, fig=None, axes=None):
    x = [0.0]
    y = [0.0]
    q_tmp = list_to_variable_dict(q)
    for key in q_tmp:
        pos = robot.get_pose(q_tmp, key).trans
        x.append(pos[0])
        y.append(pos[1])

    plot_planar_manipulator(np.array(x), np.array(y), fig_handle=fig, ax_handle=axes)


def SE2_to_SE3(T_SE2: np.array):
    """
    Takes a 3x3 SE2 transform (as a numpy array) and converts it to a 4x4 SE3 transform
    (as a numpy array)
    """
    if T_SE2.shape == (3, 3):
        T_SE3 = np.insert(T_SE2, 2, np.array([0, 0, 0]), 0)
        T_SE3 = np.insert(T_SE3, 2, np.array([0, 0, 1, 0]), 1)
        return T_SE3
    elif T_SE2.shape == (4, 4):
        return T_SE2
    else:
        raise ("Bad Transform")
