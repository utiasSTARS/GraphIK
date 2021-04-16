import numpy as np
from liegroups.numpy import SO2, SO3, SE2, SE3
from scipy.optimize import fsolve


##################################################
# Planar joints
##################################################
def fk_2d(a: list, theta: list, q: list) -> SE2:
    """Get forward kinematics from an array of link lengths and angles
    :param a: displacement along x (link length)
    :param theta: rotation offset of each link
    :param q: active angle variable
    :returns: SE2 matrix
    :rtype: lie.SE2Matrix
    """
    if len(a) > 1:
        return angle_to_se2(a[0], theta[0] + q[0]).dot(fk_2d(a[1:], theta[1:], q[1:]))
    return angle_to_se2(a[0], theta[0] + q[0])


def fk_tree_2d(a: list, theta: list, q: list, path_indices: list) -> SE2:
    """Get forward kinematics from an array of link lengths and angles
    :param a: displacement along x (link length)
    :param theta: rotation offset of each link
    :param q: active angle variable
    :param path_indices: path through the tree to traverse
    :returns: SE2 matrix
    :rtype: lie.SE2Matrix
    """
    idx = path_indices[0]
    if len(path_indices) > 1:
        return angle_to_se2(a[idx], theta[idx] + q[idx]).dot(
            fk_tree_2d(a, theta, q, path_indices[1:])
        )
    return angle_to_se2(a[idx], theta[idx] + q[idx])


##################################################
# 3D single-axis of rotation
##################################################

def dh_to_se3(a: float, alpha: float, d: float, theta: float) -> SE3:
    """Transform a single set of DH parameters into an SE3 matrix
    :param a: displacement along x
    :param alpha: rotation about x
    :param d: translation along new z
    :param theta: rotation around new z
    :returns: SE3 matrix
    :rtype: lie.SE3Matrix
    """

    TransX = SE3(SO3.identity(), np.array([a, 0, 0]))
    RX = SO3(rotx(alpha))
    RotX = SE3(RX, np.zeros(3))
    TransZ = SE3(SO3.identity(), np.array([0, 0, d]))
    RZ = SO3(rotz(theta))
    RotZ = SE3(RZ, np.zeros(3))
    return TransZ.dot(RotZ.dot(TransX.dot(RotX)))


def modified_dh_to_se3(a: float, alpha: float, d: float, theta: float) -> SE3:
    """Transform a single set of modified DH parameters into an SE3 matrix
    :param a: displacement along x
    :param alpha: rotation about x
    :param d: translation along new z
    :param theta: rotation around new z
    :returns: SE3 matrix
    :rtype: lie.SE3Matrix
    """

    TransX = SE3(SO3.identity(), np.array([a, 0, 0]))
    # RX = SO3.rotx(alpha)
    RX = SO3(rotx(alpha))
    RotX = SE3(RX, np.zeros(3))
    TransZ = SE3(SO3.identity(), np.array([0, 0, d]))
    # RZ = SO3.rotz(theta)
    RZ = SO3(rotz(theta))
    RotZ = SE3(RZ, np.zeros(3))

    return TransX.dot(RotX.dot(TransZ.dot(RotZ)))

def fk_3d(a: list, alpha: list, d: list, theta: list) -> SE3:
    """Get forward kinematics from an array of dh parameters
    :param a: displacement along x
    :param alpha: rotation about x
    :param d: translation along new z
    :param theta: rotation around new z
    :returns: SE3 matrix
    :rtype: lie.SE3Matrix
    """
    if len(a) > 1:
        return dh_to_se3(a[0], alpha[0], d[0], theta[0]).dot(
            fk_3d(a[1:], alpha[1:], d[1:], theta[1:])
        )
    return dh_to_se3(a[0], alpha[0], d[0], theta[0])


def fk_tree_3d(a: list, alpha: list, d: list, theta: list, path_indices: list) -> SE3:
    """Get forward kinematics from an array of link lengths and angles
    :param a: displacement along x (link length)
    :param theta: rotation offset of each link
    :param q: active angle variable
    :param path_indices: path through the tree to traverse
    :returns: SE2 matrix
    :rtype: lie.SE2Matrix
    """
    idx = path_indices[0]
    if len(path_indices) > 1:
        return dh_to_se3(a[idx], alpha[idx], d[idx], theta[idx]).dot(
            fk_tree_3d(a[1:], alpha[1:], d[1:], theta[1:], path_indices[1:])
        )
    return dh_to_se3(a[idx], alpha[idx], d[idx], theta[idx])


def modified_fk_3d(a: list, alpha: list, d: list, theta: list) -> SE3:
    """Get forward kinematics from an array of  modified dh parameters
    :param a: displacement along x
    :param alpha: rotation about x
    :param d: translation along new z
    :param theta: rotation around new z
    :returns: SE3 matrix
    :rtype: lie.SE3Matrix
    """
    if len(a) > 1:
        return modified_dh_to_se3(a[0], alpha[0], d[0], theta[0]).dot(
            modified_fk_3d(a[1:], alpha[1:], d[1:], theta[1:])
        )
    return modified_dh_to_se3(a[0], alpha[0], d[0], theta[0])

def inverse_dh_frame(q: np.ndarray, a: float, d: float, alpha, tol=1e-6) -> float:
    """
    Compute the approximate least-squares DH frame IK when given the other three DH parameters (a, d, alpha) and the
    origin of the next frame.

    :param q: value of the z-axis (i.e. q) for the link we are trying to localize.
    :param a: DH parameter a
    :param d: DH parameter d
    :param alpha: DH parameter alpha
    :return: float representing the DH parameter theta for this link
    """
    c_alpha = np.cos(alpha)
    s_alpha = np.sin(alpha)
    A = np.array([[a, s_alpha], [-s_alpha, a], [0.0, 0.0]])
    b = q - np.array([0.0, 0.0, c_alpha + d])
    sol_candidate = np.linalg.pinv(A) @ b
    # print("\n\n---------------------------------------------")
    # print("Solution candidate: {:}".format(sol_candidate))

    if sum(np.abs(sol_candidate)) < 1e-9:
        # print("Unobservable angle! Returning 0.")
        # print("A matrix: {:}".format(A))
        # print("q1: {:}".format(q))
        # print("alpha: {:}".format(alpha))
        # print("d: {:}".format(d))
        # print("b vector: {:}".format(b))
        return 0.0, dh_to_se3(a, alpha, d, 0.0).as_matrix()

    residual = np.abs(np.linalg.norm(sol_candidate) - 1.0)
    # print("Residual: {:}".format(residual))
    if residual <= tol:
        sol = sol_candidate / np.linalg.norm(sol_candidate)
    else:
        # Iterative least squares
        # print("Noise detected! Using iterative solver on the Lagrangian.")
        f_lam = (
            lambda lam: np.linalg.norm(
                np.linalg.inv(A.T @ A + lam * np.eye(2)) @ A.T @ b
            )
            - 1.0
        )
        lam_opt = fsolve(f_lam, x0=0.0)
        sol = np.linalg.inv(A.T @ A + lam_opt * np.eye(2)) @ A.T @ b

    # print("Solution: {:}".format(sol))
    theta = np.arctan2(sol[1], sol[0])
    T_dh = dh_to_se3(a, alpha, d, theta).as_matrix()

    return theta, T_dh

##################################################
# 3D spherical (two axes of rotation)
##################################################
def sph_to_se3(alpha: float, d: float, theta: float) -> SE3:
    """Transform a single set of DH parameters into an SE3 matrix
    :param a: displacement along x
    :param alpha: rotation about x
    :param d: translation along new z
    :param theta: rotation around new z
    :returns: SE3 matrix
    :rtype: lie.SE3Matrix
    """
    RotX = rot_axis(alpha, "x")
    TransZ = trans_axis(d, "z")
    RotZ = rot_axis(theta, "z")
    return RotZ.dot(RotX.dot(TransZ))

def fk_3d_sph(a: list, alpha: list, d: list, theta: list) -> SE3:
    """Get forward kinematics from an array of dh parameters
    :param a: displacement along x
    :param alpha: rotation about x
    :param d: translation along new z
    :param theta: rotation around new z
    :returns: SE3 matrix
    :rtype: lie.SE3Matrix
    """
    if len(d) > 1:
        return sph_to_se3(alpha[0], d[0], theta[0]).dot(
            fk_3d_sph(a[1:], alpha[1:], d[1:], theta[1:])
        )
    return sph_to_se3(alpha[0], d[0], theta[0])
