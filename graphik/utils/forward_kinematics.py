import numpy as np
import sympy as sp

from graphik.utils.geometry import *
from liegroups.numpy import SO2, SO3, SE2, SE2


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


def fk_2d_symb(a: list, theta: list, q: list) -> SE2:
    angle_full = sum(theta) + sum(q)
    # c_all = sp.cos(angle_full)
    # s_all = sp.sin(angle_full)
    R = SO2(from_angle(angle_full))
    t = np.zeros(2)
    for idx in range(len(a)):
        angle_idx = sum(theta[: idx + 1]) + sum(q[: idx + 1])
        t = t + np.array([sp.cos(angle_idx) * a[idx], sp.sin(angle_idx) * a[idx]])
    return SE2(R, t)


##################################################
# 3D single-axis of rotation
##################################################
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


##################################################
# 3D spherical (two axes of rotation)
##################################################
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


def fk_3d_sph_symb(a: list, alpha: list, d: list, theta: list) -> SE3:
    """Get forward kinematics from an array of dh parameters using sympy's trigonometric function.
    Used for setting up the symbolic cost function in graphik.solvers.local_solver.LocalSolver

    :param a: displacement along x
    :param alpha: rotation about x
    :param d: translation along new z
    :param theta: rotation around new z
    :returns: SE3 matrix
    :rtype: lie.SE3Matrix
    """
    if len(d) > 1:
        return sph_to_se3_symb(alpha[0], d[0], theta[0]).dot(
            fk_3d_sph_symb(a[1:], alpha[1:], d[1:], theta[1:])
        )
    return sph_to_se3_symb(alpha[0], d[0], theta[0])
