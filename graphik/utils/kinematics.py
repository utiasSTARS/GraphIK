import numpy as np
from liegroups.numpy import SO2, SO3, SE2, SE3

##################################################
# Planar joints
##################################################

def dh_to_se2(a: float, theta: float) -> SE2:
    """Transform a single set of DH parameters into an SE2 matrix
    :param a: link length
    :param theta: rotation
    :returns: SE2 matrix
    """
    R = SO2.from_angle(theta)
    return SE2(R, R.dot(np.array([a, 0.0])))  # TODO: rotate the translation or not?

def fk_2d(a: list, theta: list, q: list) -> SE2:
    """Get forward kinematics from an array of link lengths and angles
    :param a: displacement along x (link length)
    :param theta: rotation offset of each link
    :param q: active angle variable
    :returns: SE2 matrix
    :rtype: lie.SE2Matrix
    """
    if len(a) > 1:
        return dh_to_se2(a[0], theta[0] + q[0]).dot(fk_2d(a[1:], theta[1:], q[1:]))
    return dh_to_se2(a[0], theta[0] + q[0])


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
        return dh_to_se2(a[idx], theta[idx] + q[idx]).dot(
            fk_tree_2d(a, theta, q, path_indices[1:])
        )
    return dh_to_se2(a[idx], theta[idx] + q[idx])


##################################################
# 3D single-axis of rotation
##################################################

def trans_axis(t: float, axis="z") -> SE3:
    if axis == "z":
        return SE3(SO3.identity(), np.array([0, 0, t]))
    if axis == "y":
        return SE3(SO3.identity(), np.array([0, t, 0]))
    if axis == "x":
        return SE3(SO3.identity(), np.array([t, 0, 0]))
    raise Exception("Invalid Axis")


def rot_axis(theta: float, axis="z") -> SE3:
    if axis == "z":
        return SE3(SO3.rotz(theta), np.array([0, 0, 0]))
    if axis == "y":
        return SE3(SO3.roty(theta), np.array([0, 0, 0]))
    if axis == "x":
        return SE3(SO3.rotx(theta), np.array([0, 0, 0]))
    raise Exception("Invalid Axis")

def dh_to_se3(a: float, alpha: float, d: float, theta: float) -> SE3:
    """Transform a single set of DH parameters into an SE3 matrix
    :param a: displacement along x
    :param alpha: rotation about x
    :param d: translation along new z
    :param theta: rotation around new z
    :returns: SE3 matrix
    :rtype: lie.SE3Matrix
    """

    TransX = trans_axis(a,"x")
    RotX = rot_axis(alpha,"x")
    TransZ = trans_axis(d,"z")
    RotZ = rot_axis(theta,"z")

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

    TransX = trans_axis(a,"x")
    RotX = rot_axis(alpha,"x")
    TransZ = trans_axis(d,"z")
    RotZ = rot_axis(theta,"z")

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
    out = dh_to_se3(a[0], alpha[0], d[0], theta[0])
    for idx in range(1,len(a)):
        out = out.dot(dh_to_se3(a[idx], alpha[idx], d[idx], theta[idx]))
    return out
    # if len(a) > 1:
    #     return dh_to_se3(a[0], alpha[0], d[0], theta[0]).dot(
    #         fk_3d(a[1:], alpha[1:], d[1:], theta[1:])
    #     )
    # return dh_to_se3(a[0], alpha[0], d[0], theta[0])


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
    out = modified_dh_to_se3(a[0], alpha[0], d[0], theta[0])
    for idx in range(1,len(a)):
        out = out.dot(modified_dh_to_se3(a[idx], alpha[idx], d[idx], theta[idx]))
    return out
    # if len(a) > 1:
    #     return modified_dh_to_se3(a[0], alpha[0], d[0], theta[0]).dot(
    #         modified_fk_3d(a[1:], alpha[1:], d[1:], theta[1:])
    #     )
    # return modified_dh_to_se3(a[0], alpha[0], d[0], theta[0])
