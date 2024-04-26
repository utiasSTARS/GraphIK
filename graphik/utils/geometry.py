import numpy as np

from typing import Tuple
from numpy.typing import ArrayLike
from liegroups.numpy import SO2, SE2, SO3, SE3
from numpy import sin, cos

def angle_to_se2(a: float, theta: float) -> SE2:
    """Transform a single set of DH parameters into an SE2 matrix
    :param a: link length
    :param theta: rotation
    :returns: SE2 matrix
    :rtype: lie.SE2Matrix
    """
    # R = SO2.from_angle(theta)  # TODO: active or passive (i.e., +/- theta?)
    R = SO2.from_angle(theta)
    return SE2(R, R.dot(np.array([a, 0.0])))  # TODO: rotate the translation or not?

def skew(x):
    """
    Creates a skew symmetric matrix from vector x
    """
    X = np.array([[0., -x[2], x[1]], [x[2], 0., -x[0]], [-x[1], x[0], 0.]])
    return X

def trans_axis(t, axis="z") -> SE3:
    if axis == "z":
        return SE3(SO3.identity(), np.array([0, 0, t]))
    if axis == "y":
        return SE3(SO3.identity(), np.array([0, t, 0]))
    if axis == "x":
        return SE3(SO3.identity(), np.array([t, 0, 0]))
    raise Exception("Invalid Axis")


def rot_axis(theta, axis="z") -> SE3:
    if axis == "z":
        return SE3(SO3.rotz(theta), np.array([0, 0, 0]))
    if axis == "y":
        return SE3(SO3.roty(theta), np.array([0, 0, 0]))
    if axis == "x":
        return SE3(SO3.rotx(theta), np.array([0, 0, 0]))
    raise Exception("Invalid Axis")

def max_min_distance_revolute(r, P, C, N):
    delta = P-C
    d_min_s = N.dot(delta)**2 + (np.linalg.norm(np.cross(N, delta)) - r)**2
    if d_min_s > 0:
        d_min = np.sqrt(d_min_s)
    else:
        d_min = 0
    d_max_s = N.dot(delta)**2 + (np.linalg.norm(np.cross(N, delta)) + r)**2
    if d_max_s > 0:
        d_max = np.sqrt(d_max_s)
    else:
        d_max = 0

    return d_max, d_min

def best_fit_transform(A: ArrayLike, B: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    """
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      R: mxm rotation matrix
      t: mx1 translation vector
    """

    # try:
    assert A.shape == B.shape
    # except AssertionError:
    #     print("A: {:}".format(A))
    #     print("B: {:}".format(B))

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # translation
    #
    # special reflection case
    # if np.linalg.det(R) < 0:
    #     print("det(R) < R, reflection detected!, correcting for it ...\n")
    # Vt[2, :] *= -1
    # R = np.dot(Vt.T, U.T)

    t = centroid_B.T - np.dot(R, centroid_A.T)
    return R, t
