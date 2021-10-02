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


def extract_relative_angle_Z(T_source: np.ndarray, T_target: np.ndarray) -> float:
    """

    :param T_source: source transformation matrix
    :param T_target: target transformation matrix that transforms e.e. coords to base coords
    :return:
    """
    T_rel = np.linalg.inv(T_source).dot(T_target)
    return np.arctan2(T_rel[1, 0], T_rel[0, 0])

def skew(x):
    """
    Creates a skew symmetric matrix from vector x
    """
    X = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
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

def generate_rotation_matrix(theta: float, axis: ArrayLike) -> ArrayLike:
    R = np.array([])

    c = math.cos(theta)
    s = math.sin(theta)

    if type(axis).__name__ == "str":
        if axis == "x":
            R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis == "y":
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        elif axis == "z":
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        else:
            R = np.array([False])
    else:
        x = axis[0]
        y = axis[1]
        z = axis[2]

        R = [
            [c + x ** 2 * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y ** 2 * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z ** 2 * (1 - c)],
        ]
        R = np.array(R)

    return R


def cross_symb(x, y):
    """
    Returns the cross product of x and y.
    Supports sympy by not using np.cross()
    :param x:
    :param y:
    :return:
    """
    z1 = -x[2] * y[1] + x[1] * y[2]
    z2 = x[2] * y[0] - x[0] * y[2]
    z3 = -x[1] * y[0] + x[0] * y[1]

    return np.array([z1, z2, z3])

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
    # print(AA)
    # print(BB)

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
