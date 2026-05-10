import numpy as np

from typing import Tuple
from numpy.typing import ArrayLike, NDArray


def skew(x: NDArray) -> NDArray:
    """
    Creates a 3x3 skew symmetric matrix from a 3d vector x.
    :param x: 3d vector
    "returns: 3x3 skew symmetric matrix
    """
    X = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    return X


def max_min_distance_revolute(r, P, C, N):
    delta = P - C
    d_min_s = N.dot(delta) ** 2 + (np.linalg.norm(np.cross(N, delta)) - r) ** 2
    if d_min_s > 0:
        d_min = np.sqrt(d_min_s)
    else:
        d_min = 0
    d_max_s = N.dot(delta) ** 2 + (np.linalg.norm(np.cross(N, delta)) + r) ** 2
    if d_max_s > 0:
        d_max = np.sqrt(d_max_s)
    else:
        d_max = 0

    return d_max, d_min


def best_fit_transform(A: NDArray, B: NDArray) -> Tuple[NDArray, NDArray]:
    """
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions.
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      R: mxm rotation matrix
      t: mx1 translation vector
    """
    assert A.shape == B.shape

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.eye(A.shape[1])
    D[-1, -1] = d
    R = Vt.T @ D @ U.T

    t = centroid_B.T - np.dot(R, centroid_A.T)
    return R, t
