import numpy as np
import scipy as sp
import networkx as nx
from numpy import pi
from liegroups.numpy import SE3
from liegroups.numpy import SO3
import math

# transZ = se3.SE3(so3.SO3.identity(), np.array([0, 0, 0.5]))

dZ = 1


def transZ(d):
    return SE3(SO3.identity(), np.array([0, 0, d]))


def rotZ(x):
    return SE3(SO3.rotz(x), np.array([0, 0, 0]))


def trans_axis(d, axis="z"):
    if axis == "z":
        return SE3(SO3.identity(), np.array([0, 0, d]))
    if axis == "y":
        return SE3(SO3.identity(), np.array([0, d, 0]))
    if axis == "x":
        return SE3(SO3.identity(), np.array([d, 0, 0]))
    raise Exception("Invalid Axis")


def rot_axis(x, axis="z"):
    if axis == "z":
        return SE3(SO3.rotz(x), np.array([0, 0, 0]))
    if axis == "y":
        return SE3(SO3.roty(x), np.array([0, 0, 0]))
    if axis == "x":
        return SE3(SO3.rotx(x), np.array([0, 0, 0]))
    raise Exception("Invalid Axis")


def level2_descendants(G: nx.DiGraph, node_id):
    successors = G.successors(node_id)

    desc = []
    for su in successors:
        desc += [G.successors(su)]

    return flatten(desc)


def norm_sq(A: np.ndarray) -> np.ndarray:
    """Returns the squared L2-norm of a symbolic vector"""
    return A.transpose().dot(A)


def wraptopi(e):
    return np.mod(e + pi, 2 * pi) - pi


def flatten(l: list) -> list:
    return [item for sublist in l for item in sublist]


def list_to_variable_dict(l: list, label="p", index_start=1):
    if type(l) is dict:
        return l
    var_dict = {}
    for idx, val in enumerate(l):
        var_dict[label + str(index_start + idx)] = val
    return var_dict


def list_to_variable_dict_spherical(l: list, label="p", index_start=1, in_pairs=False):
    var_dict = {}
    if in_pairs:
        for idx, val in enumerate(l):
            if idx % 2 == 0:
                var_dict[label + str(index_start + idx // 2)] = [val]
            else:
                var_dict[label + str(index_start + (idx - 1) // 2)].append(val)
    else:
        for idx, val in enumerate(l):
            var_dict[
                label + str(index_start + idx // 2) + "_" + str(index_start + idx % 2)
            ] = val
    return var_dict


def bounds_to_spherical_bounds(bounds, max_val=np.inf):
    new_bounds = []
    for b in bounds:
        new_bounds.append(max_val)
        new_bounds.append(b)
    return new_bounds


def variable_dict_to_list(d: dict, order: list = None) -> list:
    if order is None:
        return [d[item] for item in d]
    else:
        return [d[item] for item in order]


def apply_angular_offset_2d(joint_angle_offset, z0):
    """
    :param joint_angle_offset:
    :param z0:
    :return:
    """
    R = np.zeros((2, 2))
    R[0, 0] = np.cos(joint_angle_offset)
    R[0, 1] = np.sin(joint_angle_offset)
    R[1, 0] = -np.sin(joint_angle_offset)
    R[1, 1] = R[0, 0]
    return R * z0


def constraint_violations(constraints, solution_dict):
    return [
        (
            con.args[1].subs(solution_dict) - con.args[0].subs(solution_dict),
            con.is_Equality,
        )
        for con in constraints
    ]


def normalize(x: np.array):
    """Returns normalized vector x"""
    return x / np.linalg.norm(x, 2)


def best_fit_transform(A: np.ndarray, B: np.ndarray) -> (np.ndarray, np.ndarray):
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
    #     Vt[2, :] *= -1
    #     R = np.dot(Vt.T, U.T)

    t = centroid_B.T - np.dot(R, centroid_A.T)
    return R, t


def generate_rotation_matrix(theta, axis):
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


def make_save_string(save_properties: list) -> str:
    """

    :param save_properties: list of tuples containing (property, val)
    :return: save string with underscores delimiting values and properties
    """
    save_string = ""
    return save_string.join([p + "_" + str(v) + "_" for p, v in save_properties])


def spherical_angle_bounds_to_revolute(ub_spherical, lb_spherical):
    ub = {}
    lb = {}
    count_angle_bounds = 1
    for key in ub_spherical:  # Assumes the keys are numerically sorted
        ub[f"p{count_angle_bounds}"] = np.pi
        lb[f"p{count_angle_bounds}"] = -np.pi
        count_angle_bounds += 1
        ub[f"p{count_angle_bounds}"] = ub_spherical[key]
        lb[f"p{count_angle_bounds}"] = lb_spherical[key]
        count_angle_bounds += 1

    return ub, lb


def safe_arccos(t):
    t_in = min(max(-1, t), 1)
    return np.arccos(t_in)


def bernoulli_confidence_normal_approximation(n, n_success, confidence=0.95):
    """

    :param n:
    :param n_success:
    :param confidence:
    :return:
    """
    alpha = 1.0 - confidence
    z = sp.special.ndtri(1.0 - alpha / 2.0)
    p_hat = n_success / n
    rad = z * np.sqrt((p_hat * (1 - p_hat)) / n)
    return p_hat, rad


def wilson(n, n_success, alpha=0.95):
    p = n_success / n
    z = sp.special.ndtri(1.0 - alpha / 2.0)
    denominator = 1 + z ** 2 / n
    centre_adjusted_probability = p + z * z / (2 * n)
    adjusted_standard_deviation = np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    lower_bound = (
        centre_adjusted_probability - z * adjusted_standard_deviation
    ) / denominator
    upper_bound = (
        centre_adjusted_probability + z * adjusted_standard_deviation
    ) / denominator
    return (lower_bound, upper_bound)


def bernoulli_confidence_jeffreys(n, n_success, confidence=0.95):
    alpha_low = (1.0 - confidence) / 2.0
    alpha_high = confidence + alpha_low
    a = n_success + 0.5
    b = n - n_success + 0.5
    low_end = 0.0 if n_success == 0 else sp.special.btdtri(a, b, alpha_low)
    high_end = 1.0 if n_success == n else sp.special.btdtri(a, b, alpha_high)
    p_hat = (low_end + high_end) / 2.0
    rad = (high_end - low_end) / 2.0

    return p_hat, rad


if __name__ == "__main__":

    # print("Bernoulli: ")
    # print(bernoulli_confidence_normal_approximation(100, 100))
    # print("Jeffreys: ")
    # print(bernoulli_confidence_jeffreys(100, 100))

    print("Bernoulli: ")
    print(bernoulli_confidence_normal_approximation(100, 99))
    print("Jeffreys: ")
    print(bernoulli_confidence_jeffreys(100, 99))
