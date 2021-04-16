import numpy as np
import scipy as sp
import networkx as nx
from numpy import pi

dZ = 1


def perpendicular_vector(v):
    """ Returns arbitrary perpendicular vector """

    if np.isclose(v[1], 0.) and np.isclose(v[2], 0.):

        if np.isclose(v[0], 0.):
            raise ValueError('Received zero vector.')
        else:
            return np.cross(v, np.array([0,1,0]))
    
    return np.cross(v, np.array([1,0,0]))

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


def variable_dict_to_list(d: dict, order: list = None) -> list:
    if order is None:
        return [d[item] for item in d]
    else:
        return [d[item] for item in order]


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
    # Vt[2, :] *= -1
    # R = np.dot(Vt.T, U.T)

    t = centroid_B.T - np.dot(R, centroid_A.T)
    return R, t



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


def measure_perturbation(points: dict, points_perturbed: dict) -> (float, float):
    squared_sum = 0.
    max_perturb = -np.inf
    for key in points:
        p = points[key]
        p_perturbed = points_perturbed[key]
        max_perturb = max(max_perturb, max(np.abs(p-p_perturbed)))
        squared_sum += np.linalg.norm(p - p_perturbed)**2

    return np.sqrt(squared_sum), max_perturb


def constraint_violations(constraints, solution_dict):
    return [
        (
            con.args[1].subs(solution_dict) - con.args[0].subs(solution_dict),
            con.is_Equality,
        )
        for con in constraints
    ]


if __name__ == "__main__":

    # print("Bernoulli: ")
    # print(bernoulli_confidence_normal_approximation(100, 100))
    # print("Jeffreys: ")
    # print(bernoulli_confidence_jeffreys(100, 100))

    print("Bernoulli: ")
    print(bernoulli_confidence_normal_approximation(100, 99))
    print("Jeffreys: ")
    print(bernoulli_confidence_jeffreys(100, 99))
