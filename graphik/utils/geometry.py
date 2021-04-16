import numpy as np
import sympy as sp

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
    R = SO2(from_angle(theta))
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


def exp(phi):
    """Exponential map for :math:`SO(2)`, which computes a transformation from a tangent vector:

    .. math::
        \\mathbf{C}(\\phi) =
        \\exp(\\phi^\\wedge) =
        \\cos \\phi \\mathbf{1} + \\sin \\phi 1^\\wedge =
        \\begin{bmatrix}
            \\cos \\phi  & -\\sin \\phi  \\\\
            \\sin \\phi & \\cos \\phi
        \\end{bmatrix}

    This is the inverse operation to :meth:`~liegroups.SO2.log`.
    """
    c = sp.cos(phi)
    s = sp.sin(phi)

    if type(phi) == sp.Symbol or type(phi) == sp.Add:
        return np.array([[c, -s], [s, c]])
    else:
        return np.array([[c, -s], [s, c]], dtype="float64")


def from_angle(angle_in_radians):
    """Form a rotation matrix given an angle in radians.

    See :meth:`~liegroups.SO2.exp`
    """
    return exp(angle_in_radians)

def rotmat_from_two_vector_matches(a_1, a_2, b_1, b_2):
    """ Returns C in SO(3), such that b_1 = C*a_1 and b_2 = C*a_2"""
    ## Construct orthonormal basis of 'a' frame
    a_1_u = a_1/(np.linalg.norm(a_1))
    a_2_u = a_2/(np.linalg.norm(a_2))
    alpha = a_1_u.dot(a_2_u)

    a_basis_1 = a_1_u
    a_basis_2 = a_2_u - alpha*a_1_u
    a_basis_2 /= np.linalg.norm(a_basis_2)
    a_basis_3 = np.cross(a_basis_1, a_basis_2)

    ## Construct basis of 'b' frame
    b_basis_1 = b_1/np.linalg.norm(b_1)
    b_basis_2 = b_2/np.linalg.norm(b_2) - alpha*b_basis_1
    b_basis_2 /= np.linalg.norm(b_basis_2)
    b_basis_3 = np.cross(b_basis_1, b_basis_2)

    #Basis of 'a' frame as column vectors
    M_a = np.array([a_basis_1, a_basis_2, a_basis_3])

    #Basis of 'b' frame as row vectors
    M_b = np.array([b_basis_1, b_basis_2, b_basis_3]).T

    #Direction cosine matrix from a to b!
    C = M_b.dot(M_a)
    return SO3.from_matrix(C)

def rotx(angle_in_radians):
    """Form a rotation matrix given an angle in rad about the x-axis.

    .. math::
        \\mathbf{C}_x(\\phi) =
        \\begin{bmatrix}
            1 & 0 & 0 \\\\
            0 & \\cos \\phi & -\\sin \\phi \\\\
            0 & \\sin \\phi & \\cos \\phi
        \\end{bmatrix}
    """
    c = cos(angle_in_radians)
    s = sin(angle_in_radians)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype="float64")


def roty(angle_in_radians):
    """Form a rotation matrix given an angle in rad about the y-axis.

    .. math::
        \\mathbf{C}_y(\\phi) =
        \\begin{bmatrix}
            \\cos \\phi & 0 & \\sin \\phi \\\\
            0 & 1 & 0 \\\\
            \\sin \\phi & 0 & \\cos \\phi
        \\end{bmatrix}
    """
    c = cos(angle_in_radians)
    s = sin(angle_in_radians)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype="float64")


def rotz(angle_in_radians):
    """Form a rotation matrix given an angle in rad about the z-axis.

    .. math::
        \\mathbf{C}_z(\\phi) =
        \\begin{bmatrix}
            \\cos \\phi & -\\sin \\phi & 0 \\\\
            \\sin \\phi  & \\cos \\phi & 0 \\\\
            0 & 0 & 1
        \\end{bmatrix}
    """
    # c = sp.cos(angle_in_radians)
    # s = sp.sin(angle_in_radians)

    # if type(angle_in_radians) == sp.Symbol:
    #     return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    # else:
    c = cos(angle_in_radians)
    s = sin(angle_in_radians)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype="float64")


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
        return SE3(SO3(rotz(x)), np.array([0, 0, 0]))
    if axis == "y":
        return SE3(SO3(roty(x)), np.array([0, 0, 0]))
    if axis == "x":
        return SE3(SO3(rotx(x)), np.array([0, 0, 0]))
    raise Exception("Invalid Axis")

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
