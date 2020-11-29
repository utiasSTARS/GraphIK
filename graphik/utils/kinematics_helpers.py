import numpy as np
import sympy as sp
from scipy.optimize import fsolve

from liegroups.numpy import SO2, SE2, SO3, SE3


def skew(x):
    """
    Creates a skew symmetric matrix from vector x
    """
    X = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    return X


def sph_to_se3(alpha: float, d: float, theta: float) -> SE3:
    """Transform a single set of DH parameters into an SE3 matrix
    :param a: displacement along x
    :param alpha: rotation about x
    :param d: translation along new z
    :param theta: rotation around new z
    :returns: SE3 matrix
    :rtype: lie.SE3Matrix
    """

    RX = SO3.rotx(alpha)
    RotX = SE3(RX, np.zeros(3))
    TransZ = SE3(SO3.identity(), np.array([0, 0, d]))
    RZ = SO3.rotz(theta)
    RotZ = SE3(RZ, np.zeros(3))
    return RotZ.dot(RotX.dot(TransZ))


def sph_to_se3_symb(alpha: float, d: float, theta: float) -> SE3:
    """Transform a single set of DH parameters into an SE3 matrix
    :param a: displacement along x
    :param alpha: rotation about x
    :param d: translation along new z
    :param theta: rotation around new z
    :returns: SE3 matrix
    :rtype: lie.SE3Matrix
    """

    RX = SO3(rotx(alpha))
    RotX = SE3(RX, np.zeros(3))
    TransZ = SE3(SO3.identity(), np.array([0, 0, d]))
    RZ = SO3(rotz(theta))
    RotZ = SE3(RZ, np.zeros(3))
    return RotZ.dot(RotX.dot(TransZ))


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


def extract_relative_angle_Z(T_source: np.ndarray, T_target: np.ndarray) -> float:
    """

    :param T_source: source transformation matrix
    :param T_target: target transformation matrix that transforms e.e. coords to base coords
    :return:
    """
    T_rel = np.linalg.inv(T_source).dot(T_target)
    return np.arctan2(T_rel[1, 0], T_rel[0, 0])


def inverse_modified_dh_frame(p: np.ndarray, a: float, d: float, alpha) -> float:
    """
    Compute the least-squares DH frame IK when given the other three DH parameters (a, d, alpha) and the origin of the
    next frame.

    :param p: Origin point for the link we are trying to localize.
    :param a:
    :param d:
    :param alpha:
    :return: float representing the DH parameter theta for this link
    """
    pass


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
    c = sp.cos(angle_in_radians)
    s = sp.sin(angle_in_radians)

    if type(angle_in_radians) == sp.Symbol:
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    else:
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
    c = sp.cos(angle_in_radians)
    s = sp.sin(angle_in_radians)

    if type(angle_in_radians) == sp.Symbol:
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    else:
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
    c = sp.cos(angle_in_radians)
    s = sp.sin(angle_in_radians)

    if type(angle_in_radians) == sp.Symbol:
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    else:
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype="float64")


def rotZ_symb(angle_in_radians):
    R = SO3(rotz(angle_in_radians))
    return SE3(R, np.array([0.0, 0.0, 0.0]))


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
