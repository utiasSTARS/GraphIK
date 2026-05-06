import numpy as np

##################################################
# Primitive rotation helpers
##################################################

def Rx(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def Ry(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def Rz(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def R2(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


##################################################
# Planar joints (return 3×3 SE(2) homogeneous matrices)
##################################################

def dh_to_se2(a: float, theta: float) -> np.ndarray:
    """Transform a single set of DH parameters into a 3x3 SE2 matrix."""
    R = R2(theta)
    t = R @ np.array([a, 0.0])
    T = np.eye(3)
    T[:2, :2] = R
    T[:2, 2] = t
    return T


def fk_2d(a: list, theta: list, q: list) -> np.ndarray:
    """Forward kinematics from arrays of link lengths and angles. Returns 3x3 SE(2)."""
    if len(a) > 1:
        return dh_to_se2(a[0], theta[0] + q[0]) @ fk_2d(a[1:], theta[1:], q[1:])
    return dh_to_se2(a[0], theta[0] + q[0])


def fk_tree_2d(a: list, theta: list, q: list, path_indices: list) -> np.ndarray:
    """Forward kinematics along a path through a tree. Returns 3x3 SE(2)."""
    idx = path_indices[0]
    if len(path_indices) > 1:
        return dh_to_se2(a[idx], theta[idx] + q[idx]) @ fk_tree_2d(
            a, theta, q, path_indices[1:]
        )
    return dh_to_se2(a[idx], theta[idx] + q[idx])


##################################################
# 3D single-axis transforms (return 4×4 SE(3) homogeneous matrices)
##################################################

def trans_axis(t: float, axis: str = "z") -> np.ndarray:
    T = np.eye(4)
    if axis == "z":
        T[:3, 3] = np.array([0.0, 0.0, t])
    elif axis == "y":
        T[:3, 3] = np.array([0.0, t, 0.0])
    elif axis == "x":
        T[:3, 3] = np.array([t, 0.0, 0.0])
    else:
        raise Exception("Invalid Axis")
    return T


def rot_axis(theta: float, axis: str = "z") -> np.ndarray:
    T = np.eye(4)
    if axis == "z":
        T[:3, :3] = Rz(theta)
    elif axis == "y":
        T[:3, :3] = Ry(theta)
    elif axis == "x":
        T[:3, :3] = Rx(theta)
    else:
        raise Exception("Invalid Axis")
    return T


def dh_to_se3(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Single set of DH parameters → 4x4 SE(3)."""
    TransX = trans_axis(a, "x")
    RotX = rot_axis(alpha, "x")
    TransZ = trans_axis(d, "z")
    RotZ = rot_axis(theta, "z")
    return TransZ @ RotZ @ TransX @ RotX


def modified_dh_to_se3(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Single set of modified DH parameters → 4x4 SE(3)."""
    TransX = trans_axis(a, "x")
    RotX = rot_axis(alpha, "x")
    TransZ = trans_axis(d, "z")
    RotZ = rot_axis(theta, "z")
    return TransX @ RotX @ TransZ @ RotZ


def fk_3d(a: list, alpha: list, d: list, theta: list) -> np.ndarray:
    out = dh_to_se3(a[0], alpha[0], d[0], theta[0])
    for idx in range(1, len(a)):
        out = out @ dh_to_se3(a[idx], alpha[idx], d[idx], theta[idx])
    return out


def modified_fk_3d(a: list, alpha: list, d: list, theta: list) -> np.ndarray:
    out = modified_dh_to_se3(a[0], alpha[0], d[0], theta[0])
    for idx in range(1, len(a)):
        out = out @ modified_dh_to_se3(a[idx], alpha[idx], d[idx], theta[idx])
    return out
