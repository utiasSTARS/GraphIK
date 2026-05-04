# tests/test_pymlg_conventions.py
"""Pin PyMLG's Log/Exp/adjoint conventions for graphIK.

This file is the canonical reference for what PyMLG does. graphIK code in
graphik/ relies on the conventions asserted here. If PyMLG ever changes
behavior, these tests fail loudly and the implementer must update both this
file and any dependent graphik/ code together.
"""
import numpy as np
import pytest

pymlg_numpy = pytest.importorskip("pymlg.numpy")
SE2 = pymlg_numpy.SE2
SE3 = pymlg_numpy.SE3
SO2 = pymlg_numpy.SO2
SO3 = pymlg_numpy.SO3


def test_se3_log_rotation_first():
    """SE3.Log returns [omega; v] — rotation first, translation last.

    PyMLG empirical: SE3.Log(pure_translation([1,2,3])) yields
    [0, 0, 0, 1, 2, 3]. graphIK twist constructions in robot_revolute.py
    (Task 5) must match this layout: np.hstack((omega, np.cross(-omega, q))).
    """
    T = np.eye(4)
    T[:3, 3] = np.array([1.0, 2.0, 3.0])
    xi = SE3.Log(T)
    assert xi.shape == (6,) or xi.shape == (6, 1), f"unexpected shape {xi.shape}"
    xi = xi.ravel()
    np.testing.assert_allclose(xi[:3], 0.0, atol=1e-12)
    np.testing.assert_allclose(xi[3:], [1.0, 2.0, 3.0], atol=1e-12)


def test_so3_log_is_rotation_vector():
    """SO3.Log returns axis-angle (rotation vector): axis * angle."""
    theta = 0.5
    R = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                  [np.sin(theta),  np.cos(theta), 0.0],
                  [0.0, 0.0, 1.0]])
    phi = SO3.Log(R).ravel()
    np.testing.assert_allclose(phi, [0.0, 0.0, theta], atol=1e-12)


def test_se3_exp_log_round_trip():
    rng = np.random.default_rng(42)
    for _ in range(10):
        xi = rng.normal(size=6) * 0.3
        T = SE3.Exp(xi)
        xi_back = SE3.Log(T).ravel()
        np.testing.assert_allclose(xi_back, xi, atol=1e-10)


def test_se3_inverse_round_trip():
    rng = np.random.default_rng(0)
    xi = rng.normal(size=6) * 0.5
    T = SE3.Exp(xi)
    np.testing.assert_allclose(SE3.inverse(T) @ T, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(T @ SE3.inverse(T), np.eye(4), atol=1e-12)


def test_se3_adjoint_shape_and_action():
    """SE3.adjoint(T) is 6×6; Ad(T) maps body twist to spatial twist:
       Exp(Ad(T) xi) == T @ Exp(xi) @ inv(T).
    """
    rng = np.random.default_rng(1)
    T = SE3.Exp(rng.normal(size=6) * 0.3)
    Ad = SE3.adjoint(T)
    assert Ad.shape == (6, 6)
    xi = rng.normal(size=6) * 0.1
    lhs = SE3.Exp(Ad @ xi)
    rhs = T @ SE3.Exp(xi) @ SE3.inverse(T)
    np.testing.assert_allclose(lhs, rhs, atol=1e-9)


def test_se2_log_angle_first():
    """SE2.Log returns [theta; t_x; t_y] — angle first, translation last.

    PyMLG empirical: SE2.Log(pure_translation([1,2])) yields [0, 1, 2].
    graphIK twist construction in robot_planar.py (Task 4) must select
    [omega_z, v_x, v_y] to match.
    """
    T = np.eye(3)
    T[:2, 2] = np.array([1.0, 2.0])
    xi = SE2.Log(T).ravel()
    assert xi.shape == (3,)
    np.testing.assert_allclose(xi[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(xi[1:], [1.0, 2.0], atol=1e-12)


def test_se2_exp_log_round_trip():
    rng = np.random.default_rng(2)
    for _ in range(10):
        xi = rng.normal(size=3) * 0.3
        T = SE2.Exp(xi)
        xi_back = SE2.Log(T).ravel()
        np.testing.assert_allclose(xi_back, xi, atol=1e-10)


def test_so2_exp_log_round_trip():
    """SO2.Log returns a scalar (not an array); wrap with np.atleast_1d."""
    for theta in [-1.5, -0.3, 0.0, 0.3, 1.5]:
        R = SO2.Exp(np.array([theta]))
        assert R.shape == (2, 2)
        log_val = np.atleast_1d(SO2.Log(R)).ravel()
        np.testing.assert_allclose(log_val, [theta], atol=1e-12)


def test_se3_left_jacobian_inv_at_origin_is_identity():
    """left_jacobian_inv(0) = I_6. Used by joint_angle_solver."""
    J = SE3.left_jacobian_inv(np.zeros(6))
    np.testing.assert_allclose(J, np.eye(6), atol=1e-12)


def test_se2_left_jacobian_inv_exists():
    """LocalSolver.gen_grad_ee uses SE2.left_jacobian_inv when dim==2.
    If PyMLG doesn't expose it, Task 9 must use lambda x: np.eye(3) fallback
    (matching gen_cost_and_grad_ee's existing pattern in the original code).
    """
    assert hasattr(SE2, "left_jacobian_inv"), (
        "PyMLG SE2 has no left_jacobian_inv — joint_angle_solver Task 9 "
        "must use lambda x: np.eye(3) fallback for the dim==2 branch of "
        "gen_grad_ee"
    )
    J = SE2.left_jacobian_inv(np.zeros(3))
    np.testing.assert_allclose(J, np.eye(3), atol=1e-12)


def test_so3_wedge_skew_symmetric():
    """SO3.wedge(v) is the skew-symmetric matrix of v."""
    v = np.array([0.1, 0.2, 0.3])
    W = SO3.wedge(v)
    assert W.shape == (3, 3)
    np.testing.assert_allclose(W + W.T, 0.0, atol=1e-12)
    # action on a vector: wedge(v) @ u == cross(v, u)
    u = np.array([0.4, 0.5, 0.6])
    np.testing.assert_allclose(W @ u, np.cross(v, u), atol=1e-12)
