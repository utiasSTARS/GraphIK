"""Single-source EDM loss kernels for the non-SDP IK solvers.

Two backends:
- NumPy dense (default): ``_dense_equality`` and ``_dense_limits`` build an S
  matrix once per Y and reuse it across cost/egrad/ehvp via ``memoize_last``.
- Numba AOT: ``_numba_equality`` and ``_numba_limits`` wrap the kernels
  exported from ``graphik/solvers/costs.py`` (compiled to ``costgrd.so``).

Two public factories:
- ``for_riemannian(D_goal, omega, *, psi_L, psi_U, jit, cache)`` returns
  ``(cost, egrad, ehvp)`` operating on ``(N, dim)`` Y arrays — RTR consumer.
- ``for_minimize(D_goal, omega, *, psi_L, psi_U, dim, jit, cache)`` returns
  ``(cost_and_grad, hessp)`` operating on flat ``Y_flat`` of length
  ``N*dim`` — scipy.optimize.minimize consumer.

Sign convention used everywhere: ``S = ω*(D_goal - D)``, ``cost = ‖S‖²/2``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from graphik.utils import (
    distance_matrix_from_gram,
    distance_matrix_from_pos,
    memoize_last,
)


def _import_costgrd():
    """Lazy-import the AOT-compiled cost kernels. Only invoked when jit=True."""
    try:
        from graphik.solvers.costgrd import (
            jcost, jgrad, jhess, jcost_and_grad,
            lcost, lgrad, lhess, lcost_and_grad,
        )
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "jit=True requires the AOT-compiled costgrd module. "
            "Build it via 'cd graphik/solvers && python costs.py'."
        ) from e
    return (
        jcost, jgrad, jhess, jcost_and_grad,
        lcost, lgrad, lhess, lcost_and_grad,
    )


def _dense_equality(D_goal, omega, cache=True):
    """NumPy-dense (cost, egrad, ehvp) for the equality-only EDM loss.

    Memoizes the (S, S_diag) state per-Y identity when cache=True so RTR's
    inner CG (which calls cost / egrad / ehvp at the same Y) pays the build
    once per outer iteration.
    """
    memo = memoize_last if cache else (lambda f: f)

    @memo
    def state(Y):
        S = omega * (D_goal - distance_matrix_from_pos(Y))
        S_diag = S.copy()
        np.fill_diagonal(S_diag, S_diag.diagonal() - np.sum(S_diag, axis=1))
        return S, S_diag

    def cost(Y):
        S, _ = state(Y)
        return np.linalg.norm(S) ** 2 / 2

    def egrad(Y):
        _, S_diag = state(Y)
        return 4 * S_diag.dot(Y)

    def ehvp(Y, Z):
        _, S_diag = state(Y)
        YZT = Y.dot(Z.T)
        YZT += YZT.T
        dSdZ = -omega * distance_matrix_from_gram(YZT)
        np.fill_diagonal(dSdZ, dSdZ.diagonal() - np.sum(dSdZ, axis=1))
        return 4 * (dSdZ.dot(Y) + S_diag.dot(Z))

    return cost, egrad, ehvp


def _dense_limits(D_goal, omega, psi_L, psi_U, cache=True):
    """NumPy-dense (cost, egrad, ehvp) for EDM loss with ψ_L / ψ_U slack.

    The three constraint slices (equality A0, lower-active A1, upper-active
    A2) enter gradient and Hessian only through a linear adjoint operator,
    which commutes with summation, so we fold them into a single (N, N)
    A_adj matrix. Y_diff is the pairwise-difference tensor that lets ehvp
    express the adjoint action as a batched matmul.
    """
    diff = psi_L != psi_U
    LL = diff * (psi_L > 0)
    UU = diff * (psi_U > 0)

    memo = memoize_last if cache else (lambda f: f)

    @memo
    def state(Y):
        D = distance_matrix_from_pos(Y)
        A0 = omega * (D_goal - D)
        A1 = np.maximum(psi_L - LL * D, 0)
        A2 = -np.maximum(-psi_U + UU * D, 0)
        A_sum = A0 + A1 + A2
        A_adj = A_sum.copy()
        np.fill_diagonal(A_adj, A_adj.diagonal() - np.sum(A_sum, axis=1))
        m4 = -np.where(A1 > 0, 1, 0) * LL
        m5 = -np.where(-A2 > 0, 1, 0) * UU
        m_total = -omega + m4 + m5
        Y_diff = Y[None, :, :] - Y[:, None, :]  # (N, N, d)
        return {
            "A0": A0, "A1": A1, "A2": A2,
            "A_adj": A_adj, "m_total": m_total, "Y_diff": Y_diff,
        }

    def cost(Y):
        s = state(Y)
        return (np.linalg.norm(s["A0"]) ** 2
                + np.linalg.norm(s["A1"]) ** 2
                + np.linalg.norm(s["A2"]) ** 2) / 2

    def egrad(Y):
        s = state(Y)
        return 4 * s["A_adj"].dot(Y)

    def ehvp(Y, Z):
        s = state(Y)
        d_yz = distance_matrix_from_gram(Y.dot(Z.T) + Z.dot(Y.T))
        M = s["m_total"] * d_yz
        adj_M_Y = np.matmul(M[:, None, :], s["Y_diff"]).squeeze(1)
        return 4 * (adj_M_Y + s["A_adj"].dot(Z))

    return cost, egrad, ehvp


def _numba_equality(D_goal, omega):
    """Wraps costgrd.{jcost, jgrad, jhess}. Pure call-through; no Python state."""
    inds = np.nonzero(np.triu(omega))
    jcost, jgrad, jhess, *_ = _import_costgrd()

    def cost(Y):
        return jcost(Y, D_goal, inds)

    def egrad(Y):
        return jgrad(Y, D_goal, inds)

    def ehvp(Y, Z):
        return jhess(Y, Z, D_goal, inds)

    return cost, egrad, ehvp


def _numba_limits(D_goal, omega, psi_L, psi_U):
    """Wraps costgrd.{lcost, lgrad, lhess}. Pure call-through.

    The inds set is the union of the equality, lower-bound, and upper-bound
    triangle masks; the AOT kernel decides per-pair which residual is active.
    """
    diff = psi_L != psi_U
    inds = np.nonzero(
        np.triu(omega) + np.triu(diff * (psi_L > 0)) + np.triu(diff * (psi_U > 0))
    )
    *_, lcost, lgrad, lhess, _ = _import_costgrd()

    def cost(Y):
        return lcost(Y, D_goal, omega, psi_L, psi_U, inds)

    def egrad(Y):
        return lgrad(Y, D_goal, omega, psi_L, psi_U, inds)

    def ehvp(Y, Z):
        return lhess(Y, Z, D_goal, omega, psi_L, psi_U, inds)

    return cost, egrad, ehvp


def for_riemannian(
    D_goal,
    omega,
    *,
    psi_L=None,
    psi_U=None,
    jit=False,
    cache=True,
):
    """Build (cost, egrad, ehvp) closures for an RTR-style consumer.

    All three accept Y of shape (N, dim) and return scalars / (N, dim) arrays.
    State is memoized per-Y identity when ``cache=True``; this matters for
    the dense backend (RTR's inner CG calls cost / egrad / ehvp at the same
    Y), and is a no-op for numba_aot (which is stateless from Python).

    Dispatch key is ``psi_L is None``: if ``psi_L is None``, the equality
    backend is used regardless of ``psi_U`` (callers pass both or neither
    in practice).
    """
    if jit:
        if psi_L is None:
            return _numba_equality(D_goal, omega)
        return _numba_limits(D_goal, omega, psi_L, psi_U)
    if psi_L is None:
        return _dense_equality(D_goal, omega, cache=cache)
    return _dense_limits(D_goal, omega, psi_L, psi_U, cache=cache)


def for_minimize(
    D_goal,
    omega,
    *,
    dim,
    psi_L=None,
    psi_U=None,
    jit=False,
    cache=True,
):
    """Build (cost_and_grad, hessp) closures for scipy.optimize.minimize.

    Both callables accept flat ``Y_flat`` of length ``N*dim`` and return
    flat-shaped outputs (cost as float, grad / HVP as ``(N*dim,)`` arrays).

    A small reshape cache is held at the wrapper boundary so that within a
    single scipy iteration (cost_and_grad followed by hessp at the same
    Y_flat), the same ``(N, dim)`` view is passed to the inner backend —
    which lets the dense backend's ``memoize_last`` state cache hit.
    """
    cost, egrad, ehvp = for_riemannian(
        D_goal, omega, psi_L=psi_L, psi_U=psi_U, jit=jit, cache=cache
    )

    last: list[Any] = [None, None]  # [id(Y_flat), reshaped view]

    def _Y(Y_flat):
        if cache and last[0] == id(Y_flat):
            return last[1]
        Y = Y_flat.reshape(-1, dim)
        if cache:
            last[0], last[1] = id(Y_flat), Y
        return Y

    def cost_and_grad(Y_flat):
        Y = _Y(Y_flat)
        return cost(Y), egrad(Y).ravel()

    def hessp(Y_flat, Z_flat):
        Y = _Y(Y_flat)
        return ehvp(Y, Z_flat.reshape(-1, dim)).ravel()

    return cost_and_grad, hessp
