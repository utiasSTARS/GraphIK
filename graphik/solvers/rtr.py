"""Riemannian trust-region optimizer for graphIK.

Faithful Boumal/Absil RTR with Steihaug-Toint truncated CG. The optimizer
never reaches inside the manifold or the cost; everything goes through
the duck-typed manifold methods listed in the design's Manifold contract.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Optional

import numpy as np


class StopReason(IntEnum):
    NEGATIVE_CURVATURE = 0
    EXCEEDED_TR = 1
    REACHED_TARGET_LINEAR = 2
    REACHED_TARGET_SUPERLINEAR = 3
    MAX_INNER_ITER = 4
    MODEL_INCREASED = 5


_TCG_LABEL = {
    StopReason.NEGATIVE_CURVATURE: "negative curvature",
    StopReason.EXCEEDED_TR: "exceeded trust region",
    StopReason.REACHED_TARGET_LINEAR: "reached target residual-kappa (linear)",
    StopReason.REACHED_TARGET_SUPERLINEAR: "reached target residual-theta (superlinear)",
    StopReason.MAX_INNER_ITER: "maximum inner iterations",
    StopReason.MODEL_INCREASED: "model increased",
}


def stopping_criterion_label(reason: StopReason) -> str:
    return _TCG_LABEL[reason]


@dataclass
class RTRResult:
    point: np.ndarray
    cost: float
    iterations: int
    stopping_criterion: str
    time: float
    gradient_norm: float


def trust_regions(
    manifold,
    cost: Callable,
    rgrad: Callable,
    rhess: Callable,
    x0: np.ndarray,
    *,
    max_iterations: int = 3000,
    min_gradient_norm: float = 0.5e-9,
    theta: float = 1.0,
    kappa: float = 0.1,
    rho_prime: float = 0.1,
    rho_regularization: float = 1e3,
    Delta_bar: Optional[float] = None,
    Delta0: Optional[float] = None,
    mininner: int = 1,
    maxinner: Optional[int] = None,
    preconditioner: Optional[Callable] = None,
) -> RTRResult:
    if Delta_bar is None:
        Delta_bar = manifold.typical_dist
    if Delta0 is None:
        Delta0 = Delta_bar / 8
    if maxinner is None:
        maxinner = manifold.dim

    inner = manifold.inner_product
    norm = manifold.norm

    x = x0
    fx = cost(x)
    fgradx = rgrad(x)
    norm_grad = norm(x, fgradx)

    Delta = Delta0
    iteration = 0
    start_time = time.time()
    stopping_criterion = ""

    while True:
        iteration += 1

        eta, Heta, _num_inner, stop_inner = _truncated_cg(
            manifold, x, fgradx, rhess, Delta, theta, kappa, mininner, maxinner,
            preconditioner=preconditioner,
        )

        x_prop = manifold.retraction(x, eta)
        fx_prop = cost(x_prop)

        rhonum = fx - fx_prop
        rhoden = -inner(x, fgradx, eta) - 0.5 * inner(x, eta, Heta)

        rho_reg = max(1.0, abs(fx)) * np.spacing(1) * rho_regularization
        rhonum += rho_reg
        rhoden += rho_reg

        model_decreased = rhoden >= 0
        if rhoden == 0:
            rho = float("nan")
        else:
            rho = rhonum / rhoden

        if rho < 1.0 / 4 or not model_decreased or np.isnan(rho):
            Delta = Delta / 4
        elif rho > 3.0 / 4 and stop_inner in (
            StopReason.NEGATIVE_CURVATURE,
            StopReason.EXCEEDED_TR,
        ):
            Delta = min(2 * Delta, Delta_bar)

        if model_decreased and rho > rho_prime:
            x = x_prop
            fx = fx_prop
            fgradx = rgrad(x)
            norm_grad = norm(x, fgradx)

        if norm_grad < min_gradient_norm:
            stopping_criterion = (
                f"gradient norm {norm_grad:.6e} below tolerance "
                f"{min_gradient_norm:.6e}"
            )
            break
        if iteration >= max_iterations:
            stopping_criterion = (
                f"reached max_iterations ({max_iterations})"
            )
            break

    return RTRResult(
        point=x,
        cost=fx,
        iterations=iteration,
        stopping_criterion=stopping_criterion,
        time=time.time() - start_time,
        gradient_norm=norm_grad,
    )


def _truncated_cg(
    manifold,
    x: np.ndarray,
    fgradx: np.ndarray,
    rhess: Callable,
    Delta: float,
    theta: float,
    kappa: float,
    mininner: int,
    maxinner: int,
    preconditioner: Optional[Callable] = None,
):
    inner = manifold.inner_product

    eta = manifold.zero_vector(x)
    Heta = manifold.zero_vector(x)

    r = fgradx
    e_Pe = 0.0
    z = preconditioner(x, r) if preconditioner is not None else r
    z_r = inner(x, z, r)
    d_Pd = z_r
    delta = -z
    e_Pd = 0.0
    # Convergence test compares the Euclidean residual norm; this is
    # `sqrt(z_r)` only when the preconditioner is identity. Compute
    # the Euclidean norm explicitly so the test is consistent under
    # any preconditioner.
    norm_r0 = float(np.sqrt(inner(x, r, r)))
    model_value = 0.0

    j = 0
    for j in range(int(maxinner)):
        Hdelta = rhess(x, delta)
        d_Hd = inner(x, delta, Hdelta)

        if d_Hd != 0:
            alpha = z_r / d_Hd
            e_Pe_new = e_Pe + 2 * alpha * e_Pd + alpha ** 2 * d_Pd
        else:
            alpha = 0.0
            e_Pe_new = e_Pe

        if d_Hd <= 0 or e_Pe_new >= Delta ** 2:
            tau = (
                -e_Pd + np.sqrt(e_Pd * e_Pd + d_Pd * (Delta ** 2 - e_Pe))
            ) / d_Pd
            eta = eta + tau * delta
            Heta = Heta + tau * Hdelta
            stop = (
                StopReason.NEGATIVE_CURVATURE
                if d_Hd <= 0
                else StopReason.EXCEEDED_TR
            )
            return eta, Heta, j, stop

        e_Pe = e_Pe_new
        new_eta = eta + alpha * delta
        new_Heta = Heta + alpha * Hdelta

        new_model_value = (
            inner(x, new_eta, fgradx) + 0.5 * inner(x, new_eta, new_Heta)
        )
        if new_model_value >= model_value:
            return eta, Heta, j, StopReason.MODEL_INCREASED

        eta = new_eta
        Heta = new_Heta
        model_value = new_model_value

        r = r + alpha * Hdelta
        z = preconditioner(x, r) if preconditioner is not None else r
        norm_r = float(np.sqrt(inner(x, r, r)))

        if j >= mininner and norm_r <= norm_r0 * min(norm_r0 ** theta, kappa):
            stop = (
                StopReason.REACHED_TARGET_LINEAR
                if kappa < norm_r0 ** theta
                else StopReason.REACHED_TARGET_SUPERLINEAR
            )
            return eta, Heta, j, stop

        zold_rold = z_r
        z_r = inner(x, z, r)
        beta = z_r / zold_rold
        delta = -z + beta * delta
        delta = manifold.to_tangent_space(x, delta)

        e_Pd = beta * (e_Pd + alpha * d_Pd)
        d_Pd = z_r + beta * beta * d_Pd

    return eta, Heta, j, StopReason.MAX_INNER_ITER
