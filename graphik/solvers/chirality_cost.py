"""Signed-volume penalty term for chirality-aware distance-geometric IK.

For a set of 4-tuples of point indices with target signs and per-tuple gaps
``eps``, computes::

    L(Y) = weight * sum_t max(0, eps_t - s_t * V_t(Y))**2

where ``V_t(Y)`` is the signed volume of the tetrahedron formed by the four
rows of ``Y`` selected by tuple ``t``. This is the one-sided "wrong-handed"
penalty: it is zero whenever the tetrahedron's signed volume agrees with the
reference sign by more than ``eps_t``, and grows quadratically as the volume
drifts toward zero or flips. The Hessian is not provided — the trust-region
solver still uses the distance-loss Hessian as its quadratic model and the
penalty enters only through the gradient, which is exact.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def chirality_cost_grad(
    Y: NDArray,
    tuples: NDArray,
    signs: NDArray,
    eps: NDArray,
    weight: float,
) -> tuple[float, NDArray]:
    """Compute penalty value and gradient w.r.t. ``Y``.

    Parameters
    ----------
    Y : (N, dim) float
        Point coordinates. ``dim`` must be 3.
    tuples : (K, 4) int
        Row indices of the four tetrahedron corners per chiral tuple.
    signs : (K,) float
        Target sign of each signed volume, ``+1`` or ``-1``.
    eps : (K,) float
        Per-tuple activation gap; penalty is zero when ``s_t * V_t > eps_t``.
    weight : float
        Overall scaling.
    """
    if tuples.shape[0] == 0:
        return 0.0, np.zeros_like(Y)

    P = Y[tuples]                            # (K, 4, 3)
    a = P[:, 0]
    e1 = P[:, 1] - a
    e2 = P[:, 2] - a
    e3 = P[:, 3] - a

    V = np.einsum("ki,ki->k", np.cross(e1, e2), e3) / 6.0
    r = np.maximum(0.0, eps - signs * V)
    cost = weight * float(np.dot(r, r))

    grad = np.zeros_like(Y)
    active = r > 0
    if not np.any(active):
        return cost, grad

    e1a, e2a, e3a = e1[active], e2[active], e3[active]
    coef = (-2.0 * weight * signs[active] * r[active])[:, None]   # (Ka, 1)

    dV_db = np.cross(e2a, e3a) / 6.0
    dV_dc = np.cross(e3a, e1a) / 6.0
    dV_dd = np.cross(e1a, e2a) / 6.0
    dV_da = -(dV_db + dV_dc + dV_dd)

    idx_active = tuples[active]
    np.add.at(grad, idx_active[:, 0], coef * dV_da)
    np.add.at(grad, idx_active[:, 1], coef * dV_db)
    np.add.at(grad, idx_active[:, 2], coef * dV_dc)
    np.add.at(grad, idx_active[:, 3], coef * dV_dd)
    return cost, grad
