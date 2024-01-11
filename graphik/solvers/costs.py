import numpy as np
from numba import jit
from numba.pycc import CC

cc = CC("costgrd")

@cc.export("jcost", "f8(f8[:,:],f8[:,:],UniTuple(u8[:],2))")
def jcost(Y, D_goal, inds):
    cost = 0
    dim = Y.shape[1]
    for (idx, jdx) in zip(*inds):
        nrm = 0
        for kdx in range(dim):
            nrm += (Y[idx, kdx] - Y[jdx, kdx]) ** 2
        cost += 2 * (D_goal[idx, jdx] - nrm) ** 2
    return 0.5 * cost


@cc.export("jgrad", "f8[:,:](f8[:,:],f8[:,:],UniTuple(u8[:],2))")
def jgrad(Y, D_goal, inds):
    num_el = Y.shape[0]
    dim = Y.shape[1]
    grad = np.zeros((num_el, dim))
    for (idx, jdx) in zip(*inds):
        nrm = 0
        for kdx in range(dim):
            nrm += (Y[idx, kdx] - Y[jdx, kdx]) ** 2
        for kdx in range(dim):
            grad[idx, kdx] += (
                -4 * (D_goal[idx, jdx] - nrm) * (Y[idx, kdx] - Y[jdx, kdx])
            )
            grad[jdx, kdx] += (
                -4 * (D_goal[jdx, idx] - nrm) * (Y[jdx, kdx] - Y[idx, kdx])
            )
    return 0.5 * grad


@cc.export("jhess", "f8[:,:](f8[:,:],f8[:,:],f8[:,:],UniTuple(u8[:],2))")
def jhess(Y, w, D_goal, inds):
    num_el = Y.shape[0]
    dim = Y.shape[1]
    hess = np.zeros((num_el, dim))
    for (idx, jdx) in zip(*inds):
        nrm = 0
        sc = 0
        for kdx in range(dim):
            sc += (Y[idx, kdx] - Y[jdx, kdx]) * (w[idx, kdx] - w[jdx, kdx])
            nrm += (Y[idx, kdx] - Y[jdx, kdx]) ** 2
        for kdx in range(dim):
            hess[idx, kdx] += 4 * (
                2 * sc * (Y[idx, kdx] - Y[jdx, kdx])
                + (nrm - D_goal[idx, jdx]) * (w[idx, kdx] - w[jdx, kdx])
            )
            hess[jdx, kdx] += 4 * (
                2 * sc * (Y[jdx, kdx] - Y[idx, kdx])
                + (nrm - D_goal[jdx, idx]) * (w[jdx, kdx] - w[idx, kdx])
            )
    return 0.5 * hess

@cc.export("jcost_and_grad", "Tuple((f8, f8[:,:]))(f8[:,:],f8[:,:],UniTuple(u8[:],2))")
def jcost_and_grad(Y, D_goal, inds):
    cost = 0
    dim = Y.shape[1]
    grad = np.zeros((Y.shape[0], dim))
    for (idx, jdx) in zip(*inds):
        nrm = 0
        for kdx in range(dim):
            nrm += (Y[idx, kdx] - Y[jdx, kdx]) ** 2
        for kdx in range(dim):
            grad[idx, kdx] += (
                -4 * (D_goal[idx, jdx] - nrm) * (Y[idx, kdx] - Y[jdx, kdx])
            )
            grad[jdx, kdx] += (
                -4 * (D_goal[jdx, idx] - nrm) * (Y[jdx, kdx] - Y[idx, kdx])
            )
        cost += 2 * (D_goal[idx, jdx] - nrm) ** 2
    return 0.5*cost, 0.5* grad

@cc.export("lcost", "f8(f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],UniTuple(u8[:],2))")
def lcost(Y, D_goal, omega, psi_L, psi_U, inds):
    cost = 0
    dim = Y.shape[1]
    for (idx, jdx) in zip(*inds):
        nrm = 0
        for kdx in range(dim):
            nrm += (Y[idx, kdx] - Y[jdx, kdx]) ** 2
        if omega[idx, jdx]>0:
            cost += (D_goal[idx, jdx] - nrm) ** 2
        if psi_L[idx, jdx]>0:
            cost += max((psi_L[idx, jdx] - nrm), 0) ** 2
        if psi_U[idx, jdx]>0:
            cost += max((-psi_U[idx, jdx] + nrm), 0) ** 2
    return cost

@cc.export(
    "lgrad", "f8[:,:](f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],UniTuple(u8[:],2))"
)
def lgrad(Y, D_goal, omega, psi_L, psi_U, inds):
    num_el = Y.shape[0]
    dim = Y.shape[1]
    grad = np.zeros((num_el, dim))
    for (idx, jdx) in zip(*inds):
        nrm = 0
        for kdx in range(dim):
            nrm += (Y[idx, kdx] - Y[jdx, kdx]) ** 2
        if omega[idx, jdx]:
            for kdx in range(dim): # NOTE inside to avoid needless triggering
                a = (nrm - D_goal[idx, jdx]) * (Y[idx, kdx] - Y[jdx, kdx])
                grad[idx, kdx] += a
                grad[jdx, kdx] += -a
        if psi_L[idx, jdx]:
            if max(psi_L[idx, jdx] - nrm, 0) > 0:
                for kdx in range(dim):
                    a = (nrm - psi_L[idx, jdx]) * (Y[idx, kdx] - Y[jdx, kdx])
                    grad[idx, kdx] += a
                    grad[jdx, kdx] += -a
        if psi_U[idx, jdx]:
            if max(-psi_U[idx, jdx] + nrm, 0) > 0:
                for kdx in range(dim):
                    a = (nrm - psi_U[idx, jdx]) * (Y[idx, kdx] - Y[jdx, kdx])
                    grad[idx, kdx] += a
                    grad[jdx, kdx] += -a
    return 2*grad

@cc.export("lcost_and_grad", "Tuple((f8, f8[:,:]))(f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],UniTuple(u8[:],2))")
def lcost_and_grad(Y, D_goal, omega, psi_L, psi_U, inds):
    cost = 0
    dim = Y.shape[1]
    num_el = Y.shape[0]
    grad = np.zeros((num_el, dim))

    for (idx, jdx) in zip(*inds):
        nrm = 0
        for kdx in range(dim):
            nrm += (Y[idx, kdx] - Y[jdx, kdx]) ** 2
        if omega[idx, jdx]>0:
            cost += 2 * (D_goal[idx, jdx] - nrm) ** 2
            for kdx in range(dim):
                grad[idx, kdx] += (
                    4 * (nrm - D_goal[idx, jdx]) * (Y[idx, kdx] - Y[jdx, kdx])
                )
                grad[jdx, kdx] += (
                    4 * (nrm - D_goal[jdx, idx]) * (Y[jdx, kdx] - Y[idx, kdx])
                )
        if psi_L[idx, jdx]>0:
            cost += 2 * max((psi_L[idx, jdx] - nrm), 0) ** 2
            if max(psi_L[idx, jdx] - nrm, 0) > 0:
                for kdx in range(dim):
                    grad[idx, kdx] += (
                        4 * (nrm - psi_L[idx, jdx]) * (Y[idx, kdx] - Y[jdx, kdx])
                    )
                    grad[jdx, kdx] += (
                        4 * (nrm - psi_L[jdx, idx]) * (Y[jdx, kdx] - Y[idx, kdx])
                    )
        if psi_U[idx, jdx]>0:
            cost += 2 * max((-psi_U[idx, jdx] + nrm), 0) ** 2
            if max(-psi_U[idx, jdx] + nrm, 0) > 0:
                for kdx in range(dim):
                    grad[idx, kdx] += (
                        4
                        * (nrm - psi_U[idx, jdx])
                        * (Y[idx, kdx] - Y[jdx, kdx])  # might be wrong
                    )
                    grad[jdx, kdx] += (
                        4
                        * (nrm - psi_U[jdx, idx])
                        * (Y[jdx, kdx] - Y[idx, kdx])  # might be wrong
                    )
    return 0.5*cost, 0.5*grad


# @cc.export(
#     "lhess",
#     "f8[:,:](f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],UniTuple(u8[:],2))",
# )
# def lhess(Y, w, D_goal, omega, psi_L, psi_U, inds):
#     num_el = Y.shape[0]
#     dim = Y.shape[1]
#     hess = np.zeros((num_el, dim))
#     for (idx, jdx) in zip(*inds):
#         nrm = 0
#         sc = 0
#         for kdx in range(dim):
#             nrm += (Y[idx, kdx] - Y[jdx, kdx]) ** 2
#             sc += (Y[idx, kdx] - Y[jdx, kdx]) * (w[idx, kdx] - w[jdx, kdx])
#         if omega[idx, jdx]:
#             for kdx in range(dim):
#                 hess[idx, kdx] += (
#                     2 * sc * (Y[idx, kdx] - Y[jdx, kdx])
#                     + (nrm - D_goal[idx, jdx]) * (w[idx, kdx] - w[jdx, kdx])
#                 )
#                 hess[jdx, kdx] += (
#                     2 * sc * (Y[jdx, kdx] - Y[idx, kdx])
#                     + (nrm - D_goal[jdx, idx]) * (w[jdx, kdx] - w[idx, kdx])
#                 )
#         if psi_L[idx, jdx]:
#             if max(psi_L[idx, jdx] - nrm, 0) > 0:
#                 for kdx in range(dim):
#                     hess[idx, kdx] += (
#                         2 * sc * (Y[idx, kdx] - Y[jdx, kdx])
#                         + (nrm - psi_L[idx, jdx]) * (w[idx, kdx] - w[jdx, kdx])
#                     )
#                     hess[jdx, kdx] += (
#                         2 * sc * (Y[jdx, kdx] - Y[idx, kdx])
#                         + (nrm - psi_L[jdx, idx]) * (w[jdx, kdx] - w[idx, kdx])
#                     )
#         if psi_U[idx, jdx]:
#             if max(-psi_U[idx, jdx] + nrm, 0) > 0:
#                 for kdx in range(dim):
#                     hess[idx, kdx] += (
#                         2 * sc * (Y[idx, kdx] - Y[jdx, kdx])
#                         + (nrm - psi_U[idx, jdx]) * (w[idx, kdx] - w[jdx, kdx])
#                     )
#                     hess[jdx, kdx] += (
#                         2 * sc * (Y[jdx, kdx] - Y[idx, kdx])
#                         + (nrm - psi_U[jdx, idx]) * (w[jdx, kdx] - w[idx, kdx])
#                     )

#     return 2 * hess

@cc.export(
    "lhess",
    "f8[:,:](f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],UniTuple(u8[:],2))",
)
def lhess(Y, w, D_goal, omega, psi_L, psi_U, inds):
    num_el = Y.shape[0]
    dim = Y.shape[1]
    hess = np.zeros((num_el, dim))
    for (idx, jdx) in zip(*inds):
        nrm = 0
        sc = 0
        for kdx in range(dim):
            nrm += (Y[idx, kdx] - Y[jdx, kdx]) ** 2
            sc += (Y[idx, kdx] - Y[jdx, kdx]) * (w[idx, kdx] - w[jdx, kdx])
        if omega[idx, jdx]:
            for kdx in range(dim):
                a = 2 * sc * (Y[idx, kdx] - Y[jdx, kdx])
                b = (nrm - D_goal[idx, jdx]) * (w[idx, kdx] - w[jdx, kdx])
                c = a + b
                hess[idx, kdx] += c
                hess[jdx, kdx] += -c
        if psi_L[idx, jdx] and max(psi_L[idx, jdx] - nrm, 0) > 0:
            for kdx in range(dim):
                a = 2 * sc * (Y[idx, kdx] - Y[jdx, kdx])
                b = (nrm - psi_L[idx, jdx]) * (w[idx, kdx] - w[jdx, kdx])
                c = a + b
                hess[idx, kdx] += c
                hess[jdx, kdx] += -c
        if psi_U[idx, jdx] and max(-psi_U[idx, jdx] + nrm, 0) > 0:
            for kdx in range(dim):
                a = 2 * sc * (Y[idx, kdx] - Y[jdx, kdx])
                b = (nrm - psi_U[idx, jdx]) * (w[idx, kdx] - w[jdx, kdx])
                c = a + b
                hess[idx, kdx] += c
                hess[jdx, kdx] += -c

    return 2 * hess
if __name__ == "__main__":
    cc.compile()
