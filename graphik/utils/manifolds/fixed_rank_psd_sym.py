from __future__ import division

import numpy as np
import numpy.random as rnd

from graphik.utils import memoize_last


def _inner_product_np(Y, U, V):
    # Euclidean metric on the total space.
    return np.dot(U.ravel(), V.ravel())


def _norm_np(Y, U):
    return np.linalg.norm(U)


class PSDFixedRank:
    """
    Manifold of n-by-n symmetric positive semidefinite matrices of rank k.

    A point X on the manifold is parameterized as YY^T where Y is a matrix of
    size nxk. As such, X is symmetric, positive semidefinite. We restrict to
    full-rank Y's, such that X has rank exactly k. The point X is numerically
    represented by Y (this is more efficient than working with X, which may
    be big). Tangent vectors are represented as matrices of the same size as
    Y, call them Ydot, so that Xdot = Y Ydot' + Ydot Y. The metric is the
    canonical Euclidean metric on Y.

    Since for any orthogonal Q of size k, it holds that (YQ)(YQ)' = YY',
    we "group" all matrices of the form YQ in an equivalence class. The set
    of equivalence classes is a Riemannian quotient manifold, implemented
    here.

    Notice that this manifold is not complete: if optimization leads Y to be
    rank-deficient, the geometry will break down. Hence, this geometry should
    only be used if it is expected that the points of interest will have rank
    exactly k. Reduce k if that is not the case.

    The geometry implemented here is the simplest case of the 2010 paper:
    M. Journee, P.-A. Absil, F. Bach and R. Sepulchre,
    "Low-Rank Optimization on the Cone of Positive Semidefinite Matrices".
    """

    def __init__(self, n, k, jit=False, cache_projection=False):
        self._n = n
        self._k = k
        self.name = f"YY' quotient manifold of {n}x{n} psd matrices of rank {k}"
        self.dim = int(k * n - k * (k - 1) / 2)
        # Transpose permutation of d² indices, used to skew-symmetrize
        # the projection operator. Depends only on k (fixed), so it's
        # built once at construction.
        self._perm = np.arange(k * k).reshape(k, k).T.ravel()
        # The projection's per-Y operator L is built by ``_build_projection_op``
        # and applied by ``projection``. With ``cache_projection=True`` the
        # builder is memoized on Y identity — useful when projecting many
        # tangent vectors at the same Y (e.g. RTR's inner CG).
        if cache_projection:
            self._projection_op = memoize_last(self._build_projection_op)
        else:
            self._projection_op = self._build_projection_op

        if jit:
            from numba import njit
            self.inner_product = njit(cache=True)(_inner_product_np)
            self.norm = njit(cache=True)(_norm_np)
        else:
            self.inner_product = _inner_product_np
            self.norm = _norm_np

    @property
    def typical_dist(self):
        return 10 + self._k


    @staticmethod
    def _build_lyap_matrix(X):
        # X = Y^T Y. Returns the (dim*dim) x (dim*dim) Lyapunov matrix
        # whose linear system gives the quotient correction Omega.
        dim = X.shape[0]
        if dim == 3:
            A = np.asarray([[X[0,0] + X[0,0], X[0,1] , X[0,2], X[1,0], 0, 0, X[2,0], 0, 0],
                            [X[1,0], X[1,1] + X[0,0], X[1,2], 0, X[1,0], 0, 0, X[2,0], 0],
                            [X[2,0], X[2,1], X[2,2] + X[0,0], 0, 0, X[1,0], 0, 0, X[2,0]],
                            [X[0,1], 0, 0, X[0,0] + X[1,1], X[0,1] , X[0,2], X[2,1], 0, 0],
                            [0, X[0,1], 0, X[1,0], X[1,1] + X[1,1], X[1,2], 0, X[2,1], 0],
                            [0, 0, X[0,1], X[2,0], X[2,1], X[2,2] + X[1,1], 0, 0, X[2,1]],
                            [X[0,2], 0, 0, X[1,2], 0, 0, X[0,0] + X[2,2], X[0,1] , X[0,2]],
                            [0, X[0,2], 0, 0, X[1,2], 0, X[1,0], X[1,1] + X[2,2], X[1,2]],
                            [0, 0, X[0,2], 0, 0, X[1,2], X[2,0], X[2,1], X[2,2] + X[2,2]]])
        else:  # dim == 2
            A = np.asarray([[X[0,0] + X[0,0], X[0,1], X[0,1], 0],
                            [X[1,0], X[0,1] + X[0,0], 0, X[0,1]],
                            [X[0,1], 0, X[0,0] + X[1,1], X[0,1]],
                            [0, X[0,1], X[1,0], X[1,1]+ X[1,1]]])
        return A

    @staticmethod
    def _kron_with_eye(A, d):
        # Equivalent to np.kron(A, np.eye(d)) but ~8× faster on small d:
        # the result is block-diagonal, so we just stride-write A into
        # the d block-diagonals instead of going through einsum.
        p, q = A.shape
        out = np.zeros((p * d, q * d))
        for b in range(d):
            out[b::d, b::d] = A
        return out

    def _build_projection_op(self, Y):
        # Build the (Nd × Nd) operator L such that the projection of Z
        # onto T_Y M is Z - L.dot(vec(Z)). The quotient correction is
        # Y @ Omega where A_lyap @ vec(Omega) = vec(Y^T Z - Z^T Y) and
        # A_lyap depends only on Y^T Y, so L is purely a function of Y.
        #   L = M_skew @ (Y^T ⊗ I_d),  M_skew = M - M[:, perm],
        #   M = (Y ⊗ I_d) @ A_inv.
        d = Y.shape[1]
        A_inv = np.linalg.inv(self._build_lyap_matrix(Y.T.dot(Y)))
        M = self._kron_with_eye(Y, d).dot(A_inv)
        return (M - M[:, self._perm]).dot(self._kron_with_eye(Y.T, d))

    def projection(self, Y, Z):
        L = self._projection_op(Y)
        return Z - L.dot(Z.ravel()).reshape(Y.shape)

    # rtr.py calls to_tangent_space inside tCG; same logic as projection.
    to_tangent_space = projection

    def retraction(self, Y, U):
        return Y + U


    def random_point(self):
        return rnd.randn(self._n, self._k)

    def random_tangent_vector(self, Y):
        H = self.random_point()
        P = self.projection(Y, H)
        return self._normalize(P)

    def zero_vector(self, Y):
        return np.zeros((self._n, self._k))

    def _normalize(self, Y):
        return Y / self.norm(Y, Y)
