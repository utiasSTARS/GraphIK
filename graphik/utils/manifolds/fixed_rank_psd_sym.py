from __future__ import division

import warnings
import numpy as np
import numpy.random as rnd
from numba import njit

from pymanopt.manifolds.manifold import Manifold


class PSDFixedRank(Manifold):
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

    def __init__(self, n, k):
        self._n = n
        self._k = k
        name = f"YY' quotient manifold of {n}x{n} psd matrices of rank {k}"
        dimension = int(k * n - k * (k - 1) / 2)
        super().__init__(name, dimension)

    @property
    def typical_dist(self):
        return 10 + self._k

    @staticmethod
    @njit(cache=True)
    def inner_product(Y, U, V):
        # Euclidean metric on the total space.
        return np.dot(U.ravel(), V.ravel())

    @staticmethod
    @njit(cache=True)
    def norm(Y, U):
        return np.linalg.norm(U)

    def dist(self, U, V):
        raise NotImplementedError

    @staticmethod
    @njit(cache=True)
    def projection(Y, Z):
        dim = Y.shape[1]
        X = Y.T.dot(Y)
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
        if dim == 2:
            A = np.asarray([[X[0,0] + X[0,0], X[0,1], X[0,1], 0],
                            [X[1,0], X[0,1] + X[0,0], 0, X[0,1]],
                            [X[0,1], 0, X[0,0] + X[1,1], X[0,1]],
                            [0, X[0,1], X[1,0], X[1,1]+ X[1,1]]])
        C = np.dot(Y.T,Z)- np.dot(Z.T,Y)
        Omega = np.linalg.solve(A,C.ravel()).reshape(dim,dim)
        return Z - Y.dot(Omega)

    # pymanopt 2.x calls to_tangent_space inside tCG; alias to projection per upstream.
    to_tangent_space = projection

    def euclidean_to_riemannian_gradient(self, Y, egrad):
        return egrad

    def euclidean_to_riemannian_hessian(self, Y, egrad, ehess, U):
        return self.projection(Y, ehess)

    def exp(self, Y, U):
        warnings.warn(
            "Exponential map for symmetric, fixed-rank "
            "manifold not implemented yet. Used retraction instead.",
            RuntimeWarning,
        )
        return self.retraction(Y, U)

    def retraction(self, Y, U):
        return Y + U

    def log(self, Y, U):
        raise NotImplementedError

    def random_point(self):
        return rnd.randn(self._n, self._k)

    def random_tangent_vector(self, Y):
        H = self.random_point()
        P = self.projection(Y, H)
        return self._normalize(P)

    def transport(self, Y, Z, U):
        return self.projection(Z, U)

    def pair_mean(self, X, Y):
        raise NotImplementedError

    def zero_vector(self, Y):
        return np.zeros((self._n, self._k))

    def _normalize(self, Y):
        return Y / self.norm(Y, Y)
