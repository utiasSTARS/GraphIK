from __future__ import division

import warnings
import time
import numpy as np
import numpy.linalg as la
import numpy.random as rnd
import scipy.linalg as sl
from scipy.linalg.lapack import get_lapack_funcs
from scipy.linalg.decomp_schur import schur
from scipy.linalg import solve_sylvester
from numba import njit

from scipy.linalg import solve_continuous_lyapunov as lyap
# Workaround for SciPy bug: https://github.com/scipy/scipy/pull/8082
# try:
#     from scipy.linalg import solve_continuous_lyapunov as lyap
# except ImportError:
#     from scipy.linalg import solve_lyapunov as lyap

from pymanopt.manifolds.manifold import Manifold

sfunction = lambda x: None

def proj(Y, H):
    # Projection onto the horizontal space
    # return H
    YtY = Y.T.dot(Y)
    AS = Y.T.dot(H) - H.T.dot(Y)
    Omega = lyap(YtY, AS)
    return H - Y.dot(Omega)

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

    An alternative, complete, geometry for positive semidefinite matrices of
    rank k is described in Bonnabel and Sepulchre 2009, "Riemannian Metric
    and Geometric Mean for Positive Semidefinite Matrices of Fixed Rank",
    SIAM Journal on Matrix Analysis and Applications.

    The geometry implemented here is the simplest case of the 2010 paper:
    M. Journee, P.-A. Absil, F. Bach and R. Sepulchre,
    "Low-Rank Optimization on the Cone of Positive Semidefinite Matrices".
    Paper link: http://www.di.ens.fr/~fbach/journee2010_sdp.pdf
    """

    def __init__(self, n, k):
        self._n = n
        self._k = k
        # self.proj_sum = 0

    def __str__(self):
        return "YY' quotient manifold of {:d}x{:d} psd matrices of " "rank {:d}".format(
            self._n, self._n, self._k
        )

    @property
    def dim(self):
        n = self._n
        k = self._k
        return k * n - k * (k - 1) / 2

    @property
    def typicaldist(self):
        return 10 + self._k

    def inner(self, Y, U, V):
        # Euclidean metric on the total space.
        # return float(np.tensordot(U, V)) # VERY SLOW
        return np.einsum("ij,ji->", U, V.T)

    def norm(self, Y, U):
        return la.norm(U, "fro") # SLOW
        # return np.sqrt(np.einsum("ij,ij->", U, U))

    def dist(self, U, V):
        raise NotImplementedError

    def proj(self, Y, H):
        # Projection onto the horizontal space
        return H
        # YtY = Y.T.dot(Y)
        # AS = Y.T.dot(H) - H.T.dot(Y)
        # Omega = lyap(YtY, AS)
        # return H - Y.dot(Omega)

    def egrad2rgrad(self, Y, egrad):
        return egrad

    def ehess2rhess(self, Y, egrad, ehess, U):
        return self.proj(Y, ehess)

    def exp(self, Y, U):
        warnings.warn(
            "Exponential map for symmetric, fixed-rank "
            "manifold not implemented yet. Used retraction instead.",
            RuntimeWarning,
        )
        return self.retr(Y, U)

    def retr(self, Y, U):
        return Y + U

    def log(self, Y, U):
        raise NotImplementedError

    def rand(self):
        return rnd.randn(self._n, self._k)

    def randvec(self, Y):
        H = self.rand()
        P = self.proj(Y, H)
        return self._normalize(P)

    def transp(self, Y, Z, U):
        return self.proj(Z, U)

    def pairmean(self, X, Y):
        raise NotImplementedError

    def _normalize(self, Y):
        return Y / self.norm(None, Y)
