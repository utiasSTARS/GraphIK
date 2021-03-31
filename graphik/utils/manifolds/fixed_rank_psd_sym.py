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

# Workaround for SciPy bug: https://github.com/scipy/scipy/pull/8082
# try:
#     from scipy.linalg import solve_continuous_lyapunov as lyap
# except ImportError:
#     from scipy.linalg import solve_lyapunov as lyap

from pymanopt.manifolds.manifold import Manifold

sfunction = lambda x: None

# def lyap(A, Q):
#     """
#     Solves the continuous Lyapunov equation :math:`AX + XA^H = Q`.
#     Uses the Bartels-Stewart algorithm to find :math:`X`.
#     Parameters
#     ----------
#     a : array_like
#         A square matrix
#     q : array_like
#         Right-hand side square matrix
#     Returns
#     -------
#     x : ndarray
#         Solution to the continuous Lyapunov equation
#     See Also
#     --------
#     Notes
#     -----
#     https://www.manopt.org/reference/manopt/tools/lyapunov_symmetric_eig.html
#     """

#     A = A.T + A / 2
#     D, V = np.linalg.eig(A)
#     M = (V.T.dot(Q)).dot(V)
#     W = D[:,np.newaxis] + D
#     Y = M.dot(np.linalg.pinv(W, hermitian = True))

#     return (V.dot(Y)).dot(V.T)

def lyap(a, q):
    """
    Solves the continuous Lyapunov equation :math:`AX + XA^H = Q`.
    Uses the Bartels-Stewart algorithm to find :math:`X`.
    Parameters
    ----------
    a : array_like
        A square matrix
    q : array_like
        Right-hand side square matrix
    Returns
    -------
    x : ndarray
        Solution to the continuous Lyapunov equation
    See Also
    --------
    solve_discrete_lyapunov : computes the solution to the discrete-time
        Lyapunov equation
    solve_sylvester : computes the solution to the Sylvester equation
    Notes
    -----
    The continuous Lyapunov equation is a special form of the Sylvester
    equation, hence this solver relies on LAPACK routine ?TRSYL.
    .. versionadded:: 0.11.0
    """

    # Compute the Schur decomposition form of a
    # r, u = schur(a, output="real", check_finite=False, overwrite_a=True)
    (gees,) = get_lapack_funcs(("gees",), (a,))
    # t = time.time()
    # result = gees(lambda x: None, a, lwork=-1)
    # print(time.time() - t)
    # lwork = result[-2][0].real.astype(np.int_)
    # result = gees(sfunction, a, lwork=lwork, overwrite_a=True,
    #               sort_t=0)
    result = gees(sfunction, a, overwrite_a=True, sort_t=0)
    r = result[0]
    u = result[-3]
    # Construct f = u'*q*u
    f = u.T.dot(q.dot(u))

    # Call the Sylvester equation solver
    trsyl = get_lapack_funcs("trsyl", (r, f))

    y, scale, info = trsyl(r, r, f, tranb="T")

    y *= scale

    return u.dot(y).dot(u.T)


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
        # return float(np.tensordot(U, V))
        return np.einsum("ij,ji->", U, V.T)

    def norm(self, Y, U):
        return la.norm(U, "fro")
        # return np.sqrt(np.einsum("ij,ij->", U, U))

    def dist(self, U, V):
        raise NotImplementedError

    def proj(self, Y, H):
        # Projection onto the horizontal space
        # t = time.time()
        YtY = Y.T.dot(Y)
        YtH = Y.T.dot(H)
        AS = YtH - YtH.T
        # AS = Y.T.dot(H) - H.T.dot(Y)
        Omega = lyap(YtY, AS)
        # Omega[np.abs(Omega)<1e-16] = 0
        # print(Omega)
        # Omega = solve_sylvester(YtY, YtY, AS.T).T
        # self.proj_sum += time.time() - t
        return H - Y.dot(Omega)

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
