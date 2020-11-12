#!/usr/bin/env python3
import numpy as np
import math


def dist_to_gram(D):
    J = np.identity(D.shape[0]) - (1 / (D.shape[0])) * np.ones(D.shape)
    G = -0.5 * J @ D @ J  # Gram matrix
    return G


def factor(A):
    n = A.shape[0]
    (evals, evecs) = np.linalg.eigh(A)
    evals[evals < 0] = 0  # closest SDP matrix
    X = evecs  # np.transpose(evecs)
    sqrootdiag = np.eye(n)
    for i in range(n):
        sqrootdiag[i, i] = math.sqrt(evals[i])
    X = X.dot(sqrootdiag)
    return np.fliplr(X)


## perform classic Multidimensional scaling
def MDS(B, eps=1e-5):
    n = B.shape[0]
    x = factor(B)
    (evals, evecs) = np.linalg.eigh(x)
    K = len(evals[evals > eps])
    if K < n:
        # only first K columns
        x = x[:, 0:K]
    return x


## perform PCA (new version 170309)
def PCA(B, K="None"):
    x = factor(B)
    n = B.shape[0]
    if type(K) == str:
        K = n
    if K < n:
        # only first K columns
        x = x[:, 0:K]
    return x


def linear_projection(P, F, dim):
    S = 0
    I = np.nonzero(F)
    for kdx in range(len(I[0])):
        idx = I[0][kdx]
        jdx = I[1][kdx]
        S += np.outer(P[idx, :] - P[jdx, :], P[idx, :] - P[jdx, :])

    eigval, eigvec = np.linalg.eigh(S)
    return P @ np.fliplr(eigvec)[:, :dim]


def linear_projection_randomized(P, F, dim):
    S = 0
    I = np.nonzero(F)
    for kdx in range(len(I[0])):
        idx = I[0][kdx]
        jdx = I[1][kdx]
        S += np.outer(P[idx, :] - P[jdx, :], P[idx, :] - P[jdx, :])

    eigval, eigvec = np.linalg.eigh(S)
    ev = np.fliplr(eigvec)
    q, r = np.linalg.qr(ev[:, :dim])
    U = q
    # U = q[:, :dim]
    return P @ U


## sample distance matrix
def sample_matrix(lower_limit, upper_limit):
    m, n = lower_limit.shape
    return lower_limit + np.random.rand(m, n) * (upper_limit - lower_limit)
    # return lower_limit + np.random.normal(75.0, 0.25, (m, n)) * (
    #     upper_limit - lower_limit
    # )


def distmat(Y):
    X = np.dot(Y, Y.T)
    H = np.outer(np.ones(X.shape[0]), np.diag(X))
    D = H + H.T - 2 * X
    return D
