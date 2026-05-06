import numpy as np
from scipy.sparse import csc_array
from graphik.utils.dgp import incidence_matrix_from_adjacency

def res_left_op_batched(omega, dim, vectorized=False, sparse=False):
    # B is the incidence matrix; implements p_i - p_j for p -> (N, dim)
    B = incidence_matrix_from_adjacency(omega)  # (d N)
    num_points = omega.shape[0]
    num_dist = B.shape[0]

    if vectorized:
        res_vec_all = [np.kron(B[i], np.eye(dim)).reshape(dim, num_points, dim) for i in range(num_dist)]
        operator = np.stack(res_vec_all)  # (d dim N dim)
    else:
        operator = B

    if sparse:
        operator = csc_array(operator)

    return operator

def sum_square_op(omega, dim, vectorized=False, sparse=False, reduced=False):
    B = incidence_matrix_from_adjacency(omega)

    if vectorized:
        B_vec = np.kron(B, np.eye(dim))
        operator = B_vec.T.dot(B_vec)
    else:
        operator = B.T.dot(B)

    if sparse:
        operator = csc_array(operator)

    return operator

def sum_square_op_batched(omega, dim, vectorized=False, sparse=False, flat=False):
    B = incidence_matrix_from_adjacency(omega)  # (d N)
    num_points = omega.shape[0]
    num_dist = B.shape[0]

    if vectorized:
        # d[i] = (p[i] - p[j])**2 = p @ A[i] @ p, batched over i
        res_sq_vec_all = [np.kron(np.outer(B[i], B[i]), np.eye(dim)) for i in range(num_dist)]
        operator = np.stack(res_sq_vec_all)  # (d N*dim N*dim)

        if flat:
            operator = np.ascontiguousarray(operator.reshape(num_dist * num_points * dim, -1))
    else:
        res_sq_all = [np.outer(B[i], B[i]) for i in range(num_dist)]
        operator = np.stack(res_sq_all)

        if flat:
            operator = np.ascontiguousarray(operator.reshape(num_dist * num_points, -1))

    if sparse:
        operator = csc_array(operator)

    return operator

def diff_sum_square_op_batched(omega, dim, vectorized=False, sparse=False, flat=False):
    B = incidence_matrix_from_adjacency(omega)
    num_points = omega.shape[0]
    num_dist = B.shape[0]

    if vectorized:
        res_sq_vec_all = [np.kron(np.outer(B[i], B[i]), np.eye(dim)) for i in range(num_dist)]
        operator = np.stack([M + M.T for M in res_sq_vec_all])

        if flat:
            operator = np.ascontiguousarray(operator.swapaxes(0, 1).reshape(num_points * dim, -1).T)
    else:
        res_sq_all = [np.outer(B.T[i], B.T[i]) for i in range(num_dist)]
        operator = np.stack([M + M.T for M in res_sq_all])

        if flat:
            operator = np.ascontiguousarray(operator.swapaxes(0, 1).reshape(num_points, -1).T)

    if sparse:
        operator = csc_array(operator)

    return operator
