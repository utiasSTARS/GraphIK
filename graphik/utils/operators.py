import numpy as np
from scipy.sparse import csc_array
from graphik.utils.dgp import incidence_matrix_from_adjacency

def res_left_op_batched(omega, dim, vectorized=False, sparse=False):
    inds = np.nonzero(np.triu(omega)) # distance indices
    num_points = omega.shape[0] # number of points
    num_dist = len(inds[0]) # number of distances

    # Incidence matrix, implements p_i - p_j for p -> (N dim)
    B = incidence_matrix_from_adjacency(omega) # (d N)

    if vectorized:
        res_vec_all = [np.kron(B[i], np.eye(dim)).reshape(dim,num_points,dim) for i in range(num_dist)] # (dim N*dim)
        operator = np.stack(res_vec_all) # (d dim N dim) -> operator.dot(x) -> (d dim)
    else:
        operator = B

    if sparse:
        operator = csc_array(operator)

    return operator

def sum_square_op(omega, dim, vectorized=False, sparse=False, reduced=False):
    inds = np.nonzero(np.triu(omega)) # distance indices
    num_points = omega.shape[0] # number of points
    num_dist = len(inds[0]) # number of distances

    # Incidence matrix, implements p_i - p_j for p -> (N,dim)
    B = incidence_matrix_from_adjacency(omega)

    if vectorized:
        # Implements p_i - p_j for p -> N*dim
        B_vec = np.kron(B, np.eye(dim))
        operator = B_vec.T.dot(B_vec)
    else:
        operator = B.T.dot(B)

    if sparse:
        operator = csc_array(operator)

    return operator

def sum_square_op_batched(omega, dim, vectorized=False, sparse=False, flat=False):
    inds = np.nonzero(np.triu(omega)) # distance indices
    num_points = omega.shape[0] # number of points
    num_dist = len(inds[0]) # number of distances

    # Incidence matrix, implements p_i - p_j for p -> (N,dim)
    B = incidence_matrix_from_adjacency(omega) # (d N)

    if vectorized:
        # Implements d[i] = (p[i] - p[j])**2 = p @ A[i] @ p
        res_sq_vec_all = [np.kron(np.outer(B[i], B[i]), np.eye(dim)) for i in range(num_dist)]

        # Implements batched d = (p[all_i] - p[all_j])**2
        operator = np.stack(res_sq_vec_all) # (d N*dim N*dim)

        if flat:
            # (d*N*dim N*dim)
            # operator = np.ascontiguousarray(operator.swapaxes(0,1).reshape(num_points*dim, -1).T)
            operator = np.ascontiguousarray(operator.reshape(num_dist*num_points*dim, -1))

        if sparse:
            operator = csc_array(operator)
    else:
        res_sq_all = [np.outer(B[i], B[i]) for i in range(num_dist)]

        # Implements batched d = (p[all_i] - p[all_j])**2
        operator = np.stack(res_sq_all)

        if flat:
            operator = np.ascontiguousarray(operator.reshape(num_dist*num_points,-1))

        if sparse:
            operator = csc_array(operator)

    return operator

def diff_sum_square_op_batched(omega, dim, vectorized=False, sparse=False, flat=False):
    inds = np.nonzero(np.triu(omega)) # distance indices
    num_points = omega.shape[0] # number of points
    num_dist = len(inds[0]) # number of distances

    # Incidence matrix, implements p_i - p_j for p -> (N,dim)
    B = incidence_matrix_from_adjacency(omega)

    if vectorized:
        # Implements d[i] = (p[i] - p[j])**2 = p @ A[i] @ p
        res_sq_vec_all = [np.kron(np.outer(B[i], B[i]), np.eye(dim)) for i in range(num_dist)]

        # Implements batched d = (p[all_i] - p[all_j])**2
        operator = np.stack([res_sq_vec_all[i] + res_sq_vec_all[i].T for i in range(num_dist)])

        if flat:
            operator = np.ascontiguousarray(operator.swapaxes(0,1).reshape(num_points*dim,-1).T)

        if sparse:
            operator = csc_array(operator)
    else:
        # Implements d[i] = (p[i] - p[j])**2 = p @ A[i] @ p
        res_sq_all = [np.outer(B.T[i], B.T[i]) for i in range(num_dist)] # num_points x num_points

        # Implements batched d = (p[all_i] - p[all_j])**2
        operator = np.stack([res_sq_all[i] + res_sq_all[i].T for i in range(num_dist)])

        if flat:
            operator = np.ascontiguousarray(operator.swapaxes(0,1).reshape(num_points,-1).T)

        if sparse:
            operator = csc_array(operator)

    return operator

# # Implements batched d = (p[all_i] - p[all_j])**2
# res_sq_vec_batch = np.stack(res_sq_vec_all)
# res_sq_vec_batch_inds = np.nonzero(res_sq_vec_batch)
# res_sq_vec_batch_data = res_sq_vec_batch[res_sq_vec_batch_inds]
# res_sq_vec_batch_sparse = csc_array(np.ascontiguousarray(res_sq_vec_batch.swapaxes(0,1).reshape(N*dim,-1).T))

# # Implements the gradient for batched d w.r.t p
# diff_sq_vec_batch = -2*res_sq_vec_batch
# diff_sq_vec_batch_inds = np.nonzero(diff_sq_vec_batch)
# diff_sq_vec_batch_data = diff_sq_vec_batch[diff_sq_vec_batch_inds]
# diff_sq_vec_batch_sparse = csc_array(np.ascontiguousarray(diff_sq_vec_batch.swapaxes(0,1).reshape(N*dim,-1).T))
