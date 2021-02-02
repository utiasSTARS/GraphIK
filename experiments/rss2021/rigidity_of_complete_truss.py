import numpy as np
from scipy import linalg

if __name__ == '__main__':
    # Creat 4 points with a complete rigid truss between them
    d = 3
    p0 = np.zeros(d)
    p1 = np.array([1., 0., 0.])
    p2 = np.array([0., 1., 0.])
    p3 = np.array([1., 1., 0.])  # Planar quadrilateral (square)
    # p3 = np.array([0., 0., 1.])  # Tetrahedron
    P = np.vstack([p0, p1, p2, p3])
    n = P.shape[0]

    # Form rigidity matrix
    rows = []
    for idx in range(n):
        p_idx = P[idx, :]
        for jdx in range(idx + 1, n):
            p_jdx = P[jdx, :]
            row = np.zeros(d * n)
            row[d * idx:d * (idx + 1)] = p_idx - p_jdx
            row[d * jdx:d * (jdx + 1)] = p_jdx - p_idx
            rows += [row]

    rigidity = np.vstack(rows)

    # Rank is 5
    print("Rank: {:}".format(np.linalg.matrix_rank(rigidity)))

    # Nullspace
    kernel = linalg.null_space(rigidity)
    print("Nullspace: {:}".format(kernel))

    # Rank of the nullspace is 7 (so we are infinitesimally flexible)
    print("Rank of rigidity matrix's nullspace: {:}".format(np.linalg.matrix_rank(kernel)))

    # Visualize the motions
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    x = P[:, 0]
    y = P[:, 1]
    z = P[:, 2]
    u_inds = np.arange(0, d * n, d)
    v_inds = u_inds + 1
    w_inds = v_inds + 1
    for fig_idx in range(kernel.shape[1]):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        u = kernel[u_inds, fig_idx]
        v = kernel[v_inds, fig_idx]
        w = kernel[w_inds, fig_idx]
        ax.scatter(x, y, z)
        ax.quiver(x, y, z, u, v, w, length=1.0, normalize=False)
        plt.grid()
        plt.show()
