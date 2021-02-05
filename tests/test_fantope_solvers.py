import numpy as np
from graphik.solvers.convex_iteration import solve_fantope_closed_form, solve_fantope_sdp, random_psd_matrix


if __name__ == '__main__':
    # Parameters
    N = 4
    d = 2

    # Our PSD matrix
    G = random_psd_matrix(N, d)

    # Two solution methods
    C_closed = solve_fantope_closed_form(G, d)
    C_sdp = solve_fantope_sdp(G, d)
