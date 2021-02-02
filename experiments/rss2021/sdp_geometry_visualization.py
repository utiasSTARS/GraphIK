import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    # Define SDP variables
    x2, xy, y2, x, y, s2 = sp.symbols('x2, xy, y2, x, y, s2')
    # Goal pose
    g = np.array([1., 1.])
    # Form into a symmetric matrix
    Z = np.array([[x2, xy, x],
                  [xy, y2, y],
                  [x, y, s2]])
    Z = sp.Matrix(Z)

    # SDP cone boundary's equation (equals zero)
    sdp_boundary = sp.det(Z)

    # Linear constraints
    lin_constraints = [sp.Equality(s2, 1), sp.Equality(x2 + y2, 1), sp.Equality(x2 - 2*(x + y) + y2, -1)]
    constraints = lin_constraints + [sp.Equality(sdp_boundary, 0)]

    # Solve the system
    # solution = sp.solve_poly_system(lin_constraints + [sp.det(Z)], x2, xy, y2, x, y, s2)

    # Substitute in the linear constraints to reduce variables
    boundary_curve = sdp_boundary.subs(s2, 1)
    boundary_curve = boundary_curve.subs(x2, 1-y2)
    boundary_curve = boundary_curve.subs(x, 1-y)  # Reduce the second linear constraint with the first to get this
    boundary_curve = boundary_curve.simplify()

    # Plot the boundary
    n = 30
    xy_domain = np.linspace(-1., 1., n)
    y_domain = xy_domain
    y2_domain = np.linspace(0., 2., n)

    boundary_tol = 1e-2
    interior_xy = []
    interior_y = []
    interior_y2 = []
    determinant_values = []
    for xy_val in xy_domain:
        for y_val in y_domain:
            for y2_val in y2_domain:
                # if np.abs(boundary_curve.subs(xy, xy_val).subs(y, y_val).subs(y2, y2_val)) <= boundary_tol:
                det_val = boundary_curve.subs(xy, xy_val).subs(y, y_val).subs(y2, y2_val)
                determinant_values.append(det_val)
                if det_val > 0.:
                    interior_xy.append(xy_val)
                    interior_y.append(y_val)
                    interior_y2.append(y2_val)

    # TODO: add obstacle, do heatmap, plot actual solutions, and use SDP constraint, NOT determinant!
    # TODO: eventually, add convex iteration procedure, see how it does on this toy problem (nuclear norm prob. solves) 

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.scatter(interior_xy, interior_y, interior_y2)
    # plt.xlabel('$xy$')
    # plt.ylabel('$y$')
    # # ax.zlabel('$y2$')
    # plt.grid()
    # plt.show()

    # Plot a heatmap

    # Plot the boundary curve
    # y2_solve = [sp.solve_poly_system([sp.Equality(boundary_curve.subs(xy, xy_domain[idx]).subs(y, y_domain[idx]), 0)], y2) for idx in range(len(xy_domain))]
    #
    # curve1 = [val[0][0].as_real_imag()[0] for val in y2_solve] # Take only the real part
    # curve2 = [val[1][0].as_real_imag()[0] for val in y2_solve]
    #
    # true_sols_xy = [0., 0.]
    # true_sols_y  = [0., 1.]
    # true_sols_y2 = [0., 1.]
    #
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # plt.plot(xy_domain, y_domain, curve1, 'r-')
    # plt.plot(xy_domain, y_domain, curve2, 'b--')
    # plt.scatter(true_sols_xy, true_sols_y, true_sols_y2)
    # plt.xlabel('$xy$')
    # plt.ylabel('$y$')
    # plt.xlabel('$xy$')
    # plt.grid()
    # plt.show()