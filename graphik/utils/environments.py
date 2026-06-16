"""Obstacle environments for benchmarks and examples."""
import numpy as np


def table_environment(height=0.9, width=0.8, n_height=9, n_width=8, obs_inflation=2.):
    """Sphere-approximated table: a tabletop grid plus four legs.

    Returns a list of ``(center, radius)`` pairs suitable for
    ``ProblemGraph.add_spherical_obstacle``.
    """
    # Make the tabletop
    radius = 0.5*height/n_height
    tabletop_obs = [(np.asarray([2*(i+0.5)*radius, 2*(j+0.5)*radius, height+radius]), obs_inflation*radius)
                    for i in range(-n_width//2, n_width//2) for j in range(-n_width//2, n_width//2)]

    leg1 = [(np.asarray([-width/2+radius, -width/2+radius, (2*i+1)*radius]), obs_inflation*radius) for i in range(0, n_height)]
    leg2 = [(np.asarray([-width/2+radius,  width/2-radius, (2*i+1)*radius]), obs_inflation*radius) for i in range(0, n_height)]
    leg3 = [(np.asarray([ width/2-radius, -width/2+radius, (2*i+1)*radius]), obs_inflation*radius) for i in range(0, n_height)]
    leg4 = [(np.asarray([ width/2-radius,  width/2-radius, (2*i+1)*radius]), obs_inflation*radius) for i in range(0, n_height)]

    return tabletop_obs + leg1 + leg2 + leg3 + leg4
