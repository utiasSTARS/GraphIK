import graphik
import numpy as np

from graphik.utils.roboturdf import RobotURDF, plot_balls_from_points
from matplotlib import pyplot as plt
from numpy import pi
from numpy.linalg import norm

from graphik.graphs import ProblemGraphRevolute

# from graphik.robots.revolute import Revolute3dChain
from graphik.robots import RobotRevolute
from graphik.solvers.riemannian_solver import RiemannianSolver

from numpy.testing import assert_array_less
from graphik.utils.dgp import (
    adjacency_matrix_from_graph,
    pos_from_graph,
    graph_from_pos,
)
from graphik.utils.utils import (
    best_fit_transform,
    list_to_variable_dict,
    dZ,
    safe_arccos,
)

if __name__ == "__main__":
    np.random.seed(21)

    n = 6
    angular_limits = np.minimum(np.random.rand(n) * (pi / 2) + pi / 2, pi)
    ub = angular_limits
    lb = -angular_limits

    fname = graphik.__path__[0] + "/robots/urdfs/ur10_mod.urdf"
    urdf_robot = RobotURDF(fname)
    robot = urdf_robot.make_Revolute3d(ub, lb)  # make the Revolute class from a URDF
    graph = ProblemGraphRevolute(robot)

    # scale = 0.75
    # radius = 0.4
    # bias = np.asarray([0, 0, 0.1237])
    # obstacles = [
    #     (scale * np.asarray([1, 1, 0]) + bias, radius),
    #     (scale * np.asarray([1, -1, 0]) + bias, radius),
    #     (scale * np.asarray([-1, 1, 0]) + bias, radius),
    #     (scale * np.asarray([-1, -1, 0]) + bias, radius),
    #     (scale * np.asarray([0, 0, 1]) + bias, radius),
    #     (scale * np.asarray([0, 0, -1]) + bias, radius),
    # ]
    # scale = 0.5
    # radius = 0.45
    # bias = np.asarray([0, 0, 0.1237])
    # obstacles = [
    #     (scale * np.asarray([-1, 1, 1]) + bias, radius),
    #     (scale * np.asarray([-1, 1, -1]) + bias, radius),
    #     (scale * np.asarray([1, -1, 1]) + bias, radius),
    #     (scale * np.asarray([1, -1, -1]) + bias, radius),
    #     (scale * np.asarray([1, 1, 1]) + bias, radius),
    #     (scale * np.asarray([1, 1, -1]) + bias, radius),
    #     (scale * np.asarray([-1, -1, 1]) + bias, radius),
    #     (scale * np.asarray([-1, -1, -1]) + bias, radius),
    # ]
    # phi = (1 + np.sqrt(5)) / 2
    # bias = np.asarray([0, 0, 0.1237])
    # scale = 0.5
    # radius = 0.5
    # obstacles = [
    #     (scale * np.asarray([0, 1, phi]) + bias, radius),
    #     (scale * np.asarray([0, 1, -phi]) + bias, radius),
    #     (scale * np.asarray([0, -1, -phi]) + bias, radius),
    #     (scale * np.asarray([0, -1, phi]) + bias, radius),
    #     (scale * np.asarray([1, phi, 0]) + bias, radius),
    #     (scale * np.asarray([1, -phi, 0]) + bias, radius),
    #     (scale * np.asarray([-1, -phi, 0]) + bias, radius),
    #     (scale * np.asarray([-1, phi, 0]) + bias, radius),
    #     (scale * np.asarray([phi, 0, 1]) + bias, radius),
    #     (scale * np.asarray([-phi, 0, 1]) + bias, radius),
    #     (scale * np.asarray([-phi, 0, -1]) + bias, radius),
    #     (scale * np.asarray([phi, 0, -1]) + bias, radius),
    # ]
    # for idx, obs in enumerate(obstacles):
    #     graph.add_spherical_obstacle(f"o{idx}", obs[0], obs[1])

    q_zero = list_to_variable_dict([0 , pi/6, pi/2,0,pi/4,0])
    # q_zero = robot.random_configuration()
    urdf_robot.make_scene(
        q=q_zero, with_frames=False, with_balls=True, with_edges=True, with_robot=True, transparency=0.7
    )
    scene = urdf_robot.scene
    T = urdf_robot.extract_T_zero_from_URDF()
    import trimesh
    import pyrender
    for idx, obs in enumerate(obstacles):
        _obs_sph = trimesh.primitives.Sphere(radius=obs[1])
        _obs_sph.visual.vertex_colors = [0,0,1,0.1]
        # colors, texcoords, material = pyrender.Mesh._get_trimesh_props(_obs_sph)
        # material.baseColorFactor = np.array([0.0, 0.0, 1.0, 0.2])
        obs_sph = pyrender.Mesh.from_trimesh(_obs_sph, wireframe=False)
        pose = np.eye(4)
        pose[:3,3] = obs[0]
        scene.add(obs_sph, pose=pose)
        # plot_balls_from_points(obs[0][np.newaxis],scene,"green")

    v = pyrender.Viewer(scene, use_raymond_lighting=True,)
