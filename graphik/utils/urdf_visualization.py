import graphik
import trimesh
import numpy as np
import pyrender
from itertools import combinations
from graphik.utils.roboturdf import RobotURDF

def make_scene(
    robot: RobotURDF,
    scene = None,
    q=None,
    with_frames=True,
    with_balls=True,
    with_robot=True,
    with_edges=True,
    transparency=None,
):

    # Build a full cfg covering every actuated joint so update_cfg behaves
    # statelessly (same compensation as RobotURDF.extract_T_zero_from_URDF;
    # yourdfpy's update_cfg only mutates joints whose names appear in cfg).
    partial_cfg = robot.map_to_urdf_ind(q) if q is not None else {}
    cfg = {j.name: partial_cfg.get(j.name, 0.0) for j in robot.urdf.actuated_joints}
    if with_robot:
        robot.urdf.update_cfg(cfg)
        tm_scene = robot.urdf.scene
        if scene is None:
            scene = pyrender.Scene()
        for geom_name, tm_geom in tm_scene.geometry.items():
            pose, _ = tm_scene.graph.get(geom_name)
            pr_mesh = pyrender.Mesh.from_trimesh(tm_geom)
            if transparency is not None:
                for prim in pr_mesh.primitives:
                    base = prim.material.baseColorFactor
                    if base is None:
                        continue
                    prim.material.baseColorFactor = (
                        float(base[0]), float(base[1]), float(base[2]), float(transparency),
                    )
            scene.add(pr_mesh, pose=pose)

    Ts_dict = robot.extract_T_zero_from_URDF(q=q)
    Ts = []
    for T in Ts_dict:
        T_zero = Ts_dict[T]
        Ts.append(T_zero)

    if with_frames:
        path = graphik.robots.__path__[0] + "/urdfs/meshes/frame.dae"
        scene = view_dae(path, Ts, scene=scene, return_scene_only=True)

    if with_balls:
        path = graphik.robots.__path__[0] + "/urdfs/meshes/redball.dae"
        scene = view_dae(path, Ts, scene=scene, return_scene_only=True)

    if with_edges:
        # Generate dense tuples that connect all joints
        dense_edge_indices = list(combinations(range(len(Ts)), r=2))

        # Draw cylinders between each indices
        for e in dense_edge_indices:

            m = _create_edge_cylinder_mesh(Ts[e[0]], Ts[e[1]])
            # None means the cylinder has zero height (duplicate Ts?)
            if m is not None:
                scene.add(m, pose=np.eye(4))
    return scene

def _create_edge_cylinder_mesh(T_i, T_j, radius=0.005):
    """
    Creates a cylinder that connects the 'nodes' at T_i and T_j

    Parameters
    ----------
    T_i, T_j : SE3
        SE3 representing nodes between which the cylinder will be drawn

    Returns
    -------
    m : pyrender Mesh
    """
    # Generate each segment
    seg = np.zeros((2, 3))
    seg[0] = T_i[:3, 3]
    seg[1] = T_j[:3, 3]

    # Check that the cylinder has non-negligible size
    if np.linalg.norm(seg[1] - seg[0]) < 0.001:
        return None

    # Create a gray cylinder
    cyl = trimesh.creation.cylinder(radius=radius, segment=seg)
    gray = 0.1
    cyl.visual.vertex_colors = [gray, gray, gray, 0.98]

    # Render it!
    m = pyrender.Mesh.from_trimesh(cyl)
    return m

def view_dae(dae: str, T_zero: list, scene=None, return_scene_only=False, colour=None):
    if scene is None:
        scene = pyrender.Scene()

    frame_tm = trimesh.load(dae)
    material = None
    if colour is not None:
        for tm in frame_tm.dump():
            colors, texcoords, material = pyrender.Mesh._get_trimesh_props(tm)
            if colour == "red":
                material.baseColorFactor = np.array([1.0, 0.0, 0.0, 1.0])
            elif colour == "green":
                material.baseColorFactor = np.array([0.0, 1.0, 0.0, 1.0])
            elif colour == "blue":
                material.baseColorFactor = np.array([0.0, 0.0, 1.0, 1.0])
            else:
                raise ("colour not implemented")

    # frame_tm = trimesh.load('graphik/robots/urdfs/meshes/frame.dae')
    meshes = pyrender.Mesh.from_trimesh(frame_tm.dump(), material=material)

    for T in T_zero:
        scene.add(meshes, pose=T)
    if return_scene_only:
        return scene
    else:
        v = pyrender.Viewer(scene, use_raymond_lighting=True)
        return scene


def plot_balls_from_points(
    points: np.array, scene=None, return_scene_only=False, colour=None
):
    """
    Plot red balls at each point in the nx3 array points
    Parameters
    ----------
    points : np.array
        nx3 array of points to plot the balls
    scene : pyrender.Scene
        The scene to add the balls to. If scene=None, then a new scene will be
        created
    return_scene_only : bool
        If True, will only return the scene and not plot the points. If False,
        will plot the points and return the scene.

    Returns
    -------
        scene

    """
    dae = graphik.robots.__path__[0] + "/urdfs/meshes/redball.dae"
    n, _ = points.shape
    T = []
    for i in range(n):
        T_id = np.eye(4)
        T_id[0:3, 3] = points[i, :]
        T.append(T_id)

    scene = view_dae(
        dae, T, scene=scene, return_scene_only=return_scene_only, colour=colour
    )
    return scene

def visualize(
    robot: RobotURDF,
    q=None,
    with_frames=True,
    with_balls=True,
    with_robot=True,
    with_edges=True,
    transparency=None,
):
    scene = make_scene(
        robot=robot,
        q=q,
        with_frames=with_frames,
        with_balls=with_balls,
        with_robot=with_robot,
        with_edges=with_edges,
        transparency=transparency,
    )

    v = pyrender.Viewer(scene, use_raymond_lighting=True)
