import graphik
import trimesh
import numpy as np
from itertools import combinations
from graphik.utils.roboturdf import RobotURDF


def make_scene(
    robot: RobotURDF,
    scene=None,
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

    if scene is None:
        scene = trimesh.Scene()

    if with_robot:
        robot.urdf.update_cfg(cfg)
        tm_scene = robot.urdf.scene
        for geom_name, tm_geom in tm_scene.geometry.items():
            pose, _ = tm_scene.graph.get(geom_name)
            geom = tm_geom.copy()
            if transparency is not None:
                _apply_transparency(geom, transparency)
            scene.add_geometry(
                geom, transform=pose, node_name=f"robot_{geom_name}"
            )

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
        for idx, e in enumerate(dense_edge_indices):
            cyl = _create_edge_cylinder_mesh(Ts[e[0]], Ts[e[1]])
            # None means the cylinder has zero height (duplicate Ts?)
            if cyl is not None:
                scene.add_geometry(cyl, transform=np.eye(4), node_name=f"edge_{idx}")
    return scene


def _apply_transparency(tm_mesh, transparency):
    """Set the alpha channel of a trimesh's face colors. transparency in [0, 1].

    Promotes TextureVisuals to ColorVisuals first because trimesh's
    TextureVisuals does not expose face_colors (texture-mapped COLLADA links
    from real URDFs wrap as TextureVisuals, not ColorVisuals). This loses the
    diffuse-texture appearance — flat-shaded under transparency — but is the
    only way to get a working alpha channel without bypassing the visual layer.
    """
    alpha = int(np.clip(transparency, 0.0, 1.0) * 255)
    if isinstance(tm_mesh, trimesh.Trimesh):
        if not isinstance(tm_mesh.visual, trimesh.visual.color.ColorVisuals):
            base = tm_mesh.visual.to_color()
            # to_color() on a TextureVisuals returns a ColorVisuals with a
            # single-RGBA vertex_colors array (the diffuse color); face_colors
            # expansion then index-errors. Tile the diffuse RGBA per-face
            # explicitly so face_colors is well-formed.
            base_rgba = np.asarray(base.vertex_colors).reshape(-1)[:4]
            face_colors = np.tile(base_rgba, (len(tm_mesh.faces), 1))
            tm_mesh.visual = trimesh.visual.color.ColorVisuals(
                mesh=tm_mesh, face_colors=face_colors
            )
        colors = tm_mesh.visual.face_colors.copy()
        colors[:, 3] = alpha
        tm_mesh.visual.face_colors = colors


def _create_edge_cylinder_mesh(T_i, T_j, radius=0.005):
    """
    Creates a cylinder that connects the 'nodes' at T_i and T_j.

    Parameters
    ----------
    T_i, T_j : SE3
        SE3 representing nodes between which the cylinder will be drawn

    Returns
    -------
    cyl : trimesh.Trimesh or None
        None if the segment is shorter than 1 mm (duplicate Ts).
    """
    # Generate each segment
    seg = np.zeros((2, 3))
    seg[0] = T_i[:3, 3]
    seg[1] = T_j[:3, 3]

    # Check that the cylinder has non-negligible size
    if np.linalg.norm(seg[1] - seg[0]) < 0.001:
        return None

    # Create a gray cylinder with the original (gray=0.1, alpha=0.98) coloring
    cyl = trimesh.creation.cylinder(radius=radius, segment=seg)
    gray = int(0.1 * 255)
    cyl.visual.face_colors = [gray, gray, gray, int(0.98 * 255)]
    return cyl


def view_dae(dae: str, T_zero: list, scene=None, return_scene_only=False, colour=None):
    if scene is None:
        scene = trimesh.Scene()

    frame_obj = trimesh.load(dae)
    # trimesh.load returns either a Scene or a Trimesh; normalize to a list of Trimesh.
    if isinstance(frame_obj, trimesh.Scene):
        meshes = list(frame_obj.geometry.values())
    elif isinstance(frame_obj, trimesh.Trimesh):
        meshes = [frame_obj]
    else:
        meshes = list(frame_obj.dump())

    if colour is not None:
        rgba_table = {
            "red":   [255,   0,   0, 255],
            "green": [  0, 255,   0, 255],
            "blue":  [  0,   0, 255, 255],
        }
        if colour not in rgba_table:
            raise ValueError(f"colour {colour!r} not implemented")
        rgba = rgba_table[colour]
        for tm in meshes:
            tm.visual.face_colors = rgba

    for ti, T in enumerate(T_zero):
        for mi, tm in enumerate(meshes):
            scene.add_geometry(
                tm, transform=T, node_name=f"dae_{id(tm)}_{ti}_{mi}"
            )

    if return_scene_only:
        return scene
    else:
        scene.show()
        return scene


def plot_balls_from_points(
    points: np.ndarray, scene=None, return_scene_only=False, colour=None
):
    """
    Plot red balls at each point in the nx3 array points.

    Parameters
    ----------
    points : np.ndarray
        nx3 array of points to plot the balls at
    scene : trimesh.Scene
        The scene to add the balls to. If scene=None, then a new scene will be
        created.
    return_scene_only : bool
        If True, returns the scene without opening a viewer. If False, opens
        the trimesh viewer and returns the scene.

    Returns
    -------
    scene : trimesh.Scene
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
    scene.show()
