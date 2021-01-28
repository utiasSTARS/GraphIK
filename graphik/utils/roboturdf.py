from graphik.utils.utils import flatten
from urdfpy import URDF
from liegroups import SE3, SO3
import numpy as np
import trimesh
import pyrender
from graphik.graphs.graph_base import RobotRevoluteGraph
from graphik.robots.robot_base import RobotRevolute
import graphik
from operator import itemgetter


class RobotURDF(object):
    def __init__(self, fname):
        self.fname = fname
        self.urdf = URDF.load(fname)

        self.urdf_ind_to_q, self.q_to_urdf_ind = self.joint_map()
        self.n_q_joints = len(self.q_to_urdf_ind)
        self.n_urdf_joints = len(self.urdf_ind_to_q)

        self.ee_joints = None
        self.T_zero = self.extract_T_zero_from_URDF(frame="joint")
        self.scene = None

        self.parents = None
        # self.parents = self.get_parents()

    def joint_map(self):
        urdf_ind_to_q = {}
        q_to_urdf_ind = {}
        q_to_names = {}
        q_ind = 1
        label = "p{0}"
        # for j, joint in enumerate(self.urdf.joints):
        for joint in self.urdf.actuated_joints:
            j = self.urdf.joints.index(joint)
            urdf_ind_to_q[j] = label.format(q_ind)
            q_to_urdf_ind[label.format(q_ind)] = j
            # urdf_ind_to_q[j] = q_ind
            # q_to_urdf_ind[q_ind] = j
            q_ind += 1

        return urdf_ind_to_q, q_to_urdf_ind

    def find_first_joint(self):
        """
        Finds the first joint who's parent link is 'world' link. ASSUMES URDF has a link named 'world'!
        """
        world_link = self.find_link_by_name("world")
        joint = self.find_actuated_joints_with_parent_link(world_link)
        return joint[0]

    def find_actuated_joints_with_parent_link(self, link):
        parent_joints = []
        for joint in self.urdf.joints:
            if joint.parent == link.name:
                if not (joint in self.urdf.actuated_joints):
                    joints = self.find_actuated_joints_with_parent_link(
                        self.find_link_by_name(joint.child)
                    )
                    parent_joints.extend(joints)
                else:
                    parent_joints.append(joint)

        return parent_joints

    def find_joints_with_parent_link(self, link):
        parent_joints = []
        for joint in self.urdf.joints:
            if joint.parent == link.name:
                parent_joints.append(joint)

        return parent_joints

    def find_joints_actuated_child_joints(self, joint):
        child_link = self.find_link_by_name(joint.child)
        children_joints = self.find_actuated_joints_with_parent_link(child_link)
        return children_joints

    def find_joints_child_joints_from_list(self, joint, joints):
        child_link = self.find_link_by_name(joint.child)
        children_joints = []
        for j in joints:
            parent_link = self.find_link_by_name(j.parent)
            if child_link == parent_link:
                children_joints.append(j)
        return children_joints

    def get_parents(self, joints):
        base_joint = self.find_first_joint()
        if not (base_joint in joints):
            raise ("Base joint not in joints")

        label_base = "p{0}"
        # parents = {'p0': [label_base.format(joints.index(base_joint))]}
        parents = {}

        for joint in joints:
            children = self.find_joints_child_joints_from_list(joint, joints)
            if children == []:
                child_labels = []
            else:
                child_labels = [label_base.format(joints.index(cj)) for cj in children]
            parent_label = label_base.format(joints.index(joint))
            parents[parent_label] = child_labels

        self.parents = parents

    def actuated_joint_index(self, joint):
        try:
            return self.urdf.actuated_joints.index(joint)
        except ValueError:
            raise ("joint not an actuated joint")

    def find_link_by_name(self, name):
        for link in self.urdf.links:
            if link.name == name:
                return link
        return None

    def find_joint_by_name(self, name):
        for joint in self.urdf.joints:
            if joint.name == name:
                return joint
        return None

    def extract_T_zero_from_URDF(self, q=None, frame="joint"):
        """
        T is located at the joint's origin, the rotation such that
        z_hat points along the joint axis.
        """
        if q is not None:
            urdf_q = self.map_to_urdf_ind(q)
            cfg = urdf_q
        else:
            cfg = {}
        fk = self.urdf.link_fk(cfg=cfg)
        T = {}
        for joint in self.urdf.actuated_joints:
            # get child link frame
            child_link = self.find_link_by_name(joint.child)
            T_link = SE3.from_matrix(fk[child_link])
            if frame == "joint":
                # URDF frames are aligned with the links
                # An additional rotation needs to be applied
                # to align the Z axis with the joint frame
                joint_axis = joint.axis
                T_joint_axis = get_T_from_joint_axis(joint_axis)
                T_joint = np.dot(T_link.as_matrix(), T_joint_axis.inv().as_matrix())
                T[joint] = SE3.from_matrix(T_joint)
            else:
                T[joint] = T_link

        ee_joints = self.find_end_effector_joints()
        for ee_joint in ee_joints:
            ee_link = self.find_link_by_name(ee_joint.child)
            T[ee_joint] = SE3.from_matrix(fk[ee_link])

        return T

    def find_end_effector_joints(self):
        """
        Finds end-effector joints. Assumes that the end effector frame has
        a fixed joint.

        Returns
        -------

        ee_joints : list
            List of urdfpy joints that correspond to the End Effectors

        """
        ee_joints = []

        for joint in self.urdf.joints:
            child_joints = self.find_joints_actuated_child_joints(joint)
            if child_joints == []:
                # child_link = self.find_link_by_name(joint.child)
                # ee_joint = self.find_joints_with_parent_link(child_link)
                # if ee_joint == []:
                #    # No fixed joint for ee
                #    raise("There is an end effector joint that isn't a fixed frame")
                # ee_joints.extend(ee_joint)
                ee_joints.append(joint)

        self.ee_joints = ee_joints

        return ee_joints

    def map_to_urdf_ind(self, q):
        """
        maps a dictionary so the keys (joint ind) in q map to the correct
        joint indices in URDF representation
        """

        q_keys = list(q.keys())
        urdf_ind = itemgetter(*q_keys)(self.q_to_urdf_ind)
        names = [self.urdf.joints[i].name for i in urdf_ind]
        # urdf_q = dict(zip(urdf_ind, list(q.values())))
        urdf_q = dict(zip(names, list(q.values())))

        return urdf_q

    def visualize(
        self,
        q=None,
        with_frames=True,
        with_balls=True,
        with_robot=True,
        transparency=None,
    ):
        self.make_scene(
            q=q,
            with_frames=with_frames,
            with_balls=with_balls,
            with_robot=with_robot,
            transparency=transparency,
        )

        v = pyrender.Viewer(self.scene, use_raymond_lighting=True)

    def make_scene(
        self,
        q=None,
        with_frames=True,
        with_balls=True,
        with_robot=True,
        transparency=None,
    ):

        if q is not None:
            urdf_q = self.map_to_urdf_ind(q)
            # cfg = list(urdf_q.values())
            cfg = urdf_q
        else:
            cfg = {}
        if with_robot:
            robot_scene = self.urdf.show(
                cfg=cfg, return_scene=True, transparency=transparency
            )
            if self.scene is None:
                self.scene = robot_scene
            else:
                for node in robot_scene.get_nodes():
                    self.scene.add_node(node)

        Ts_dict = self.extract_T_zero_from_URDF(q=q)
        Ts = []
        for T in Ts_dict:
            T_zero = Ts_dict[T]
            Ts.append(T_zero)

        if with_frames:
            path = graphik.robots.__path__[0] + "/urdfs/meshes/frame.dae"
            self.scene = view_dae(path, Ts, scene=self.scene, return_scene_only=True)

        if with_balls:
            path = graphik.robots.__path__[0] + "/urdfs/meshes/redball.dae"
            self.scene = view_dae(path, Ts, scene=self.scene, return_scene_only=True)

    def joint_limits(self):
        ub = {}
        lb = {}

        for joint in self.urdf.actuated_joints:
            ubi = np.clip(joint.limit.upper, -np.pi, np.pi)
            lbi = np.clip(joint.limit.lower, -np.pi, np.pi)
            label = "p{0}"
            joint_label = label.format(self.actuated_joint_index(joint))
            ub[joint_label] = ubi
            lb[joint_label] = lbi

        return ub, lb

    def get_graphik_labels(self, joints):
        """
        Assigned joint labels according to the p{i}, where i is the joint
        index. The first joint has label p0
        Parameters
        ----------
        joints : list
            list of urdfpy joints

        Returns
        -------
        labels : list
            list of the labels
        """
        n = len(joints)

        label = "p{0}"
        labels = [label.format(i) for i in range(n)]
        return labels

    def make_Revolute3d(self, ub, lb):
        # if all the child lists have len 1, then chain, otherwise tree
        params = {}

        # assign parents
        joints = list(self.T_zero.keys())
        self.get_parents(joints)
        params["parents"] = self.parents

        # Assign Transforms
        T_labels = self.get_graphik_labels(joints)
        T_zero = dict(zip(T_labels, self.T_zero.values()))
        T0 = T_zero["p0"]
        for key, val in T_zero.items():
            T_zero[key] = T0.inv().dot(val)
        # T_zero['root'] = SE3.identity()
        params["T_zero"] = T_zero

        l = 0
        for cl in self.parents.values():
            l += len(cl)
        if l == len(self.parents.keys()) - 1:
            # number of children == number of joints
            # ub, lb = self.joint_limits()
            params["ub"] = ub
            params["lb"] = lb
            # return Revolute3dChain(params)
            return RobotRevolute(params)
        else:
            # return Revolute3dTree(params)
            return RobotRevolute(params)


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
        scene.add(meshes, pose=T.as_matrix())
    if return_scene_only:
        return scene
    else:
        v = pyrender.Viewer(scene, use_raymond_lighting=True)
        return scene


def skew(x):
    x = flatten(x)
    X = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    return X


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
        T_zero = SE3.from_matrix(T_id)
        T.append(T_zero)

    scene = view_dae(
        dae, T, scene=scene, return_scene_only=return_scene_only, colour=colour
    )
    return scene


def get_T_from_joint_axis(axis: str, switch=False):
    """
    Take in the axis string from urdf "X X X" and return the rotation matrix
    assoicated with that axis
    """
    norm = np.linalg.norm
    z_hat = np.array([0, 0, 1])

    if switch:
        sgn = -1.0
    else:
        sgn = 1.0

    if all(np.isclose(axis, -z_hat)):
        R = SO3.rotx(np.pi).as_matrix()
    elif not all(np.isclose(axis, z_hat)):
        rot_axis = np.cross(axis, z_hat)
        # rot_axis = np.cross(z_hat, axis)
        ang = -np.arcsin(norm(rot_axis) / (norm(axis) * norm(z_hat)))
        rot_axis = normalize(rot_axis)

        rot_axis = rot_axis.reshape(3, 1)

        R = (
            np.eye(3) * np.cos(ang)
            + (1 - np.cos(ang)) * np.dot(rot_axis, rot_axis.transpose())
            - np.sin(ang) * skew(rot_axis)
        )
    else:
        R = np.eye(3)

    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = np.zeros(3)
    T = SE3.from_matrix(T)

    return T


def normalize(x):
    return x / np.linalg.norm(x)


def load_ur10(limits=None):
    fname = graphik.__path__[0] + "/robots/urdfs/ur10_mod.urdf"
    urdf_robot = RobotURDF(fname)
    n = 6
    if limits is None:
        ub = np.ones(n) * np.pi
        lb = -ub
    else:
        lb = limits[0]
        ub = limits[1]
    robot = urdf_robot.make_Revolute3d(ub, lb)  # make the Revolute class from a URDF
    # print(robot.structure.nodes())
    graph = RobotRevoluteGraph(robot)
    return robot, graph


def load_truncated_ur10(n: int):
    """
    Produce a robot and graph representing the first n links of a UR10.
    """
    a_full = [0, -0.612, -0.5723, 0, 0, 0]
    d_full = [0.1273, 0, 0, 0.1639, 0.1157, 0.0922]
    al_full = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]
    th_full = [0, 0, 0, 0, 0, 0]
    a = a_full[0:n]
    d = d_full[0:n]
    al = al_full[0:n]
    th = th_full[0:n]
    ub = (np.pi) * np.ones(n)
    lb = -ub
    modified_dh = False
    params = {
        "a": a[:n],
        "alpha": al[:n],
        "d": d[:n],
        "theta": th[:n],
        "lb": lb[:n],
        "ub": ub[:n],
        "modified_dh": modified_dh,
    }

    robot = RobotRevolute(params)
    graph = RobotRevoluteGraph(robot)
    return robot, graph
