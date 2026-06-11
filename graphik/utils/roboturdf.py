from graphik.utils import skew, normalize
from yourdfpy import URDF
from yourdfpy.urdf import Joint as _YJoint, Link as _YLink
from pymlg.numpy import SE3
import numpy as np
from graphik.graphs import ProblemGraph
from graphik.robots import Robot
import graphik
from operator import itemgetter

# yourdfpy 0.0.60's Joint and Link are @dataclass(eq=False) but define a
# custom structural __eq__ that compares fields. Per the dataclass spec, this
# combination leaves __hash__ set to None, making instances unhashable.
# yourdfpy itself acknowledges this in urdf.py with a TODO at the top of
# update_cfg. RobotURDF.T_zero is keyed by Joint objects (see line ~157
# below: T[joint] = T_joint), so we need them hashable.
#
# Restore identity-based __hash__. WARNING: this intentionally violates the
# Python invariant a == b ⇒ hash(a) == hash(b). Two distinct Joint instances
# with identical fields will compare equal under == but hash differently.
# Safe for graphIK only because every T_zero key/lookup uses the canonical
# instance from self._joints / self.urdf.actuated_joints. Do NOT put Joint
# or Link objects from independent URDF.load() calls into the same set or
# dict — dedup by hash will not match dedup by ==.
if _YJoint.__hash__ is None:
    _YJoint.__hash__ = object.__hash__
if _YLink.__hash__ is None:
    _YLink.__hash__ = object.__hash__


class RobotURDF(object):
    def __init__(self, fname):
        self.fname = fname
        self.urdf = URDF.load(fname)
        self._joints = list(self.urdf.robot.joints)
        self._links = list(self.urdf.robot.links)

        self.urdf_ind_to_q, self.q_to_urdf_ind = self.joint_map()
        self.n_q_joints = len(self.q_to_urdf_ind)
        self.n_urdf_joints = len(self.urdf_ind_to_q)

        self.ee_joints = self.find_end_effector_joints()
        self.T_zero = self.extract_T_zero_from_URDF(frame="joint")
        self.scene = None

        self.parents = None

    def joint_map(self):
        urdf_ind_to_q = {}
        q_to_urdf_ind = {}
        q_ind = 1
        label = "p{0}"

        for joint in self.urdf.actuated_joints:
            j = self._joints.index(joint)
            urdf_ind_to_q[j] = label.format(q_ind)
            q_to_urdf_ind[label.format(q_ind)] = j
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
        for joint in self._joints:
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
        for joint in self._joints:
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
            raise ValueError("Base joint not in joints")

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
            raise ValueError(f"{joint.name} is not an actuated joint") from None

    def find_link_by_name(self, name):
        for link in self._links:
            if link.name == name:
                return link
        return None

    def find_joint_by_name(self, name):
        for joint in self._joints:
            if joint.name == name:
                return joint
        return None

    def extract_T_zero_from_URDF(self, q=None, frame="joint"):
        """
        T is located at the joint's origin, the rotation such that
        z_hat points along the joint rotation axis.
        """
        # yourdfpy's update_cfg is stateful and partial (only mutates joints
        # whose names appear in cfg), unlike urdfpy's link_fk which was
        # stateless. Build a full cfg covering every actuated joint, defaulting
        # unspecified ones to 0, so this method behaves like urdfpy's link_fk
        # regardless of what q the URDF was last queried at.
        partial_cfg = self.map_to_urdf_ind(q) if q is not None else {}
        cfg = {j.name: partial_cfg.get(j.name, 0.0) for j in self.urdf.actuated_joints}
        self.urdf.update_cfg(cfg)
        T = {}
        for joint in self.urdf.actuated_joints:
            # get child link frame
            child_link = self.find_link_by_name(joint.child)
            T_link = self.urdf.get_transform(child_link.name)  # 4x4 ndarray
            if frame == "joint":
                joint_axis = joint.axis
                T_joint_axis = get_T_from_joint_axis(joint_axis)
                T_joint = T_link @ SE3.inverse(T_joint_axis)
                T[joint] = T_joint
            else:
                T[joint] = T_link

        for ee_joint in self.ee_joints:
            ee_link = self.find_link_by_name(ee_joint.child)
            T[ee_joint] = self.urdf.get_transform(ee_link.name)

        return T

    def find_end_effector_joints(self):
        """
        Finds end-effector joints. Assumes that the end effector frame has
        a fixed joint.

        Returns
        -------

        ee_joints : list
            List of URDF Joint objects that correspond to the End Effectors

        """
        ee_joints = []

        for joint in self._joints:
            child_joints = self.find_joints_actuated_child_joints(joint)
            if child_joints == []:
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
        names = [self._joints[i].name for i in urdf_ind]
        # urdf_q = dict(zip(urdf_ind, list(q.values())))
        urdf_q = dict(zip(names, list(q.values())))

        return urdf_q

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
            list of URDF Joint objects

        Returns
        -------
        labels : list
            list of the labels
        """
        n = len(joints)

        label = "p{0}"
        labels = [label.format(i) for i in range(n)]
        return labels

    def make_Revolute3d(self, ub, lb, randomized_links = False, randomize_percentage = 0.4):
        # if all the child lists have len 1, then chain, otherwise tree
        params = {}

        # assign parents
        joints = list(self.T_zero.keys())
        self.get_parents(joints)
        params["parents"] = self.parents

        T_list = list(self.T_zero.values())
        if randomized_links:
            T_mod = list(T_list)
            for idx in range(len(T_list) - 1):
                T_delta = SE3.inverse(T_list[idx]) @ T_list[idx + 1]
                t_delta = T_delta[:3, 3] * (
                    (1 - randomize_percentage) + 2 * randomize_percentage * np.random.rand()
                )
                t_delta[np.abs(t_delta) < 1e-6] = 0
                T_delta = T_delta.copy()
                T_delta[:3, 3] = t_delta
                T_mod[idx + 1] = T_mod[idx] @ T_delta
            T_list = T_mod

        # Assign Transforms
        T_labels = self.get_graphik_labels(joints)
        T_zero = dict(zip(T_labels, T_list))
        T0 = T_zero["p0"]
        for key, val in T_zero.items():
            T_zero[key] = SE3.inverse(T0) @ val
        params["T_zero"] = T_zero
        params["num_joints"] = self.n_q_joints

        params["joint_limits_upper"] = ub
        params["joint_limits_lower"] = lb
        return Robot({**params, "dim": 3})

def get_T_from_joint_axis(axis: np.ndarray):
    """
    Take in the axis vector from urdf and return the 4x4 SE(3) transform
    that aligns the z-axis with the joint axis.
    """
    norm = np.linalg.norm
    z_hat = np.array([0, 0, 1])

    if all(np.isclose(axis, -z_hat)):
        from graphik.utils.kinematics import Rx
        R = Rx(np.pi)
    elif not all(np.isclose(axis, z_hat)):
        rot_axis = np.cross(axis, z_hat)
        ang = -np.arcsin(norm(rot_axis) / (norm(axis) * norm(z_hat)))
        rot_axis = normalize(rot_axis)
        rot_axis = rot_axis.reshape(3, 1)

        R = (
            np.eye(3) * np.cos(ang)
            + (1 - np.cos(ang)) * np.dot(rot_axis, rot_axis.transpose())
            - np.sin(ang) * skew(rot_axis.ravel())
        )
    else:
        R = np.eye(3)

    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = np.zeros(3)

    return T

def load_schunk_lwa4p(limits=None, randomized_links = False, randomize_percentage = 0.4):
    fname = graphik.__path__[0] + "/robots/urdfs/lwa4p.urdf"
    urdf_robot = RobotURDF(fname)
    n = urdf_robot.n_q_joints
    if limits is None:
        ub = np.ones(n) * np.pi
        lb = -ub
    else:
        lb = limits[0]
        ub = limits[1]
    robot = urdf_robot.make_Revolute3d(ub, lb, randomized_links, randomize_percentage)  # make the Revolute class from a URDF
    graph = ProblemGraph(robot)
    return robot, graph


def load_schunk_lwa4d(limits=None, randomized_links = False, randomize_percentage = 0.4):
    fname = graphik.__path__[0] + "/robots/urdfs/lwa4d.urdf"
    urdf_robot = RobotURDF(fname)
    n = urdf_robot.n_q_joints
    if limits is None:
        ub = np.ones(n) * np.pi
        lb = -ub
    else:
        lb = limits[0]
        ub = limits[1]
    robot = urdf_robot.make_Revolute3d(ub, lb, randomized_links, randomize_percentage)  # make the Revolute class from a URDF
    graph = ProblemGraph(robot)
    return robot, graph


def load_kuka(limits=None, randomized_links = False, randomize_percentage = 0.4):
    fname = graphik.__path__[0] + "/robots/urdfs/kuka_iiwr.urdf"
    urdf_robot = RobotURDF(fname)
    n = urdf_robot.n_q_joints
    if limits is None:
        ub = np.ones(n) * np.pi
        lb = -ub
    else:
        lb = limits[0]
        ub = limits[1]
    robot = urdf_robot.make_Revolute3d(ub, lb, randomized_links, randomize_percentage)  # make the Revolute class from a URDF
    graph = ProblemGraph(robot)
    return robot, graph

def load_panda(limits=None, randomized_links = False, randomize_percentage = 0.4):
    fname = graphik.__path__[0] + "/robots/urdfs/panda_arm.urdf"
    urdf_robot = RobotURDF(fname)
    n = urdf_robot.n_q_joints
    if limits is None:
        ub = np.ones(n) * np.pi
        lb = -ub
    else:
        lb = limits[0]
        ub = limits[1]
    robot = urdf_robot.make_Revolute3d(ub, lb, randomized_links, randomize_percentage)  # make the Revolute class from a URDF
    # print(robot.structure.nodes())
    graph = ProblemGraph(robot)
    return robot, graph

def load_ur10(limits=None, randomized_links = False, randomize_percentage = 0.4):
    fname = graphik.__path__[0] + "/robots/urdfs/ur10_mod.urdf"
    urdf_robot = RobotURDF(fname)
    n = urdf_robot.n_q_joints
    if limits is None:
        ub = np.ones(n) * np.pi
        lb = -ub
    else:
        lb = limits[0]
        ub = limits[1]
    robot = urdf_robot.make_Revolute3d(ub, lb, randomized_links, randomize_percentage)  # make the Revolute class from a URDF
    # print(robot.structure.nodes())
    graph = ProblemGraph(robot)
    return robot, graph


def load_truncated_ur10(n: int, limits=None):
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
    if limits is None:
        ub = np.pi * np.ones(n)
        lb = -ub
    else:
        lb = limits[0]
        ub = limits[1]
    modified_dh = False
    params = {
        "a": a[:n],
        "alpha": al[:n],
        "d": d[:n],
        "theta": th[:n],
        "joint_limits_lower": lb[:n],
        "joint_limits_upper": ub[:n],
        "modified_dh": modified_dh,
        "num_joints": n,
    }

    robot = Robot({**params, "dim": 3})
    graph = ProblemGraph(robot)
    return robot, graph
