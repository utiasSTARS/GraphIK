from graphik.utils import skew, normalize
from urdfpy import URDF
from liegroups import SE3, SO3
import numpy as np
from graphik.graphs import ProblemGraphRevolute
from graphik.robots import RobotRevolute
import graphik
from operator import itemgetter


class RobotURDF(object):
    def __init__(self, fname):
        self.fname = fname
        self.urdf = URDF.load(fname)

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
            j = self.urdf.joints.index(joint)
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
        z_hat points along the joint rotation axis.
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

        for ee_joint in self.ee_joints:
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

    def make_Revolute3d(self, ub, lb, randomized_links = False, randomize_percentage = 0.4):
        # if all the child lists have len 1, then chain, otherwise tree
        params = {}

        # assign parents
        joints = list(self.T_zero.keys())
        self.get_parents(joints)
        params["parents"] = self.parents

        T_list = list(self.T_zero.values())
        if randomized_links:
            T_mod = T_list
            for idx in range(len(T_list)-1):
                T_delta = T_list[idx].inv().dot(T_list[idx+1]) # delta translation
                t_delta = T_delta.trans*((1-randomize_percentage) + 2*randomize_percentage*np.random.rand()) # variation
                t_delta[np.abs(t_delta) < 1e-6] = 0
                T_delta.trans = t_delta
                T_mod[idx+1] = T_mod[idx].dot(T_delta)
            T_list = T_mod

        # Assign Transforms
        T_labels = self.get_graphik_labels(joints)
        # T_zero = dict(zip(T_labels, self.T_zero.values()))
        T_zero = dict(zip(T_labels, T_list))
        T0 = T_zero["p0"]
        for key, val in T_zero.items():
            T_zero[key] = T0.inv().dot(val)
        params["T_zero"] = T_zero
        params["num_joints"] = self.n_q_joints

        l = 0
        for cl in self.parents.values():
            l += len(cl)
        if l == len(self.parents.keys()) - 1:
            params["joint_limits_upper"] = ub
            params["joint_limits_lower"] = lb
            return RobotRevolute(params)
        else:
            return RobotRevolute(params)

def get_T_from_joint_axis(axis: np.ndarray):
    """
    Take in the axis string from urdf "X X X" and return the rotation matrix
    assoicated with that axis
    """
    norm = np.linalg.norm
    z_hat = np.array([0, 0, 1])

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
    graph = ProblemGraphRevolute(robot)
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
    graph = ProblemGraphRevolute(robot)
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
    graph = ProblemGraphRevolute(robot)
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
    graph = ProblemGraphRevolute(robot)
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
    graph = ProblemGraphRevolute(robot)
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
        "num_joints": n,
    }

    robot = RobotRevolute(params)
    graph = ProblemGraphRevolute(robot)
    return robot, graph
