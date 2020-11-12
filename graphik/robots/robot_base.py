from abc import ABC, abstractmethod
import numpy as np
import sympy as sp
import networkx as nx
from numpy import sqrt, sin, cos, pi, arctan2, cross
from numpy.linalg import norm
from liegroups.numpy._base import SEMatrixBase
from liegroups.numpy import SO2, SO3, SE2, SE3
from graphik.utils.utils import (
    flatten,
    level2_descendants,
    wraptopi,
    list_to_variable_dict,
    trans_axis,
    rot_axis,
)
from graphik.utils.kinematics_helpers import (
    from_angle,
    rotx,
    roty,
    rotz,
    dh_to_se3,
    inverse_dh_frame,
    extract_relative_angle_Z,
    rotZ_symb,
    cross_symb,
)


LOWER = "lower_limit"
UPPER = "upper_limit"
BOUNDED = "bounded"
DIST = "weight"
POS = "pos"
ROOT = "p0"


def skew(x):
    """
    Creates a skew symmetric matrix from vector x
    """
    X = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    return X


def angle_to_se2(a: float, theta: float) -> SE2:
    """Transform a single set of DH parameters into an SE2 matrix
    :param a: link length
    :param theta: rotation
    :returns: SE2 matrix
    :rtype: lie.SE2Matrix
    """
    # R = SO2.from_angle(theta)  # TODO: active or passive (i.e., +/- theta?)
    R = SO2(from_angle(theta))
    return SE2(R, R.dot(np.array([a, 0.0])))  # TODO: rotate the translation or not?


def fk_2d(a: list, theta: list, q: list) -> SE2:
    """Get forward kinematics from an array of link lengths and angles
    :param a: displacement along x (link length)
    :param theta: rotation offset of each link
    :param q: active angle variable
    :returns: SE2 matrix
    :rtype: lie.SE2Matrix
    """
    if len(a) > 1:
        return angle_to_se2(a[0], theta[0] + q[0]).dot(fk_2d(a[1:], theta[1:], q[1:]))
    return angle_to_se2(a[0], theta[0] + q[0])


def fk_2d_symb(a: list, theta: list, q: list) -> SE2:
    angle_full = sum(theta) + sum(q)
    # c_all = sp.cos(angle_full)
    # s_all = sp.sin(angle_full)
    R = SO2(from_angle(angle_full))
    t = np.zeros(2)
    for idx in range(len(a)):
        angle_idx = sum(theta[: idx + 1]) + sum(q[: idx + 1])
        t = t + np.array([sp.cos(angle_idx) * a[idx], sp.sin(angle_idx) * a[idx]])
    return SE2(R, t)


def fk_3d(a: list, alpha: list, d: list, theta: list) -> SE3:
    """Get forward kinematics from an array of dh parameters
    :param a: displacement along x
    :param alpha: rotation about x
    :param d: translation along new z
    :param theta: rotation around new z
    :returns: SE3 matrix
    :rtype: lie.SE3Matrix
    """
    if len(a) > 1:
        return dh_to_se3(a[0], alpha[0], d[0], theta[0]).dot(
            fk_3d(a[1:], alpha[1:], d[1:], theta[1:])
        )
    return dh_to_se3(a[0], alpha[0], d[0], theta[0])


def modified_dh_to_se3(a: float, alpha: float, d: float, theta: float) -> SE3:
    """Transform a single set of modified DH parameters into an SE3 matrix
    :param a: displacement along x
    :param alpha: rotation about x
    :param d: translation along new z
    :param theta: rotation around new z
    :returns: SE3 matrix
    :rtype: lie.SE3Matrix
    """

    TransX = SE3(SO3.identity(), np.array([a, 0, 0]))
    # RX = SO3.rotx(alpha)
    RX = SO3(rotx(alpha))
    RotX = SE3(RX, np.zeros(3))
    TransZ = SE3(SO3.identity(), np.array([0, 0, d]))
    # RZ = SO3.rotz(theta)
    RZ = SO3(rotz(theta))
    RotZ = SE3(RZ, np.zeros(3))

    return TransX.dot(RotX.dot(TransZ.dot(RotZ)))


def modified_fk_3d(a: list, alpha: list, d: list, theta: list) -> SE3:
    """Get forward kinematics from an array of  modified dh parameters
    :param a: displacement along x
    :param alpha: rotation about x
    :param d: translation along new z
    :param theta: rotation around new z
    :returns: SE3 matrix
    :rtype: lie.SE3Matrix
    """
    if len(a) > 1:
        return modified_dh_to_se3(a[0], alpha[0], d[0], theta[0]).dot(
            modified_fk_3d(a[1:], alpha[1:], d[1:], theta[1:])
        )
    return modified_dh_to_se3(a[0], alpha[0], d[0], theta[0])


def sph_to_se3(alpha: float, d: float, theta: float) -> SE3:
    # TODO something is not right here
    """Transform a single set of DH parameters into an SE3 matrix
    :param a: displacement along x
    :param alpha: rotation about x
    :param d: translation along new z
    :param theta: rotation around new z
    :returns: SE3 matrix
    :rtype: lie.SE3Matrix
    """

    RX = SO3.rotx(alpha)
    RotX = SE3(RX, np.zeros(3))
    TransZ = SE3(SO3.identity(), np.array([0, 0, d]))
    RZ = SO3.rotz(theta)
    RotZ = SE3(RZ, np.zeros(3))
    return RotZ.dot(RotX.dot(TransZ))


def sph_to_se3_symb(alpha: float, d: float, theta: float) -> SE3:
    # TODO something is not right here
    """Transform a single set of DH parameters into an SE3 matrix
    :param a: displacement along x
    :param alpha: rotation about x
    :param d: translation along new z
    :param theta: rotation around new z
    :returns: SE3 matrix
    :rtype: lie.SE3Matrix
    """

    RX = SO3(rotx(alpha))
    RotX = SE3(RX, np.zeros(3))
    TransZ = SE3(SO3.identity(), np.array([0, 0, d]))
    RZ = SO3(rotz(theta))
    RotZ = SE3(RZ, np.zeros(3))
    return RotZ.dot(RotX.dot(TransZ))


def fk_3d_sph(a: list, alpha: list, d: list, theta: list) -> SE3:
    """Get forward kinematics from an array of dh parameters
    :param a: displacement along x
    :param alpha: rotation about x
    :param d: translation along new z
    :param theta: rotation around new z
    :returns: SE3 matrix
    :rtype: lie.SE3Matrix
    """
    if len(d) > 1:
        return sph_to_se3(alpha[0], d[0], theta[0]).dot(
            fk_3d_sph(a[1:], alpha[1:], d[1:], theta[1:])
        )
    return sph_to_se3(alpha[0], d[0], theta[0])


def fk_3d_sph_symb(a: list, alpha: list, d: list, theta: list) -> SE3:
    """Get forward kinematics from an array of dh parameters using sympy's trigonometric function.
    Used for setting up the symbolic cost function in graphik.solvers.local_solver.LocalSolver

    :param a: displacement along x
    :param alpha: rotation about x
    :param d: translation along new z
    :param theta: rotation around new z
    :returns: SE3 matrix
    :rtype: lie.SE3Matrix
    """
    if len(d) > 1:
        return sph_to_se3_symb(alpha[0], d[0], theta[0]).dot(
            fk_3d_sph_symb(a[1:], alpha[1:], d[1:], theta[1:])
        )
    return sph_to_se3_symb(alpha[0], d[0], theta[0])


def fk_tree_3d(a: list, alpha: list, d: list, theta: list, path_indices: list) -> SE3:
    """Get forward kinematics from an array of link lengths and angles
    :param a: displacement along x (link length)
    :param theta: rotation offset of each link
    :param q: active angle variable
    :param path_indices: path through the tree to traverse
    :returns: SE2 matrix
    :rtype: lie.SE2Matrix
    """
    idx = path_indices[0]
    if len(path_indices) > 1:
        return dh_to_se3(a[idx], alpha[idx], d[idx], theta[idx]).dot(
            fk_tree_3d(a[1:], alpha[1:], d[1:], theta[1:], path_indices[1:])
        )
    return dh_to_se3(a[idx], alpha[idx], d[idx], theta[idx])


def fk_tree_2d(a: list, theta: list, q: list, path_indices: list) -> SE2:
    """Get forward kinematics from an array of link lengths and angles
    :param a: displacement along x (link length)
    :param theta: rotation offset of each link
    :param q: active angle variable
    :param path_indices: path through the tree to traverse
    :returns: SE2 matrix
    :rtype: lie.SE2Matrix
    """
    idx = path_indices[0]
    if len(path_indices) > 1:
        return angle_to_se2(a[idx], theta[idx] + q[idx]).dot(
            fk_tree_2d(a, theta, q, path_indices[1:])
        )
    return angle_to_se2(a[idx], theta[idx] + q[idx])


class Robot(ABC):
    """
    Describes the kinematic parameters for a robot whose joints and links form a tree (no loops like in parallel
    mechanisms).
    """

    def __init__(self):
        self.lambdified = False

    @abstractmethod
    def get_pose(self, node_inputs: dict, query_node: str):
        """Given a list q of N joint variables, calculate the Nth joint's pose.

        :param node_inputs: joint variables node names as keys mapping to values
        :param query_node: node ID of node whose pose we want
        :returns: SE2 or SE3 pose
        :rtype: lie.SE3Matrix
        """
        raise NotImplementedError

    @abstractmethod
    def random_configuration(self):
        """
        Returns a random set of joint values within the joint limits
        determined by lb and ub.
        """
        raise NotImplementedError

    @property
    def n(self) -> int:
        """
        :return: number of links, or joints (including root)
        """
        return self._n

    @n.setter
    def n(self, n: int):
        self._n = n

    @property
    def dim(self) -> int:
        """
        :return: dimension of the robot (2 or 3)
        """
        return self._dim

    @dim.setter
    def dim(self, dim: int):
        self._dim = dim

    @property
    def structure(self) -> nx.DiGraph:
        """
        :return: graph representing the robot's structure
        """
        return self._structure

    @structure.setter
    def structure(self, structure: nx.DiGraph):
        self._structure = structure

    @property
    def kinematic_map(self) -> dict:
        """
        :return: topological graph of the robot's structure
        """
        return self._kinematic_map

    @kinematic_map.setter
    def kinematic_map(self, kmap: dict):
        self._kinematic_map = kmap

    @property
    def limit_edges(self) -> list:
        """
        :return: list of limited edges
        """
        return self._limit_edges

    @limit_edges.setter
    def limit_edges(self, lim: list):
        self._limit_edges = lim

    @property
    def T_base(self) -> SEMatrixBase:
        """
        :return: Transform to robot base frame
        """
        return self._T_base

    @T_base.setter
    def T_base(self, T_base: SEMatrixBase):
        self._T_base = T_base

    @property
    def ub(self) -> dict:
        """
        :return: Upper limits on joint values
        """
        return self._ub

    @ub.setter
    def ub(self, ub: dict):
        self._ub = ub if type(ub) is dict else list_to_variable_dict(flatten([ub]))

    @property
    def lb(self) -> dict:
        """
        :return: Lower limits on joint values
        """
        return self._lb

    @lb.setter
    def lb(self, lb: dict):
        self._lb = lb if type(lb) is dict else list_to_variable_dict(flatten([lb]))

    ########################################
    #           DH PARAMETERS
    ########################################
    @property
    def d(self) -> dict:
        return self._d

    @d.setter
    def d(self, d: dict):
        self._d = d if type(d) is dict else list_to_variable_dict(flatten([d]))

    @property
    def al(self) -> dict:
        return self._al

    @al.setter
    def al(self, al: dict):
        self._al = al if type(al) is dict else list_to_variable_dict(flatten([al]))

    @property
    def a(self) -> dict:
        return self._a

    @a.setter
    def a(self, a: dict):
        self._a = a if type(a) is dict else list_to_variable_dict(flatten([a]))

    @property
    def th(self) -> dict:
        return self._th

    @th.setter
    def th(self, th: dict):
        self._th = th if type(th) is dict else list_to_variable_dict(flatten([th]))

    @property
    def spherical(self) -> bool:
        return False

    @property
    def lambdified(self) -> bool:
        return self._lambdified

    @lambdified.setter
    def lambdified(self, lambdified: bool):
        self._lambdified = lambdified

    def lambdify_get_pose(self):
        """
        Sets the fast full joint kinematics function with lambdify.
        """
        full_pose_expression = sp.symarray(
            "dummy", (self.dim + 1, self.dim + 1, self.n)
        )
        sym_vars = {}
        variable_angles = list(list_to_variable_dict(self.n * [0.0]).keys())
        sym_vars_list = []
        if not self.spherical:
            for var in variable_angles:
                sym_vars[var] = sp.symbols(var)
                sym_vars_list.append(sym_vars[var])
        else:
            for var in variable_angles:
                sym_vars[var] = sp.symbols([var + "_1", var + "_2"])
                sym_vars_list.append(sym_vars[var][0])
                sym_vars_list.append(sym_vars[var][1])
        for idx, var in enumerate(variable_angles):
            if self.dim == 2 or self.spherical:
                full_pose_expression[:, :, idx] = self.get_pose(
                    sym_vars, var, symb=True
                ).as_matrix()

            else:
                full_pose_expression[:, :, idx] = self.get_pose(
                    sym_vars, var, symb=True
                ).as_matrix()
        # if not self.spherical:
        #     x = sp.symarray("x", (self.n,))
        # else:
        #     x = sp.symarray("x", (self.n*2,))
        self.get_full_pose_lambdified = sp.lambdify(
            [sym_vars_list], full_pose_expression, "numpy"
        )
        self.lambdified = True

    def get_full_pose_fast_lambdify(self, node_inputs: dict):
        assert (
            self.lambdified
        ), "This robot has not yet been lambdified: call robot.lambdifiy_get_pose() first."
        input_list = list(node_inputs.values())
        pose_tensor = np.array(self.get_full_pose_lambdified(input_list))
        pose_dict = {}
        if self.spherical:
            for idx in range(self.n):
                pose_dict[f"p{idx+1}"] = pose_tensor[:, :, idx]
        else:
            for idx, key in enumerate(node_inputs):
                pose_dict[key] = pose_tensor[:, :, idx]
        return pose_dict


class RobotPlanar(Robot, ABC):
    def __init__(self, leaves_only_end_effectors=False):
        self.leaves_only_end_effectors = leaves_only_end_effectors
        self.dim = 2
        super(RobotPlanar, self).__init__()

    @property
    def end_effectors(self) -> list:
        """
        Returns the names of end effector nodes and the nodes
        preceeding them (required for orientation goals) as
        a list of lists.
        """
        if not hasattr(self, "_end_effectors"):
            S = self.structure
            if self.leaves_only_end_effectors:
                self._end_effectors = [[x] for x in S if S.out_degree(x) == 0]
            else:
                self._end_effectors = [
                    [x, y]
                    for x in S
                    if S.out_degree(x) == 0
                    for y in S.predecessors(x)
                    if DIST in S[y][x]
                    if S[y][x][DIST] < np.inf
                ]

        return self._end_effectors

    def get_pose(self, node_inputs: dict, query_node: str, symb: bool = False):
        """
        Returns an SE2 element corresponding to the location
        of the query_node in the configuration determined by
        node_inputs.
        """
        if query_node == "p0":
            return SE2.identity()

        path_nodes = self.kinematic_map["p0"][query_node][1:]
        q = np.array([node_inputs[node] for node in path_nodes])
        a = np.array([self.a[node] for node in path_nodes])
        th = np.array([self.th[node] for node in path_nodes])
        if symb:
            return fk_2d_symb(a, th, q)
        else:
            return fk_2d(a, th, q)

    def joint_variables(self, G: nx.Graph) -> dict:
        """
        Finds the set of decision variables corresponding to the
        graph realization G.

        :param G: networkx.DiGraph with known vertex positions
        :returns: array of joint variables t
        :rtype: np.ndarray
        """
        R = {"p0": SO2.identity()}
        joint_variables = {}

        for u, v, dat in self.structure.edges(data=DIST):
            if dat:
                diff_uv = G.nodes[v][POS] - G.nodes[u][POS]
                len_uv = np.linalg.norm(diff_uv)
                sol = np.linalg.solve(len_uv * R[u].as_matrix(), diff_uv)
                theta_idx = np.math.atan2(sol[1], sol[0])
                joint_variables[v] = wraptopi(theta_idx)
                Rz = SO2.from_angle(theta_idx)
                R[v] = R[u].dot(Rz)

        return joint_variables

    def set_limits(self):
        """
        Sets known bounds on the distances between joints.
        This is induced by link length and joint limits.
        """
        S = self.structure
        self.limit_edges = []
        for u in S:
            # direct successors are fully known
            for v in (suc for suc in S.successors(u) if suc):
                S[u][v]["upper_limit"] = S[u][v][DIST]
                S[u][v]["lower_limit"] = S[u][v][DIST]
            for v in (des for des in level2_descendants(S, u) if des):
                ids = self.kinematic_map[u][v]  # TODO generate this at init
                l1 = self.a[ids[1]]
                l2 = self.a[ids[2]]
                lb = self.lb[ids[2]]  # symmetric limit
                ub = self.ub[ids[2]]  # symmetric limit
                lim = max(abs(ub), abs(lb))
                S.add_edge(u, v)
                S[u][v]["upper_limit"] = l1 + l2
                S[u][v]["lower_limit"] = sqrt(
                    l1 ** 2 + l2 ** 2 - 2 * l1 * l2 * cos(pi - lim)
                )
                S[u][v][BOUNDED] = "below"
                self.limit_edges += [[u, v]]  # TODO remove/fix

    def random_configuration(self):
        q = {}
        for key in self.structure:
            if key != "p0":
                q[key] = self.lb[key] + (self.ub[key] - self.lb[key]) * np.random.rand()
        return q

    def jacobian_cost(self, joint_angles: dict, ee_goals) -> np.ndarray:
        """
        Calculate the planar robot's Jacobian with respect to the Euclidean squared cost function.
        """
        kinematic_map = self.kinematic_map["p0"]  # get map to all nodes from root
        end_effector_nodes = ee_goals.keys()
        J = np.zeros(self.n)
        for (
            ee
        ) in end_effector_nodes:  # iterate through end-effector nodes, assumes sorted
            ee_path = kinematic_map[ee][
                1:
            ]  # [:-1]  # no last node, only phys. joint locations
            t_ee = self.get_pose(joint_angles, ee).trans
            dg_ee_x = t_ee[0] - ee_goals[ee].trans[0]
            dg_ee_y = t_ee[1] - ee_goals[ee].trans[1]
            for (pdx, joint_p) in enumerate(ee_path):  # algorithm fills Jac per column
                p_idx = int(joint_p[1:]) - 1
                for jdx in range(pdx, len(ee_path)):
                    node_jdx = ee_path[jdx]
                    theta_jdx = sum([joint_angles[key] for key in ee_path[0 : jdx + 1]])
                    J[p_idx] += (
                        2.0
                        * self.a[node_jdx]
                        * (-dg_ee_x * np.sin(theta_jdx) + dg_ee_y * np.cos(theta_jdx))
                    )

        return J

    def hessian_cost(self, joint_angles: dict, ee_goals) -> np.ndarray:
        """
        Calculate the planar robot's Hessian with respect to the Euclidean squared cost function.
        """
        kinematic_map = self.kinematic_map["p0"]  # get map to all nodes from root
        end_effector_nodes = ee_goals.keys()
        H = np.zeros((self.n, self.n))
        for (
            ee
        ) in end_effector_nodes:  # iterate through end-effector nodes, assumes sorted
            ee_path = kinematic_map[ee][
                1:
            ]  # [:-1]  # no last node, only phys. joint locations
            t_ee = self.get_pose(joint_angles, ee).trans
            dg_ee_x = t_ee[0] - ee_goals[ee].trans[0]
            dg_ee_y = t_ee[1] - ee_goals[ee].trans[1]
            for (pdx, joint_p) in enumerate(ee_path):  # algorithm fills Hess per column
                p_idx = int(joint_p[1:]) - 1
                sin_p_term = 0.0
                cos_p_term = 0.0
                for jdx in range(pdx, len(ee_path)):
                    node_jdx = ee_path[jdx]
                    theta_jdx = sum([joint_angles[key] for key in ee_path[0 : jdx + 1]])
                    sin_p_term += self.a[node_jdx] * np.sin(theta_jdx)
                    cos_p_term += self.a[node_jdx] * np.cos(theta_jdx)

                for (qdx, joint_q) in enumerate(
                    ee_path[pdx:]
                ):  # TODO: check if starting from pdx works
                    qdx = qdx + pdx
                    q_idx = int(joint_q[1:]) - 1
                    sin_q_term = 0.0
                    cos_q_term = 0.0
                    for kdx in range(qdx, len(ee_path)):
                        node_kdx = ee_path[kdx]
                        theta_kdx = sum(
                            [joint_angles[key] for key in ee_path[0 : kdx + 1]]
                        )
                        sin_q_term += self.a[node_kdx] * np.sin(theta_kdx)
                        cos_q_term += self.a[node_kdx] * np.cos(theta_kdx)

                    # assert(q_idx >= p_idx)
                    H[p_idx, q_idx] += (
                        2.0 * sin_q_term * sin_p_term
                        - 2.0 * dg_ee_x * cos_q_term
                        + 2.0 * cos_p_term * cos_q_term
                        - 2.0 * dg_ee_y * sin_q_term
                    )

        return H + H.T - np.diag(np.diag(H))


class RobotSpherical(Robot, ABC):
    def __init__(self, leaves_only_end_effectors=False):
        self.dim = 3
        super(RobotSpherical, self).__init__()
        self.leaves_only_end_effectors = leaves_only_end_effectors

    @property
    def end_effectors(self) -> list:
        """
        Returns the names of end effector nodes and the nodes
        preceeding them (required for orientation goals) as
        a list of lists.
        """
        if not hasattr(self, "_end_effectors"):
            S = self.structure
            if self.leaves_only_end_effectors:
                self._end_effectors = [[x] for x in S if S.out_degree(x) == 0]
            else:
                self._end_effectors = [
                    [x, y]
                    for x in S
                    if S.out_degree(x) == 0
                    for y in S.predecessors(x)
                    if DIST in S[y][x]
                    if S[y][x][DIST] < np.inf
                ]

        return self._end_effectors

    def get_pose(self, joint_values: dict, query_node: str, symb=False) -> SE3:
        """
        Returns an SE3 element corresponding to the location
        of the query_node in the configuration determined by
        node_inputs.
        """
        if query_node == "p0":
            return SE3.identity()
        path_nodes = self.kinematic_map["p0"][query_node][1:]
        q = np.array([joint_values[node][0] for node in path_nodes])
        alpha = np.array([joint_values[node][1] for node in path_nodes])
        a = np.array([self.a[node] for node in path_nodes])
        d = np.array([self.d[node] for node in path_nodes])

        if symb:
            return fk_3d_sph_symb(a, alpha, d, q)
        else:
            return fk_3d_sph(a, alpha, d, q)

    def joint_variables(self, G: nx.Graph, T_final: dict = None) -> np.ndarray:
        """
        Finds the set of decision variables corresponding to the
        graph realization G.

        :param G: networkx.DiGraph with known vertex positions
        :returns: array of joint variables t
        :rtype: np.ndarray
        """
        R = {"p0": SO3.identity()}
        joint_variables = {}
        for u, v, dat in self.structure.edges(data=DIST):
            if dat:
                diff_uv = G.nodes[v][POS] - G.nodes[u][POS]
                len_uv = np.linalg.norm(diff_uv)

                sol = np.linalg.lstsq(len_uv * R[u].as_matrix(), diff_uv)
                sol = sol[0]

                theta_idx = np.math.atan2(sol[1], sol[0]) + pi / 2
                Rz = SO3.rotz(theta_idx)

                # x = (R[u].dot(Rz)).as_matrix()[:, 0]
                # alpha_idx = abs(wraptopi(np.math.acos(sol[2])))
                alpha_idx = abs(np.math.acos(min(sol[2], 1)))
                Rx = SO3.rotx(alpha_idx)

                joint_variables[v] = [wraptopi(theta_idx), alpha_idx]
                R[v] = R[u].dot(Rz.dot(Rx))

        return joint_variables

    def set_limits(self):
        """
        Sets known bounds on the distances between joints.
        This is induced by link length and joint limits.
        """
        S = self.structure
        self.limit_edges = []
        for u in S:
            # direct successors are fully known
            for v in (suc for suc in S.successors(u) if suc):
                S[u][v][UPPER] = S[u][v][DIST]
                S[u][v][LOWER] = S[u][v][DIST]
            for v in (des for des in level2_descendants(S, u) if des):
                ids = self.kinematic_map[u][v]
                l1 = self.d[ids[1]]
                l2 = self.d[ids[2]]
                lb = self.lb[ids[2]]
                ub = self.ub[ids[2]]
                lim = max(abs(ub), abs(lb))
                S.add_edge(u, v)
                S[u][v][UPPER] = l1 + l2
                S[u][v][LOWER] = sqrt(l1 ** 2 + l2 ** 2 - 2 * l1 * l2 * cos(pi - lim))
                S[u][v][BOUNDED] = "below"
                self.limit_edges += [[u, v]]  # TODO remove/fix

    def random_configuration(self):
        """
        Returns a random set of joint values within the joint limits
        determined by lb and ub.
        """
        q = {}
        for key in self.structure:
            if key != "p0":
                q[key] = [
                    -pi + 2 * pi * np.random.rand(),
                    np.abs(
                        wraptopi(
                            self.lb[key]
                            + (self.ub[key] - self.lb[key]) * np.random.rand()
                        )
                    ),
                ]
        return q


class RobotRevolute(Robot):
    def __init__(self, params):
        self.axis_length = 1
        self.dim = 3

        if "T_base" in params:
            self.T_base = params["T_base"]
        else:
            self.T_base = SE3.from_matrix(np.identity(4))

        # Use frame poses at zero conf if provided, if not use DH
        if "T_zero" in params:
            self.T_zero = params["T_zero"]
            self.n = len(self.T_zero) - 1  # number of links
        else:
            if "modified_dh" in params:
                self.modified_dh = params["modified_dh"]
            else:
                self.modified_dh = False

            if all(k in params for k in ("a", "d", "alpha", "theta")):
                self.a = params["a"]
                self.d = params["d"]
                self.al = params["alpha"]
                self.th = params["theta"]
                self.n = len(self.al)  # number of links
            else:
                raise Exception("Robot description not provided.")

        # Topological "map" of the robot
        if "parents" in params:
            self.parents = nx.DiGraph(params["parents"])
        else:
            names = [f"p{idx}" for idx in range(self.n + 1)]
            self.parents = nx.path_graph(names, nx.DiGraph)

        self.kinematic_map = nx.shortest_path(self.parents)

        # joint limits TODO currently assuming symmetric around 0
        if "lb" and "ub" in params:
            self.lb = params["lb"]
            self.ub = params["ub"]
        else:
            self.lb = list_to_variable_dict(self.n * [-pi])
            self.ub = list_to_variable_dict(self.n * [pi])

        self.structure = self.structure_graph()
        self.limit_edges = []  # edges enforcing joint limits
        self.limited_joints = []  # joint limits that can be enforced
        self.set_limits()
        super(RobotRevolute, self).__init__()

    def get_pose(self, joint_angles: dict, query_node: str, symb: bool = False) -> SE3:
        """
        Returns an SE3 element corresponding to the location
        of the query_node in the configuration determined by
        node_inputs.
        """
        kinematic_map = self.kinematic_map
        parents = self.parents
        T_ref = self.T_zero
        T = T_ref["p0"]
        for node in kinematic_map["p0"][query_node][1:]:
            pred = [u for u in parents.predecessors(node)]
            T_rel = T_ref[pred[0]].inv().dot(T_ref[node])
            # print(T_rel)
            if symb:
                T = (T.dot(rotZ_symb(joint_angles[node]))).dot(T_rel)
            else:
                T = T.dot(rot_axis(joint_angles[node], "z")).dot(T_rel)
        return T

    @property
    def end_effectors(self) -> list:
        """
        Returns a list of end effector node pairs, since it's the
        last two points that are defined for a full pose.
        """
        S = self.parents
        return [[x, f"q{x[1:]}"] for x in S if S.out_degree(x) == 0]

    @property
    def T_zero(self) -> dict:
        if not hasattr(self, "_T_zero"):
            T = {}
            T["p0"] = self.T_base
            kinematic_map = self.kinematic_map
            for ee in self.end_effectors:
                for node in kinematic_map["p0"][ee[0]][1:]:
                    path_nodes = kinematic_map["p0"][node][1:]

                    q = np.array([0 for node in path_nodes])
                    a = np.array([self.a[node] for node in path_nodes])
                    alpha = np.array([self.al[node] for node in path_nodes])
                    th = np.array([self.th[node] for node in path_nodes])
                    d = np.array([self.d[node] for node in path_nodes])

                    if not self.modified_dh:
                        T[node] = fk_3d(a, alpha, d, q + th)
                    else:
                        T[node] = modified_fk_3d(a, alpha, d, q + th)
            self._T_zero = T

        return self._T_zero

    @T_zero.setter
    def T_zero(self, T_zero: dict):
        self._T_zero = T_zero

    @property
    def parents(self) -> nx.DiGraph:
        return self._parents

    @parents.setter
    def parents(self, parents: nx.DiGraph):
        self._parents = parents

    def get_all_poses(self, joint_angles: dict) -> dict:
        kinematic_map = self.kinematic_map
        parents = self.parents
        T_ref = self.T_zero
        T = {}
        T["p0"] = T_ref["p0"]
        for ee in self.end_effectors:
            for node in kinematic_map["p0"][ee[0]][1:]:
                pred = [u for u in parents.predecessors(node)]
                T_rel = T_ref[pred[0]].inv().dot(T_ref[node])
                T[node] = T[pred[0]].dot(rot_axis(joint_angles[node], "z")).dot(T_rel)
        return T

    def get_all_poses_symb(self, joint_angles: dict) -> dict:
        kinematic_map = self.kinematic_map
        parents = self.parents
        T_ref = self.T_zero
        T = {}
        T["p0"] = T_ref["p0"]
        for ee in self.end_effectors:
            for node in kinematic_map["p0"][ee[0]][1:]:
                pred = [u for u in parents.predecessors(node)]
                T_rel = T_ref[pred[0]].inv().dot(T_ref[node])
                T[node] = T[pred[0]].dot(rotZ_symb(joint_angles[node])).dot(T_rel)
        return T

    def structure_graph(self) -> nx.DiGraph:
        kinematic_map = self.kinematic_map
        axis_length = self.axis_length
        parents = self.parents
        T = self.T_zero

        S = nx.empty_graph(create_using=nx.DiGraph)
        for ee in self.end_effectors:
            for node in kinematic_map["p0"][ee[0]]:
                aux_node = f"q{node[1:]}"
                node_pos = T[node].trans
                aux_node_pos = T[node].dot(trans_axis(axis_length, "z")).trans

                # Generate nodes for joint
                S.add_nodes_from(
                    [
                        (node, {POS: node_pos}),
                        (
                            aux_node,
                            {POS: aux_node_pos},
                        ),
                    ]
                )

                # Generate edges
                S.add_edge(node, aux_node)
                for pred in parents.predecessors(node):
                    S.add_edges_from([(pred, node), (pred, aux_node)])
                    S.add_edges_from(
                        [(f"q{pred[1:]}", node), (f"q{pred[1:]}", aux_node)]
                    )

        # Generate all edge weights
        for u, v in S.edges():
            S[u][v][DIST] = norm(S.nodes[u][POS] - S.nodes[v][POS])
            S[u][v][LOWER] = S[u][v][DIST]
            S[u][v][UPPER] = S[u][v][DIST]

        # Delete positions used for weights
        for u in S.nodes:
            del S.nodes[u][POS]
        return S

    def jacobian(self, joint_angles: dict, query_frame: str = "") -> np.ndarray:
        """
        Calculate the full robot Jacobian for all end-effectors.
        TODO: make frame selectable
        """

        kinematic_map = self.kinematic_map["p0"]  # get map to all nodes from root
        end_effector_nodes = []
        for ee in self.end_effectors:  # get p nodes in end-effectors
            if ee[0][0] == "p":
                end_effector_nodes += [ee[0]]
            else:
                end_effector_nodes += [ee[1]]

        Ts = self.get_all_poses(joint_angles)  # all frame poses

        J = np.zeros([0, self.n])
        for ee in end_effector_nodes:  # iterate through end-effector nodes
            ee_path = kinematic_map[ee][1:]  # no last node, only phys. joint locations

            T_0_ee = Ts[ee].as_matrix()  # ee frame
            p_ee = T_0_ee[0:3, -1]  # ee position

            Jp = np.zeros([3, self.n])  # translation jac
            Jo = np.zeros([3, self.n])  # rotation jac (angle-axis)
            for joint in ee_path:  # algorithm fills Jac per column
                T_0_i = Ts[list(self.parents.predecessors(joint))[0]].as_matrix()
                z_hat_i = T_0_i[:3, 2]
                p_i = T_0_i[:3, -1]
                j_idx = int(joint[1:]) - 1
                Jp[:, j_idx] = cross(z_hat_i, p_ee - p_i)
                Jo[:, j_idx] = z_hat_i

            J_ee = np.vstack([Jp, Jo])
            J = np.vstack([J, J_ee])  # stack big jac for multiple ee

        return J

    def jacobian_linear_symb(
        self, joint_angles: dict, pose_term=False, ee_keys=None
    ) -> dict:
        """
        Calculate the robot's linear velocity Jacobian for all end-effectors.
        """
        kinematic_map = self.kinematic_map["p0"]  # get map to all nodes from root

        if ee_keys is None:
            end_effector_nodes = []
            for ee in self.end_effectors:  # get p nodes in end-effectors
                if ee[0][0] == "p":
                    end_effector_nodes += [ee[0]]
                else:
                    end_effector_nodes += [ee[1]]
        else:
            end_effector_nodes = ee_keys
        Ts = self.get_all_poses_symb(joint_angles)  # all frame poses
        J = {}  # np.zeros([0, len(node_names) - 1])
        for ee in end_effector_nodes:  # iterate through end-effector nodes
            ee_path = kinematic_map[ee][
                1:
            ]  # [:-1]  # no last node, only phys. joint locations

            T_0_ee = Ts[ee].as_matrix()  # ee frame
            if pose_term:
                dZ = np.array([0.0, 0.0, 1.0])
                p_ee = T_0_ee[0:3, 0:3] @ dZ + T_0_ee[0:3, -1]
            else:
                p_ee = T_0_ee[0:3, -1]  # ee position

            Jp = np.zeros([3, self.n], dtype=object)  # translation jac
            for joint in ee_path:  # algorithm fills Jac per column
                T_0_i = Ts[list(self.parents.predecessors(joint))[0]].as_matrix()
                z_hat_i = T_0_i[:3, 2]
                if pose_term:
                    p_i = T_0_i[0:3, 0:3] @ dZ + T_0_i[0:3, -1]
                else:
                    p_i = T_0_i[:3, -1]
                j_idx = int(joint[1:]) - 1  # node_names.index(joint) - 1
                Jp[:, j_idx] = cross_symb(z_hat_i, p_ee - p_i)
            J[ee] = Jp
        return J

    def hessian(self, joint_angles: dict, query_frame: str = "", J=None) -> np.ndarray:
        """
        Calculates the Hessian at query_frame geometrically.
        TODO: assumes the end effectors are numerically sorted
        """
        if J is None:
            J = self.jacobian(joint_angles)

        end_effector_nodes = []
        kinematic_map = self.kinematic_map["p0"]  # get map to all nodes from root
        for ee in self.end_effectors:  # get p nodes in end-effectors
            if ee[0][0] == "p":
                end_effector_nodes += [ee[0]]
            if ee[1][0] == "p":
                end_effector_nodes += [ee[1]]

        end_effector_nodes.sort(key=lambda val: int(val[1:]))

        num_ee = len(end_effector_nodes)
        N = J.shape[1]
        M = 6  # 3 translation + 3 rotation axes

        H = np.zeros([num_ee, M, N, N])  # fun fact: this is called a quartix

        for ee_ind, ee in enumerate(end_effector_nodes):
            ee_path = kinematic_map[ee][1:]
            visited = []
            for joint in ee_path:
                visited += [joint]
                jdx = int(joint[1:]) - 1
                for joint_base in visited:
                    idx = int(joint_base[1:]) - 1
                    h = (
                        cross(
                            J[ee_ind * 6 + 3 : ee_ind * 6 + 6, idx],
                            J[ee_ind * 6 + 0 : ee_ind * 6 + 3, jdx],
                        ),
                        cross(
                            J[ee_ind * 6 + 3 : ee_ind * 6 + 6, idx],
                            J[ee_ind * 6 + 3 : ee_ind * 6 + 6, jdx],
                        ),
                    )
                    ee_i = end_effector_nodes.index(ee)
                    H[ee_i, :, idx, jdx] = np.concatenate(h, axis=0).T
                    H[ee_i, :, jdx, idx] = H[ee_i, :, idx, jdx]

        # for ee_ind, ee in enumerate(end_effector_nodes):
        #     for idx in range(N):
        #         successors = set(self.parents.successors(f"p{idx + 1}"))
        #         for jdx in range(idx, N):
        #             if idx == jdx or f"p{jdx + 1}" in successors:
        #                 h = (
        #                     cross(J[ee_ind*6 + 3:ee_ind*6 + 6, jdx], J[ee_ind*6 + 0:ee_ind*6 + 3, idx]),
        #                     cross(J[ee_ind*6 + 3:ee_ind*6 + 6, jdx], J[ee_ind*6 + 3:ee_ind*6 + 6, idx]),
        #                 )
        #                 ee_i = end_effector_nodes.index(ee)
        #                 H[ee_i, :, idx, jdx] = np.concatenate(h, axis=0).T
        #                 H[ee_i, :, jdx, idx] = H[ee_i, :, idx, jdx]

        return H

    def hessian_linear_symb(
        self,
        joint_angles: dict,
        J=None,
        query_frame: str = "",
        pose_term=False,
        ee_keys=None,
    ) -> np.ndarray:
        """
        Calculates the Hessian at query_frame geometrically.
        """
        # dZ = np.array([0., 0., 1.])  # For the pose_term = True case
        if J is None:
            J = self.jacobian_linear_symb(joint_angles, pose_term=pose_term)
        kinematic_map = self.kinematic_map["p0"]  # get map to all nodes from root

        if ee_keys is None:
            end_effector_nodes = []
            for ee in self.end_effectors:  # get p nodes in end-effectors
                if ee[0][0] == "p":
                    end_effector_nodes += [ee[0]]
                if ee[1][0] == "p":
                    end_effector_nodes += [ee[1]]
        else:
            end_effector_nodes = ee_keys

        N = len(joint_angles)
        M = 3  # 3 translation
        H = {}
        Ts = self.get_all_poses_symb(joint_angles)
        for ee in end_effector_nodes:
            J_ee = J[ee]
            H_ee = np.zeros((M, N, N), dtype=object)
            ee_path = kinematic_map[ee][1:]

            visited = []

            for joint in ee_path:
                visited += [joint]
                jdx = int(joint[1:]) - 1
                for joint_base in visited:
                    idx = int(joint_base[1:]) - 1
                    T_0_base = Ts[
                        list(self.parents.predecessors(joint_base))[0]
                    ].as_matrix()
                    z_hat_base = T_0_base[:3, 2]
                    h = cross_symb(z_hat_base, J_ee[0:3, jdx])
                    H_ee[:, idx, jdx] = h
                    H_ee[:, jdx, idx] = H_ee[:, idx, jdx]

            H[ee] = H_ee
        return H

    def euclidean_cost_hessian(self, J: dict, K: dict, r: dict):
        """
        Based on 'Solving Inverse Kinematics Using Exact Hessian Matrices', Erleben, 2019
        :param J: dictionary of linear velocity kinematic Jacobians
        :param K: dictionary of tensors representing second order derivative information
        :param r: dictionary where each value for key ee is goal_ee - F_ee(theta)
        :return:
        """
        H = 0
        for e in J.keys():
            J_e = J[e]
            N = J_e.shape[1]
            H += J_e.T @ J_e
            # TODO: Try with einsum for speed, maybe?
            for idx in range(N):
                for jdx in range(idx, N):
                    dH = K[e][:, idx, jdx].T @ r[e]
                    H[idx, jdx] -= dH
                    if idx != jdx:
                        H[jdx, idx] -= dH
        return (
            H  # TODO: the Hessian as written is for a coefficient of 1/2 for each term
        )

    def max_min_distance(self, T0: SE3, T1: SE3, T2: SE3):
        """
        Given three frames, find the maximum and minimum distances between the
        frames T0 and T2. It is assumed that the two frames are connected by an
        unlimited revolute joint with its rotation axis being the z-axis
        of the frame T1.
        """
        tol = 10e-10
        # T_rel_01 = T0.inv().dot(T1)
        T_rel_12 = T1.inv().dot(T2)

        p0 = T0.as_matrix()[0:3, 3]
        z1 = T1.as_matrix()[0:3, 2]
        x1 = T1.as_matrix()[0:3, 0]
        p1 = T1.as_matrix()[0:3, 3]
        p2 = T2.as_matrix()[0:3, 3]

        p0_proj = p0 - (z1.dot(p0 - p1)) * z1  # p0 projected onto T1 plane
        p2_proj = p2 - (z1.dot(p2 - p1)) * z1  # p2 projected onto T1 plane

        if norm(p1 - p0_proj) < tol or norm(p2_proj - p1) < tol:
            d = norm(T2.trans - T0.trans)
            return d, d, False

        r = norm(p2_proj - p1)  # radius of circle p2_proj is on
        delta_th = arctan2(cross(x1, p2_proj - p1).dot(z1), np.dot(x1, p2_proj - p1))

        # closest and farthest point from p0_proj
        sol_1 = r * (p0_proj - p1) / norm(p0_proj - p1) + p1
        sol_2 = -r * (p0_proj - p1) / norm(p0_proj - p1) + p1
        sol_min = min(sol_1 - p0_proj, sol_2 - p0_proj, key=norm) + p0_proj
        sol_max = max(sol_1 - p0_proj, sol_2 - p0_proj, key=norm) + p0_proj

        th_max = arctan2(cross(x1, sol_max - p1).dot(z1), np.dot(x1, sol_max - p1))
        th_min = arctan2(cross(x1, sol_min - p1).dot(z1), np.dot(x1, sol_min - p1))

        rot_min = rot_axis(th_min - delta_th, "z")
        d_min = norm(T1.dot(rot_min).dot(T_rel_12).trans - T0.trans)

        rot_max = rot_axis(th_max - delta_th, "z")
        d_max = norm(T1.dot(rot_max).dot(T_rel_12).trans - T0.trans)

        if abs(th_max - delta_th) < tol and d_max > d_min:
            return d_max, d_min, "below"
        elif abs(th_min - delta_th) < tol and d_max > d_min:
            return d_max, d_min, "above"
        else:
            return d_max, d_min, False

    def set_limits(self):
        """
        Sets known bounds on the distances between joints.
        This is induced by link length and joint limits.
        """
        K = self.parents
        S = self.structure
        T = self.T_zero
        kinematic_map = self.kinematic_map
        transZ = trans_axis(self.axis_length, "z")
        for u in K:
            for v in (des for des in K.successors(u) if des):
                S[u][v][LOWER] = S[u][v][DIST]
                S[u][v][UPPER] = S[u][v][DIST]
            for v in (des for des in level2_descendants(K, u) if des):
                names = [
                    (f"p{u[1:]}", f"p{v[1:]}"),
                    (f"p{u[1:]}", f"q{v[1:]}"),
                    (f"q{u[1:]}", f"p{v[1:]}"),
                    (f"q{u[1:]}", f"q{v[1:]}"),
                ]

                for ids in names:
                    path = kinematic_map[u][v]
                    T0, T1, T2 = [T[path[0]], T[path[1]], T[path[2]]]

                    if "q" in ids[0]:
                        T0 = T0.dot(transZ)
                    if "q" in ids[1]:
                        T2 = T2.dot(transZ)

                    d_max, d_min, limit = self.max_min_distance(T0, T1, T2)

                    if limit:

                        rot_limit = rot_axis(self.ub[v], "z")

                        T_rel = T1.inv().dot(T2)

                        d_limit = norm(T1.dot(rot_limit).dot(T_rel).trans - T0.trans)

                        if limit == "above":
                            d_max = d_limit
                        else:
                            d_min = d_limit

                        self.limited_joints += [v]
                        self.limit_edges += [[ids[0], ids[1]]]  # TODO remove/fix

                    S.add_edge(ids[0], ids[1])
                    if d_max == d_min:
                        S[ids[0]][ids[1]][DIST] = d_max
                    S[ids[0]][ids[1]][UPPER] = d_max
                    S[ids[0]][ids[1]][LOWER] = d_min
                    S[ids[0]][ids[1]][BOUNDED] = limit

    def joint_angles_from_graph(self, G: nx.Graph, T_final: dict = None) -> np.ndarray:
        """
        Calculate joint angles from a complete set of point positions.
        """
        # TODO: make this more readable
        tol = 1e-10
        q_zero = list_to_variable_dict(self.n * [0])
        kinematic_map = self.kinematic_map
        parents = self.parents
        get_pose = self.get_pose

        T = {}
        T["p0"] = self.T_base
        theta = {}

        for ee in self.end_effectors:
            path = kinematic_map["p0"][ee[0]][1:]
            axis_length = self.axis_length
            for node in path:
                aux_node = f"q{node[1:]}"
                pred = [u for u in parents.predecessors(node)]

                T_prev = T[pred[0]]

                T_prev_0 = get_pose(q_zero, pred[0])
                T_0 = get_pose(q_zero, node)
                T_rel = T_prev_0.inv().dot(T_0)
                T_0_q = get_pose(q_zero, node).dot(trans_axis(axis_length, "z"))
                T_rel_q = T_prev_0.inv().dot(T_0_q)

                p = G.nodes[node][POS] - T_prev.trans
                q = G.nodes[aux_node][POS] - T_prev.trans
                ps = T_prev.inv().as_matrix()[:3, :3].dot(p)
                qs = T_prev.inv().as_matrix()[:3, :3].dot(q)

                zs = skew(np.array([0, 0, 1]))
                cp = (T_rel.trans - ps) + zs.dot(zs).dot(T_rel.trans)
                cq = (T_rel_q.trans - qs) + zs.dot(zs).dot(T_rel_q.trans)
                ap = zs.dot(T_rel.trans)
                aq = zs.dot(T_rel_q.trans)
                bp = zs.dot(zs).dot(T_rel.trans)
                bq = zs.dot(zs).dot(T_rel_q.trans)

                c0 = cp.dot(cp) + cq.dot(cq)
                c1 = 2 * (cp.dot(ap) + cq.dot(aq))
                c2 = 2 * (cp.dot(bp) + cq.dot(bq))
                c3 = ap.dot(ap) + aq.dot(aq)
                c4 = bp.dot(bp) + bq.dot(bq)
                c5 = 2 * (ap.dot(bp) + aq.dot(bq))

                # poly = [c0 -c2 +c4, 2*c1 - 2*c5, 2*c0 + 4*c3 -2*c4, 2*c1 + 2*c5, c0 + c2 + c4]
                diff = np.array(
                    [
                        c1 - c5,
                        2 * c2 + 4 * c3 - 4 * c4,
                        3 * c1 + 3 * c5,
                        8 * c2 + 4 * c3 - 4 * c4,
                        -4 * c1 + 4 * c5,
                    ]
                )
                if all(diff < tol):
                    theta[node] = 0
                else:
                    # coeffs = np.array([2*c2 + 4*c3 - 4*c4, 6*c1 + 6*c5, 24*c2 + 12*c3 - 12*c4, -16*c1 + 16*c5])
                    sols = np.roots(
                        diff
                    )  # solutions to the Whaba problem for fixed axis

                    def error_test(x):
                        if abs(x.imag) > 0:
                            return 1e6
                        x = -2 * arctan2(x.real, 1)
                        return (
                            c0
                            + c1 * sin(x)
                            - c2 * cos(x)
                            + c3 * sin(x) ** 2
                            + c4 * cos(x) ** 2
                            - c5 * sin(2 * x) / 2
                        )

                    sol = min(sols, key=error_test)
                    theta[node] = -2 * arctan2(sol.real, 1)

                T[node] = (T_prev.dot(rot_axis(theta[node], "z"))).dot(T_rel)

            if T_final is None:
                return theta

            if (
                T_final[ee[0]] is not None
                and norm(cross(T_rel.trans, np.array([0, 0, 1]))) < tol
            ):
                T_th = (T[node]).inv().dot(T_final[ee[0]]).as_matrix()
                theta[ee[0]] += np.arctan2(T_th[1, 0], T_th[0, 0])

        return theta

    def random_configuration(self):
        """
        Returns a random set of joint values within the joint limits
        determined by lb and ub.
        """
        q = {}
        for key in self.parents:
            if key != "p0":
                q[key] = self.lb[key] + (self.ub[key] - self.lb[key]) * np.random.rand()
        return q
