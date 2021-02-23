from typing import Any, Dict, Union, List

import networkx as nx
import numpy as np
from graphik.robots.robot_base import Robot, SEMatrix
from graphik.utils.constants import *
from graphik.utils.geometry import cross_symb, rot_axis, skew, trans_axis
from graphik.utils.kinematics import fk_3d, modified_fk_3d
from graphik.utils.utils import list_to_variable_dict
from liegroups.numpy import SE3
from numpy import arctan2, cos, cross, pi, sin
from numpy.linalg import norm
from numpy.typing import ArrayLike


class RobotRevolute(Robot):
    def __init__(self, params):
        self.dim = 3
        self.n = params.get("num_joints", len(params["lb"]))  # number of joints
        self.axis_length = params.get("axis_length", 1)  # distance between p and q
        self.T_base = params.get("T_base", SE3.identity())  # base frame

        # Topological "map" of the robot, if not provided assume chain
        if "parents" in params:
            self.parents = nx.DiGraph(params["parents"])
        else:
            self.parents = nx.path_graph(
                [f"p{idx}" for idx in range(self.n + 1)], nx.DiGraph
            )

        # A dict of shortest paths between joints for forward kinematics
        self.kinematic_map = nx.shortest_path(self.parents)

        # Use frame poses at zero conf if provided, otherwise construct from DH
        if "T_zero" in params:
            self.T_zero = params["T_zero"]
        else:
            try:
                self.a, self.d, self.al, self.th, self.modified_dh = (
                    params["a"],
                    params["d"],
                    params["alpha"],
                    params["theta"],
                    params["modified_dh"],
                )
            except KeyError:
                raise Exception("Robot description not provided.")

        self.generate_structure_graph()

        self.lb = params.get("lb", dict(zip(self.joint_ids, self.n * [-pi])))
        self.ub = params.get("ub", dict(zip(self.joint_ids, self.n * [pi])))

        self.set_limits()
        super(RobotRevolute, self).__init__()

    @property
    def end_effectors(self) -> list:
        """
        Returns a list of end effector node pairs, since it's the
        last two points that are defined for a full pose.
        """
        if not hasattr(self, "_end_effectors"):
            self._end_effectors = [
                [x, f"q{x[1:]}"]
                for x in self.parents
                if self.parents.out_degree(x) == 0
            ]
        return self._end_effectors

    @property
    def T_zero(self) -> dict:
        if not hasattr(self, "_T_zero"):
            T = {"p0": self.T_base}
            kinematic_map = self.kinematic_map
            for ee in self.end_effectors:
                for node in kinematic_map["p0"][ee[0]][1:]:
                    path_nodes = kinematic_map["p0"][node][1:]

                    q = np.asarray([0 for node in path_nodes])
                    a = np.asarray([self.a[node] for node in path_nodes])
                    alpha = np.asarray([self.al[node] for node in path_nodes])
                    th = np.asarray([self.th[node] for node in path_nodes])
                    d = np.asarray([self.d[node] for node in path_nodes])

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

    def generate_structure_graph(self):
        trans_z = trans_axis(self.axis_length, "z")
        T = self.T_zero

        S = nx.empty_graph(create_using=nx.DiGraph)

        for ee in self.end_effectors:
            k_map = self.kinematic_map["p0"][ee[0]]
            for idx in range(len(k_map)):
                cur, aux_cur = k_map[idx], f"q{k_map[idx][1:]}"
                cur_pos, aux_cur_pos = T[cur].trans, T[cur].dot(trans_z).trans
                dist = norm(cur_pos - aux_cur_pos)

                # Add nodes for joint and edge between them
                S.add_nodes_from([(cur, {POS: cur_pos}), (aux_cur, {POS: aux_cur_pos})])
                S.add_edge(
                    cur, aux_cur, **{DIST: dist, LOWER: dist, UPPER: dist, BOUNDED: []}
                )

                # If there exists a preceeding joint, connect it to new
                if idx != 0:
                    pred, aux_pred = (k_map[idx - 1], f"q{k_map[idx-1][1:]}")
                    for u in [pred, aux_pred]:
                        for v in [cur, aux_cur]:
                            dist = norm(S.nodes[u][POS] - S.nodes[v][POS])
                            S.add_edge(
                                u,
                                v,
                                **{DIST: dist, LOWER: dist, UPPER: dist, BOUNDED: []},
                            )
                    S[pred][cur][TRANSFORM] = T[pred].inv().dot(T[cur])

        # Delete positions used for weights
        for u in S.nodes:
            del S.nodes[u][POS]

        # Set node type to robot
        nx.set_node_attributes(S, ROBOT, TYPE)

        # Set structure graph attribute
        self.structure = S
        return S

    def get_pose(self, joint_angles: dict, query_node: str) -> SE3:
        """
        Returns an SE3 element corresponding to the location
        of the query_node in the configuration determined by
        node_inputs.
        """
        kinematic_map = self.kinematic_map["p0"]["p" + query_node[1:]]

        T = self.T_base

        for idx in range(len(kinematic_map) - 1):
            pred, cur = kinematic_map[idx], kinematic_map[idx + 1]
            T_rel = self.structure[pred][cur][TRANSFORM]
            T_rot = rot_axis(joint_angles[cur], "z")
            T = T.dot(T_rot).dot(T_rel)

        if query_node[0] == "q":
            T_trans = trans_axis(self.axis_length, "z")
            T = T.dot(T_trans)

        return T

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
            return d_max, d_min, BELOW
        elif abs(th_min - delta_th) < tol and d_max > d_min:
            return d_max, d_min, ABOVE
        else:
            return d_max, d_min, False

    def set_limits(self):
        """
        Sets known bounds on the distances between joints.
        This is induced by link length and joint limits.
        """
        S = self.structure
        T = self.T_zero
        self.limit_edges = []  # edges enforcing joint limits
        self.limited_joints = []  # joint limits that can be enforced
        kinematic_map = self.kinematic_map
        T_axis = trans_axis(self.axis_length, "z")

        for ee in self.end_effectors:
            k_map = self.kinematic_map["p0"][ee[0]]
            for idx in range(2, len(k_map)):
                cur, prev = k_map[idx], k_map[idx - 2]
                names = [
                    (f"p{prev[1:]}", f"p{cur[1:]}"),
                    (f"p{prev[1:]}", f"q{cur[1:]}"),
                    (f"q{prev[1:]}", f"p{cur[1:]}"),
                    (f"q{prev[1:]}", f"q{cur[1:]}"),
                ]
                for ids in names:
                    path = kinematic_map[prev][cur]
                    T0, T1, T2 = [T[path[0]], T[path[1]], T[path[2]]]

                    if "q" in ids[0]:
                        T0 = T0.dot(T_axis)
                    if "q" in ids[1]:
                        T2 = T2.dot(T_axis)

                    d_max, d_min, limit = self.max_min_distance(T0, T1, T2)

                    if limit:

                        rot_limit = rot_axis(self.ub[cur], "z")

                        T_rel = T1.inv().dot(T2)

                        d_limit = norm(T1.dot(rot_limit).dot(T_rel).trans - T0.trans)

                        if limit == ABOVE:
                            d_max = d_limit
                        else:
                            d_min = d_limit

                        self.limited_joints += [cur]
                        self.limit_edges += [[ids[0], ids[1]]]  # TODO remove/fix

                    S.add_edge(ids[0], ids[1])
                    if d_max == d_min:
                        S[ids[0]][ids[1]][DIST] = d_max
                    S[ids[0]][ids[1]][BOUNDED] = [limit]
                    S[ids[0]][ids[1]][UPPER] = d_max
                    S[ids[0]][ids[1]][LOWER] = d_min

    def joint_variables(
        self, G: nx.DiGraph, T_final: Dict[str, SE3] = None
    ) -> Dict[str, Any]:
        """
        Calculate joint angles from a complete set of point positions.
        """
        # TODO: make this more readable
        tol = 1e-10
        q_zero = list_to_variable_dict(self.n * [0])
        get_pose = self.get_pose
        axis_length = self.axis_length

        T = {}
        T["p0"] = self.T_base
        theta = {}

        for ee in self.end_effectors:
            k_map = self.kinematic_map["p0"][ee[0]]
            for idx in range(1, len(k_map)):
                cur, aux_cur = k_map[idx], f"q{k_map[idx][1:]}"
                pred, aux_pred = (k_map[idx - 1], f"q{k_map[idx-1][1:]}")

                T_prev = T[pred]

                T_prev_0 = get_pose(q_zero, pred)
                T_0 = get_pose(q_zero, cur)
                T_rel = T_prev_0.inv().dot(T_0)
                T_0_q = get_pose(q_zero, cur).dot(trans_axis(axis_length, "z"))
                T_rel_q = T_prev_0.inv().dot(T_0_q)

                p = G.nodes[cur][POS] - T_prev.trans
                q = G.nodes[aux_cur][POS] - T_prev.trans
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
                diff = np.asarray(
                    [
                        c1 - c5,
                        2 * c2 + 4 * c3 - 4 * c4,
                        3 * c1 + 3 * c5,
                        8 * c2 + 4 * c3 - 4 * c4,
                        -4 * c1 + 4 * c5,
                    ]
                )
                if all(diff < tol):
                    theta[cur] = 0
                else:
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
                    theta[cur] = -2 * arctan2(sol.real, 1)

                T[cur] = (T_prev.dot(rot_axis(theta[cur], "z"))).dot(T_rel)

            if T_final is None:
                return theta

            if (
                T_final[ee[0]] is not None
                and norm(cross(T_rel.trans, np.asarray([0, 0, 1]))) < tol
            ):
                T_th = (T[cur]).inv().dot(T_final[ee[0]]).as_matrix()
                theta[ee[0]] += np.arctan2(T_th[1, 0], T_th[0, 0])

        return theta

    def random_configuration(self):
        """
        Returns a random set of joint values within the joint limits
        determined by lb and ub.
        """
        q = {}
        for key in self.joint_ids:
            if key != "p0":
                q[key] = self.lb[key] + (self.ub[key] - self.lb[key]) * np.random.rand()
        return q

    def jacobian(
        self,
        joint_angles: Dict[str, float],
        nodes: Union[List[str], str],
        Ts: Dict[str, SEMatrix] = None,
    ) -> Dict[str, ArrayLike]:
        """
        Calculate the robot's linear velocity Jacobian for all end-effectors.
        """
        kinematic_map = self.kinematic_map["p0"]  # get map to all nodes from root

        # find end-effector nodes #FIXME so much code
        if nodes is None:
            nodes = []
            for ee in self.end_effectors:  # get p nodes in end-effectors
                if ee[0][0] == "p":
                    nodes += [ee[0]]
                elif ee[1][0] == "p":
                    nodes += [ee[1]]

        if Ts is None:
            Ts = self.get_all_poses(joint_angles)  # precompute all poses

        J = {}
        for node in nodes:  # iterate through end-effector nodes
            path = kinematic_map[node][1:]  # find nodes for joints to end-effector
            p_ee = Ts[node].trans

            J[node] = np.zeros([6, self.n])
            for idx, joint in enumerate(path):  # algorithm fills Jac per column
                T_0_i = Ts[list(self.parents.predecessors(joint))[0]]
                z_hat_i = T_0_i.rot.mat[:3, 2]
                p_i = T_0_i.trans
                J[node][:3, idx] = np.cross(z_hat_i, p_ee - p_i)
                J[node][3:, idx] = z_hat_i
        return J

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
            J = self.jacobian(joint_angles, pose_term=pose_term)
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
        # Ts = self.get_all_poses_symb(joint_angles)
        Ts = self.get_all_poses(joint_angles)
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