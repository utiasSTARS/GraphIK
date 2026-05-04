from typing import Dict, List, Any, Optional
import numpy as np
import numpy.linalg as la
from graphik.robots import RobotRevolute
from graphik.graphs.graph_base import ProblemGraph
from graphik.utils import *
from pymlg.numpy import SE3
import networkx as nx
from numpy import cos, pi, sqrt, arctan2, cross


class ProblemGraphRevolute(ProblemGraph):
    def __init__(
        self,
        robot: RobotRevolute,
        params: Dict = {},
    ):
        super(ProblemGraphRevolute, self).__init__(robot, params)

        base = self.base_subgraph()
        structure = self.structure_graph()

        composition = nx.compose(base, structure)
        self.add_nodes_from(composition.nodes(data=True))
        self.add_edges_from(composition.edges(data=True))

        self.set_limits()
        self.root_angle_limits()

    def base_subgraph(self) -> nx.DiGraph:
        axis_length = self.axis_length
        base = nx.DiGraph(
            [
                ("p0", "x"),
                ("p0", "y"),
                ("p0", "q0"),
                ("x", "y"),
                ("y", "q0"),
                ("q0", "x"),
            ]
        )
        base.add_nodes_from(
            [
                ("p0", {POS: np.array([0, 0, 0]), TYPE: [ROBOT, BASE]}),
                ("x", {POS: np.array([axis_length, 0, 0]), TYPE: [BASE]}),
                ("y", {POS: np.array([0, -axis_length, 0]), TYPE: [BASE]}),
                ("q0", {POS: np.array([0, 0, axis_length]), TYPE: [ROBOT, BASE]}),
            ]
        )
        for u, v in base.edges():
            base[u][v][DIST] = la.norm(base.nodes[u][POS] - base.nodes[v][POS])
            base[u][v][LOWER] = base[u][v][DIST]
            base[u][v][UPPER] = base[u][v][DIST]
            base[u][v][BOUNDED] = []
        return base

    def structure_graph(self):
        trans_z = trans_axis(self.axis_length, "z")
        robot = self.robot

        structure = nx.empty_graph(create_using=nx.DiGraph)

        for ee in robot.end_effectors:
            k_map = robot.kinematic_map[ROOT][ee]
            for idx in range(len(k_map)):
                cur, aux_cur = k_map[idx], f"q{k_map[idx][1:]}"
                cur_pos, aux_cur_pos = (
                    robot.nodes[cur]["T0"][:3, 3],
                    (robot.nodes[cur]["T0"] @ trans_z)[:3, 3],
                )
                type = [ROBOT]
                if cur == ee:
                    type += [END_EFFECTOR]
                if cur == ROOT:
                    type += [BASE]

                # Add nodes for joint and edge between them
                structure.add_nodes_from(
                    [
                        (cur, {POS: cur_pos, TYPE: type}),
                        (aux_cur, {POS: aux_cur_pos, TYPE: type}),
                    ]
                )

                dist = la.norm(cur_pos - aux_cur_pos)

                structure.add_edge(
                    cur, aux_cur, **{DIST: dist, LOWER: dist, UPPER: dist, BOUNDED: []}
                )

                # If there exists a preceeding joint, connect it to new
                if idx != 0:
                    pred, aux_pred = (k_map[idx - 1], f"q{k_map[idx-1][1:]}")
                    for u in [pred, aux_pred]:
                        for v in [cur, aux_cur]:
                            dist = la.norm(
                                structure.nodes[u][POS] - structure.nodes[v][POS]
                            )
                            structure.add_edge(
                                u,
                                v,
                                **{DIST: dist, LOWER: dist, UPPER: dist, BOUNDED: []},
                            )

        # Delete positions used for weights
        for u in structure.nodes:
            del structure.nodes[u][POS]

        return structure

    def root_angle_limits(self):
        axis_length = self.axis_length
        robot = self.robot
        upper_limits = self.robot.ub
        limited_joints = self.limited_joints
        T1 = robot.nodes[ROOT]["T0"]
        base_names = ["x", "y"]
        names = ["p1", "q1"]
        T_axis = trans_axis(axis_length, "z")

        for base_node in base_names:
            for node in names:
                T0 = np.eye(4)
                T0[:3, 3] = self.nodes[base_node][POS]
                if node[0] == "p":
                    T2 = robot.nodes["p1"]["T0"]
                else:
                    T2 = robot.nodes["p1"]["T0"] @ T_axis

                N = T1[:3, 2]
                C = T1[:3, 3] + (N.dot(T2[:3, 3] - T1[:3, 3])) * N
                r = np.linalg.norm(T2[:3, 3] - C)
                P = T0[:3, 3]
                d_max, d_min = max_min_distance_revolute(r, P, C, N)
                d = np.linalg.norm(T2[:3, 3] - T0[:3, 3])

                if d_max == d_min:
                    limit = False
                elif d == d_max:
                    limit = BELOW
                elif d == d_min:
                    limit = ABOVE
                else:
                    limit = None

                if limit:
                    if node[0] == "p":
                        T_rel = SE3.inverse(T1) @ robot.nodes["p1"]["T0"]
                    else:
                        T_rel = SE3.inverse(T1) @ (robot.nodes["p1"]["T0"] @ T_axis)

                    d_limit = la.norm(
                        (T1 @ rot_axis(upper_limits["p1"], "z") @ T_rel)[:3, 3]
                        - T0[:3, 3]
                    )

                    if limit == ABOVE:
                        d_max = d_limit
                    else:
                        d_min = d_limit
                    limited_joints += ["p1"]

                self.add_edge(base_node, node)
                if d_max == d_min:
                    self[base_node][node][DIST] = d_max
                self[base_node][node][BOUNDED] = [limit]
                self[base_node][node][UPPER] = d_max
                self[base_node][node][LOWER] = d_min

    def set_limits(self):
        """
        Sets known bounds on the distances between joints.
        This is induced by link length and joint limits.
        """
        S = self.structure
        robot = self.robot
        kinematic_map = self.robot.kinematic_map
        T_axis = trans_axis(self.axis_length, "z")
        end_effectors = self.robot.end_effectors
        upper_limits = self.robot.ub

        limited_joints = []  # joint limits that can be enforced
        for ee in end_effectors:
            k_map = kinematic_map[ROOT][ee]
            for idx in range(2, len(k_map)):
                cur, prev = k_map[idx], k_map[idx - 2]
                names = [
                    (MAIN_PREFIX + str(prev[1:]), MAIN_PREFIX + str(cur[1:])),
                    (MAIN_PREFIX + str(prev[1:]), AUX_PREFIX + str(cur[1:])),
                    (AUX_PREFIX + str(prev[1:]), MAIN_PREFIX + str(cur[1:])),
                    (AUX_PREFIX + str(prev[1:]), AUX_PREFIX + str(cur[1:])),
                ]
                for ids in names:
                    path = kinematic_map[prev][cur]
                    T0, T1, T2 = [
                        robot.nodes[path[0]]["T0"],
                        robot.nodes[path[1]]["T0"],
                        robot.nodes[path[2]]["T0"],
                    ]

                    if AUX_PREFIX in ids[0]:
                        T0 = T0 @ T_axis
                    if AUX_PREFIX in ids[1]:
                        T2 = T2 @ T_axis

                    N = T1[:3, 2]
                    C = T1[:3, 3] + (N.dot(T2[:3, 3] - T1[:3, 3])) * N
                    r = la.norm(T2[:3, 3] - C)
                    P = T0[:3, 3]
                    d_max, d_min = max_min_distance_revolute(r, P, C, N)

                    d = la.norm(T2[:3, 3] - T0[:3, 3])
                    if d_max == d_min:
                        limit = False
                    elif d == d_max:
                        limit = BELOW
                    elif d == d_min:
                        limit = ABOVE
                    else:
                        limit = None

                    if limit:
                        rot_limit = rot_axis(upper_limits[cur], "z")

                        T_rel = SE3.inverse(T1) @ T2

                        d_limit = la.norm((T1 @ rot_limit @ T_rel)[:3, 3] - T0[:3, 3])

                        if limit == ABOVE:
                            d_max = d_limit
                        else:
                            d_min = d_limit

                        limited_joints += [cur]

                    self.add_edge(ids[0], ids[1])
                    if d_max == d_min:
                        S[ids[0]][ids[1]][DIST] = d_max
                    self[ids[0]][ids[1]][BOUNDED] = [limit]
                    self[ids[0]][ids[1]][UPPER] = d_max
                    self[ids[0]][ids[1]][LOWER] = d_min

        self.limited_joints = limited_joints

    def _pose_goal(self, T_goal: Dict[str, np.ndarray]) -> Dict[str, ArrayLike]:
        pos = {}
        for u, T_goal_u in T_goal.items():
            v = AUX_PREFIX + u[1:]
            pos[u] = T_goal_u[:3, 3]
            pos[v] = (T_goal_u @ trans_axis(self.axis_length, "z"))[:3, 3]
        return pos

    def joint_variables(
        self, G: nx.DiGraph, T_final: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, float]:
        """
        Finds the set of decision variables corresponding to the
        graph realization G.
        """
        tol = 1e-10
        axis_length = self.axis_length
        end_effectors = self.robot.end_effectors
        kinematic_map = self.robot.kinematic_map

        T = {}
        T[ROOT] = self.robot.T_base

        # resolve scale
        x_hat = G.nodes["x"][POS] - G.nodes["p0"][POS]
        y_hat = G.nodes["y"][POS] - G.nodes["p0"][POS]
        z_hat = G.nodes["q0"][POS] - G.nodes["p0"][POS]

        # resolve rotation and translation
        x = normalize(x_hat)
        y = normalize(y_hat)
        z = normalize(z_hat)
        R = np.vstack((x, -y, z)).T
        B = np.eye(4)
        B[:3, :3] = R
        B[:3, 3] = G.nodes[ROOT][POS]
        B_inv = SE3.inverse(B)

        omega_z = skew(np.array([0, 0, 1]))

        theta = {}
        for ee in end_effectors:
            k_map = kinematic_map[ROOT][ee]
            for idx in range(1, len(k_map)):
                cur, aux_cur = k_map[idx], f"q{k_map[idx][1:]}"
                pred, aux_pred = (k_map[idx - 1], f"q{k_map[idx-1][1:]}")

                T_prev = T[pred]

                T_prev_0 = self.robot.nodes[pred]["T0"]
                T_0 = self.robot.nodes[cur]["T0"]
                T_0_q = self.robot.nodes[cur]["T0"] @ trans_axis(axis_length, "z")
                T_rel = SE3.inverse(T_prev_0) @ T_0
                ps_0 = (SE3.inverse(T_prev_0) @ T_0)[:3, 3]
                qs_0 = (SE3.inverse(T_prev_0) @ T_0_q)[:3, 3]

                # predicted p and q expressed in previous frame
                p = B_inv[:3, :3] @ G.nodes[cur][POS] + B_inv[:3, 3]
                qnorm = G.nodes[cur][POS] + (
                    G.nodes[aux_cur][POS] - G.nodes[cur][POS]
                ) / la.norm(G.nodes[aux_cur][POS] - G.nodes[cur][POS])
                q = B_inv[:3, :3] @ qnorm + B_inv[:3, 3]

                T_prev_inv = SE3.inverse(T_prev)
                ps = T_prev_inv[:3, :3] @ (p - T_prev[:3, 3])
                qs = T_prev_inv[:3, :3] @ (q - T_prev[:3, 3])

                theta[cur] = arctan2(
                    -qs_0.dot(omega_z).dot(qs),
                    qs_0.dot(omega_z.dot(omega_z.T)).dot(qs),
                )

                T[cur] = (T_prev @ rot_axis(theta[cur], "z")) @ T_rel

            if (T_final is not None) and (
                la.norm(cross(T_rel[:3, 3], np.asarray([0, 0, 1]))) < tol
            ):
                T_th = SE3.inverse(T[cur]) @ T_final[ee]
                theta[ee] = wraptopi(theta[ee] + arctan2(T_th[1, 0], T_th[0, 0]))

        return theta

    def get_pose(self, joint_angles: Dict[str, float], query_node: str) -> np.ndarray:
        T = self.robot.pose(joint_angles, query_node)

        if query_node[0] == AUX_PREFIX:
            T_trans = trans_axis(self.axis_length, "z")
            T = T @ T_trans

        return T

    def distance_bounds_from_sampling(self):
        robot = self.robot
        ids = self.node_ids
        q_rand = robot.random_configuration()
        D_min = self.distance_matrix_from_joints(q_rand)
        D_max = self.distance_matrix_from_joints(q_rand)

        for _ in range(2000):
            q_rand = robot.random_configuration()
            D_rand = self.distance_matrix_from_joints(q_rand)
            D_max[D_rand > D_max] = D_rand[D_rand > D_max]
            D_min[D_rand < D_min] = D_rand[D_rand < D_min]

        for idx in range(len(D_max)):
            for jdx in range(len(D_max)):
                e1 = ids[idx]
                e2 = ids[jdx]
                self.add_edge(e1, e2)
                self[e1][e2][LOWER] = sqrt(D_min[idx, jdx])
                self[e1][e2][UPPER] = sqrt(D_max[idx, jdx])
                if abs(D_max[idx, jdx] - D_min[idx, jdx]) < 1e-5:
                    self[e1][e2][DIST] = abs(D_max[idx, jdx] - D_min[idx, jdx])


if __name__ == "__main__":
    import graphik
    from graphik.utils.roboturdf import RobotURDF

    n = 6
    ub = (pi) * np.ones(n)
    lb = -ub
    modified_dh = False

    fname = graphik.__path__[0] + "/robots/urdfs/ur10_mod.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/lwa4p.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/lwa4d.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/panda_arm.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/kuka_iiwr.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/kuka_lwr.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/jaco2arm6DOF_no_hand.urdf"

    urdf_robot = RobotURDF(fname)
    robot = urdf_robot.make_Revolute3d(ub, lb)  # make the Revolute class from a URDF
    graph = ProblemGraphRevolute(robot)
    print(graph.nodes(data=True))
    print(graph.base_nodes)
    print(graph.structure_nodes)
    print(graph.end_effector_nodes)
