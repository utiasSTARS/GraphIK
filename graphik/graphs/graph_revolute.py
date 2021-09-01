from typing import Dict, List, Any
import numpy as np
import numpy.linalg as la
from graphik.robots import RobotRevolute
from graphik.graphs.graph_base import RobotGraph
from graphik.utils import *
from liegroups.numpy import SE3
from liegroups.numpy.se3 import SE3Matrix
from liegroups.numpy.so3 import SO3Matrix
import networkx as nx
from numpy import cos, pi, sqrt, arctan2, cross


class RobotRevoluteGraph(RobotGraph):
    def __init__(
        self,
        robot: RobotRevolute,
        params: Dict = {},
    ):

        self.dim = 3
        self.robot = robot
        self.axis_length = params.get("axis_length", 1)  # distance between p and q

        self.base = self.base_subgraph()
        self.structure = self.structure_graph()
        self.set_limits()

        self.directed = nx.compose(self.base, self.structure)
        self.directed = self.root_angle_limits(self.directed)
        super(RobotRevoluteGraph, self).__init__()

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
                ("p0", {POS: np.array([0, 0, 0])}),
                ("x", {POS: np.array([axis_length, 0, 0]), TYPE: "base"}),
                ("y", {POS: np.array([0, -axis_length, 0]), TYPE: "base"}),
                ("q0", {POS: np.array([0, 0, axis_length])}),
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
        end_effectors = self.robot.end_effectors
        kinematic_map = self.robot.kinematic_map

        structure = nx.empty_graph(create_using=nx.DiGraph)

        for ee in end_effectors:
            k_map = kinematic_map[ROOT][ee[0]]
            for idx in range(len(k_map)):
                cur, aux_cur = k_map[idx], f"q{k_map[idx][1:]}"
                cur_pos, aux_cur_pos = (
                    robot.nodes[cur]["T0"].trans,
                    robot.nodes[cur]["T0"].dot(trans_z).trans,
                )
                dist = la.norm(cur_pos - aux_cur_pos)

                # Add nodes for joint and edge between them
                structure.add_nodes_from(
                    [(cur, {POS: cur_pos}), (aux_cur, {POS: aux_cur_pos})]
                )
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
                    # structure[pred][cur][TRANSFORM] = (
                    #     robot.nodes[pred]["T0"].inv().dot(robot.nodes[cur]["T0"])
                    # )

        # Delete positions used for weights
        for u in structure.nodes:
            del structure.nodes[u][POS]

        # Set node type to robot
        nx.set_node_attributes(structure, ROBOT, TYPE)

        return structure

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
            k_map = kinematic_map[ROOT][ee[0]]
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
                        T0 = T0.dot(T_axis)
                    if AUX_PREFIX in ids[1]:
                        T2 = T2.dot(T_axis)

                    N = T1.as_matrix()[0:3, 2]
                    C = T1.trans + (N.dot(T2.trans - T1.trans)) * N
                    r = la.norm(T2.trans - C)
                    P = T0.trans
                    d_max, d_min = max_min_distance_revolute(r, P, C, N)

                    d = la.norm(T2.trans - T0.trans)
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

                        T_rel = T1.inv().dot(T2)

                        d_limit = la.norm(T1.dot(rot_limit).dot(T_rel).trans - T0.trans)

                        if limit == ABOVE:
                            d_max = d_limit
                        else:
                            d_min = d_limit

                        limited_joints += [cur]

                    S.add_edge(ids[0], ids[1])
                    if d_max == d_min:
                        S[ids[0]][ids[1]][DIST] = d_max
                    S[ids[0]][ids[1]][BOUNDED] = [limit]
                    S[ids[0]][ids[1]][UPPER] = d_max
                    S[ids[0]][ids[1]][LOWER] = d_min

        self.limited_joints = limited_joints

    def root_angle_limits(self, G: nx.DiGraph) -> nx.DiGraph:
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
                T0 = SE3.from_matrix(np.identity(4))
                T0.trans = G.nodes[base_node][POS]
                if node[0] == "p":
                    T2 = robot.nodes["p1"]["T0"]
                else:
                    T2 = robot.nodes["p1"]["T0"].dot(T_axis)

                N = T1.as_matrix()[0:3, 2]
                C = T1.trans + (N.dot(T2.trans - T1.trans)) * N
                r = np.linalg.norm(T2.trans - C)
                P = T0.trans
                d_max, d_min = max_min_distance_revolute(r, P, C, N)
                d = np.linalg.norm(T2.trans - T0.trans)

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
                        T_rel = T1.inv().dot(robot.nodes["p1"]["T0"])
                    else:
                        T_rel = T1.inv().dot(robot.nodes["p1"]["T0"].dot(T_axis))

                    d_limit = la.norm(
                        T1.dot(rot_axis(upper_limits["p1"], "z")).dot(T_rel).trans
                        - T0.trans
                    )

                    if limit == ABOVE:
                        d_max = d_limit
                    else:
                        d_min = d_limit
                    limited_joints += ["p1"]  # joint at p0 is limited

                G.add_edge(base_node, node)
                if d_max == d_min:
                    G[base_node][node][DIST] = d_max
                G[base_node][node][BOUNDED] = [limit]
                G[base_node][node][UPPER] = d_max
                G[base_node][node][LOWER] = d_min

        return G

    def realization(self, joint_angles: Dict[str, float]) -> nx.DiGraph:
        """
        Given a dictionary of joint variables generate a representative graph.
        This graph will be fully connected.
        """

        axis_length = self.axis_length
        T_all = self.robot.get_all_poses(joint_angles)

        P = {}
        for node, T in T_all.items():
            P[node] = T.trans
            P["q" + node[1:]] = T.dot(trans_axis(axis_length, "z")).trans
        return self.complete_from_pos(P)

    def joint_variables(
        self, G: nx.DiGraph, T_final: Dict[str, SE3] = None
    ) -> Dict[str, Any]:
        """
        Calculate joint angles from a complete set of point positions.
        """
        # TODO: make this more readable
        tol = 1e-10
        q_zero = list_to_variable_dict(self.robot.n * [0])
        axis_length = self.axis_length
        end_effectors = self.robot.end_effectors
        kinematic_map = self.robot.kinematic_map

        T = {}
        T[ROOT] = self.robot.nodes[ROOT]

        # resolve scale
        x_hat = G.nodes["x"][POS] - G.nodes["p0"][POS]
        y_hat = G.nodes["y"][POS] - G.nodes["p0"][POS]
        z_hat = G.nodes["q0"][POS] - G.nodes["p0"][POS]
        scale = np.roots(
            [
                x_hat.dot(x_hat) + y_hat.dot(y_hat) + z_hat.dot(z_hat),
                -2
                * (
                    np.linalg.norm(x_hat)
                    + np.linalg.norm(y_hat)
                    + np.linalg.norm(z_hat)
                ),
                3,
            ]
        )
        scale = scale[0].real

        # resolve rotation and translation
        x = normalize(x_hat)
        y = normalize(y_hat)
        z = normalize(z_hat)
        R = np.vstack((x, -y, z)).T
        B = SE3Matrix(SO3Matrix(R), scale * G.nodes[ROOT][POS])

        theta = {}

        for ee in end_effectors:
            k_map = kinematic_map[ROOT][ee[0]]
            for idx in range(1, len(k_map)):
                cur, aux_cur = k_map[idx], f"q{k_map[idx][1:]}"
                pred, aux_pred = (k_map[idx - 1], f"q{k_map[idx-1][1:]}")

                T_prev = T[pred]

                T_prev_0 = self.get_pose(q_zero, pred)
                T_0 = self.get_pose(q_zero, cur)
                T_rel = T_prev_0.inv().dot(T_0)
                T_0_q = self.get_pose(q_zero, cur).dot(trans_axis(axis_length, "z"))
                T_rel_q = T_prev_0.inv().dot(T_0_q)

                p = B.inv().dot(scale * G.nodes[cur][POS]) - T_prev.trans
                qnorm = G.nodes[cur][POS] + (
                    G.nodes[aux_cur][POS] - G.nodes[cur][POS]
                ) / np.linalg.norm(G.nodes[aux_cur][POS] - G.nodes[cur][POS])
                q = B.inv().dot(scale * qnorm) - T_prev.trans
                # q = B.inv().dot(scale*G.nodes[aux_cur][POS]) - T_prev.trans
                ps = T_prev.inv().as_matrix()[:3, :3].dot(p)  # in prev. joint frame
                qs = T_prev.inv().as_matrix()[:3, :3].dot(q)  # in prev. joint frame

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
                and la.norm(cross(T_rel.trans, np.asarray([0, 0, 1]))) < tol
            ):
                T_th = (T[cur]).inv().dot(T_final[ee[0]]).as_matrix()
                theta[ee[0]] += arctan2(T_th[1, 0], T_th[0, 0])

        return theta

    def get_pose(self, joint_angles: dict, query_node: str) -> SE3:
        T = self.robot.pose(joint_angles, query_node)

        if query_node[0] == AUX_PREFIX:
            T_trans = trans_axis(self.axis_length, "z")
            T = T.dot(T_trans)

        return T

    def distance_bounds_from_sampling(self):
        robot = self.robot
        G = self.directed
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
                G.add_edge(e1, e2)
                G[e1][e2][LOWER] = np.sqrt(D_min[idx, jdx])
                G[e1][e2][UPPER] = np.sqrt(D_max[idx, jdx])
                if abs(D_max[idx, jdx] - D_min[idx, jdx]) < 1e-5:
                    G[e1][e2][DIST] = abs(D_max[idx, jdx] - D_min[idx, jdx])


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
    graph = RobotRevoluteGraph(robot)
    print(graph.directed.edges(data=True))
    # import timeit

    # print(
    #     max(
    #         timeit.repeat(
    #             "graph.realization(robot.random_configuration())",
    #             globals=globals(),
    #             number=1,
    #             repeat=1000,
    #         )
    #     )
    # )
