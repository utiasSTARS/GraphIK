from typing import Dict, List, Any, Optional, Union
import numpy as np
import numpy.linalg as la
from graphik.robots import RobotRevolute
from graphik.graphs.graph_base import ProblemGraph
from graphik.utils import *
from liegroups.numpy import SE3
from liegroups.numpy.se3 import SE3Matrix
from liegroups.numpy.so3 import SO3Matrix
import networkx as nx
from numpy import cos, pi, sqrt, arctan2, cross

class ProblemGraphRevolute(ProblemGraph):
    def __init__(
        self,
        robot: RobotRevolute,
        params: Dict = {},
    ):
        super(ProblemGraphRevolute, self).__init__(robot, params)

        # initialize the base and structure subgraphs
        base = self.init_base_subgraph()
        structure = self.init_structure_subgraph()
        composition = nx.compose(base, structure)

        self.add_nodes_from(composition.nodes(data=True))
        self.add_edges_from(composition.edges(data=True))

        self.set_limits()
        self.root_angle_limits()

    def init_base_subgraph(self) -> nx.DiGraph:
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

    def init_structure_subgraph(self):
        trans_z = trans_axis(self.axis_length, "z")
        robot = self.robot

        structure = nx.empty_graph(create_using=nx.DiGraph)

        for ee in robot.end_effectors:
            k_map = robot.kinematic_map[ROOT][ee]
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

        # Delete positions used for weights
        for u in structure.nodes:
            del structure.nodes[u][POS]

        # Set node type to robot
        nx.set_node_attributes(structure, [ROBOT], TYPE)
        structure.nodes[ROOT][TYPE] = [ROBOT, BASE]
        structure.nodes["q0"][TYPE] = [ROBOT, BASE]

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
                T0 = SE3.from_matrix(np.identity(4))
                T0.trans = self.nodes[base_node][POS]
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

                    self.add_edge(ids[0], ids[1])
                    if d_max == d_min:
                        S[ids[0]][ids[1]][DIST] = d_max
                    self[ids[0]][ids[1]][BOUNDED] = [limit]
                    self[ids[0]][ids[1]][UPPER] = d_max
                    self[ids[0]][ids[1]][LOWER] = d_min

        self.limited_joints = limited_joints

    def _pose_goal(self, T_goal: Dict[str, SE3]) -> Dict[str, ArrayLike]:
        pos = {}
        for u, T_goal_u in T_goal.items():
            v = AUX_PREFIX + u[1:]
            pos[u] = T_goal_u.trans
            pos[v] = T_goal_u.dot(trans_axis(self.axis_length, "z")).trans
        return pos

    def joint_variables(self, G: nx.DiGraph, T_final: Optional[Dict[str, SE3]] = None) -> Dict[str, float]:
        """
        Finds the set of decision variables corresponding to the
        graph realization G.

        :param G: networkx.DiGraph with known vertex positions
        :param T_final: poses of end-effectors in case two final frames aligned along z
        :returns: Dictionary of joint angles
        :rtype: Dict[str, float]
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
        B = SE3Matrix(SO3Matrix(R), G.nodes[ROOT][POS])

        omega_z = skew(np.array([0,0,1]));

        theta = {}
        for ee in end_effectors:
            k_map = kinematic_map[ROOT][ee]
            for idx in range(1, len(k_map)):
                cur, aux_cur = k_map[idx], f"q{k_map[idx][1:]}"
                pred, aux_pred = (k_map[idx - 1], f"q{k_map[idx-1][1:]}")

                T_prev = T[pred]

                T_prev_0 = self.robot.nodes[pred]["T0"] # previous p xf at 0
                T_0 = self.robot.nodes[cur]["T0"] # cur p xf at 0
                T_0_q = self.robot.nodes[cur]["T0"].dot(trans_axis(axis_length, "z")) # cur q xf at 0
                T_rel = T_prev_0.inv().dot(T_0) # relative xf
                ps_0 = T_prev_0.inv().dot(T_0).trans # relative xf
                qs_0 = T_prev_0.inv().dot(T_0_q).trans # rel q xf

                # predicted p and q expressed in previous frame
                p = B.inv().dot(G.nodes[cur][POS])
                qnorm = G.nodes[cur][POS] + (
                    G.nodes[aux_cur][POS] - G.nodes[cur][POS]
                ) / la.norm(G.nodes[aux_cur][POS] - G.nodes[cur][POS])
                q = B.inv().dot(qnorm)
                ps = T_prev.inv().as_matrix()[:3, :3].dot(p - T_prev.trans)  # in prev. joint frame
                qs = T_prev.inv().as_matrix()[:3, :3].dot(q - T_prev.trans)  # in prev. joint frame

                theta[cur] = arctan2(-qs_0.dot(omega_z).dot(qs), qs_0.dot(omega_z.dot(omega_z.T)).dot(qs))

                T[cur] = (T_prev.dot(rot_axis(theta[cur], "z"))).dot(T_rel)

            # if the rotation axis of final joint is aligned with ee frame z axis,
            # get angle from EE pose if available
            if ((T_final is not None) and (la.norm(cross(T_rel.trans, np.asarray([0, 0, 1]))) < tol)):
                T_th = (T[cur]).inv().dot(T_final[ee]).as_matrix()
                theta[ee] = wraptopi(theta[ee] +  arctan2(T_th[1, 0], T_th[0, 0]))
        return theta

    def get_pose(self, joint_angles: Dict[str, float], query_node: str) -> SE3:
        T = self.robot.pose(joint_angles, query_node)
        if query_node[0] == AUX_PREFIX:
            T_trans = trans_axis(self.axis_length, "z")
            T = T.dot(T_trans)
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

def random_revolute_robot_graph(
    n: int, a_range=(-0.5, 0.5), d_range=(0, 0.5)
) -> ProblemGraphRevolute:

    params = {
        "a": [],
        "alpha": [],
        "d": [],
        "theta": [],
        "num_joints": n,
        "modified_dh": False,
    }

    # we fix the first joint for simplicity
    params["alpha"] += [-np.pi/2 + np.pi * np.random.randint(2)]
    params["a"] += [0]
    params["d"] += [(d_range[0] + (d_range[1] - d_range[0]) * np.random.rand()) * np.random.randint(2)]
    # params["theta"] += [(np.pi) * np.random.randint(2)]
    params["theta"] += [0]

    for _ in range(n-2):
        params["theta"] += [0]
        # if (len(params["alpha"]) > 1) and ((params["alpha"][-1] != 0) and (params["alpha"][-2] != 0)):
        #     params["alpha"] += [0]
        # elif (len(params["alpha"]) > 1) and ((params["alpha"][-1] == 0) and (params["alpha"][-2] == 0)):
        #     params["alpha"] += [-np.pi/2 + (np.pi) * np.random.randint(2)]
        # elif params["alpha"][-1] != 0:
        #     # if previous joint had alpha != 0, the next one is either opposite or 0
        #     params["alpha"] += [-params["alpha"][-1] * np.random.randint(2)]
        if (len(params["alpha"]) > 1) and ((params["alpha"][-1] == 0) and (params["alpha"][-2] == 0)):
            params["alpha"] += [-np.pi/2 + (np.pi) * np.random.randint(2)]
        else:
            # params["alpha"] += [(-np.pi/2 + np.pi * np.random.randint(2)) * (np.random.randint(2))]
            params["alpha"] += [(-np.pi/2 + np.pi * np.random.randint(2)) * (np.random.randint(4) > 0)]

        if params["alpha"][-1] != 0:
            params["a"] += [0]
            params["d"] += [(d_range[0] + (d_range[1] - d_range[0]) * np.random.rand()) * (np.random.randint(2))]
        else:
            params["a"] += [a_range[0] + (a_range[1] - a_range[0]) * np.random.rand()]
            params["d"] += [0]
            params["alpha"][-1] += np.pi * np.random.randint(2) # alpha can also be -pi instead of 0

    params["alpha"] += [0]
    params["a"] += [0]
    params["d"] += [(d_range[0] + (d_range[1] - d_range[0]) * np.random.rand()) * np.random.randint(2)]
    params["theta"] += [0]

    robot = RobotRevolute(params)
    graph = ProblemGraphRevolute(robot)
    return graph

if __name__ == "__main__":
    import graphik
    from graphik.utils.roboturdf import RobotURDF
    np.set_printoptions(precision=4, suppress=True)

    n = 7
    ub = (pi) * np.ones(n)
    lb = -ub
    modified_dh = False

    # fname = graphik.__path__[0] + "/robots/urdfs/ur10_mod.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/lwa4p.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/lwa4d.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/panda_arm.urdf"
    fname = graphik.__path__[0] + "/robots/urdfs/kuka_iiwr.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/kuka_lwr.urdf"
    # fname = graphik.__path__[0] + "/robots/urdfs/jaco2arm6DOF_no_hand.urdf"

    urdf_robot = RobotURDF(fname)
    robot = urdf_robot.make_Revolute3d(ub, lb)  # make the Revolute class from a URDF
    # graph = ProblemGraphRevolute(robot,)
    # q_init = graph.robot.random_configuration()
    # q_init[graph.robot.end_effectors[0]] = 0
    # G_init = graph.realization(q_init)
    # D = distance_matrix_from_graph(graph)
    # print(D)
    T_urdf = list(nx.get_edge_attributes(robot, "T").values())

    for idx in range(robot.n):
        print(T_urdf[idx].as_matrix())
        print("-------")
    print("-------")
    #
    # # UR10 coordinates for testing
    modified_dh = False
    a = [0, 0, 0, 0, 0, 0, 0]
    d = [0.34, 0, 0.40, 0, 0.40, 0, 0.126]
    al = [-np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0]
    th = [0, 0, 0, 0, 0, 0, 0]

    params = {
        "a": a,
        "alpha": al,
        "d": d,
        "theta": th,
        "modified_dh": modified_dh,
        "num_joints": 7,
    }
    robot = RobotRevolute(params)
    graph = ProblemGraphRevolute(robot)
    T_dh = list(nx.get_edge_attributes(robot, "T").values())
    for idx in range(robot.n):
        print(T_dh[idx].as_matrix())
        print("-------")
    print("-------")
    # #
    # graph = random_revolute_robot_graph(6)
    # robot = graph.robot
    # # robot = RobotRevolute(params)
    # T_dh = list(nx.get_edge_attributes(robot, "T").values())
    # for idx in range(robot.n):
    #     print(T_dh[idx].as_matrix())
    #     print("-------")
