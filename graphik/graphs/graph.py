"""Unified ProblemGraph class parameterized by dim in {2, 3}."""
from math import sqrt, atan2
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import numpy.linalg as la
from numpy import cos, pi, arctan2, cross
from numpy.typing import ArrayLike
from pymlg.numpy import SE3

from graphik.robots.robot import Robot
from graphik.utils import *  # noqa: F401,F403  (existing convention; brings constants and helpers into scope: ROOT, POS, TYPE, BASE, ROBOT, END_EFFECTOR, OBSTACLE, BOUNDED, BELOW, ABOVE, LOWER, UPPER, DIST, MAIN_PREFIX, AUX_PREFIX, TRANSFORM, trans_axis, rot_axis, skew, normalize, wraptopi, R2, best_fit_transform, level2_descendants, max_min_distance_revolute, distance_matrix_from_graph, adjacency_matrix_from_graph, graph_complete_edges)


def _bounded_tags(limit) -> List[str]:
    return [limit] if limit in (BELOW, ABOVE) else []


class ProblemGraph(nx.DiGraph):
    """
    Graph with positions and distance bounds describing an IK problem.
    Reads `dim` from the supplied robot. 2D and 3D share base class plumbing
    (EDM completion, obstacles, anchor APIs); the construction loops and a
    handful of dim-specific methods branch on `self.dim`.
    """

    def __init__(self, robot: Robot, params: Optional[Dict] = None):
        params = params or {}
        super().__init__(
            robot=robot,
            dim=robot.dim,
            axis_length=params.get("axis_length", 1),
        )

        base = self._build_base_subgraph()
        structure = self._build_structure_subgraph()
        composition = nx.compose(base, structure)
        self.add_nodes_from(composition.nodes(data=True))
        self.add_edges_from(composition.edges(data=True))

        self.set_limits()
        self.root_angle_limits()

    # ------------------------------------------------------------------
    # Properties (unchanged from the previous ProblemGraph base class)
    # ------------------------------------------------------------------
    @property
    def base_nodes(self) -> List[str]:
        try:
            return self._base_nodes
        except AttributeError:
            self._base_nodes = [n for n, d in self.nodes(data=TYPE) if BASE in d]
            return self._base_nodes

    @property
    def structure_nodes(self) -> List[str]:
        try:
            return self._structure_nodes
        except AttributeError:
            self._structure_nodes = [n for n, d in self.nodes(data=TYPE) if ROBOT in d]
            return self._structure_nodes

    @property
    def end_effector_nodes(self) -> List[str]:
        try:
            return self._end_effector_nodes
        except AttributeError:
            self._end_effector_nodes = [n for n, d in self.nodes(data=TYPE) if END_EFFECTOR in d]
            return self._end_effector_nodes

    @property
    def base(self) -> nx.DiGraph:
        return self.to_directed(as_view=True).subgraph(self.base_nodes)

    @property
    def structure(self) -> nx.DiGraph:
        return self.to_directed(as_view=True).subgraph(self.structure_nodes)

    @property
    def robot(self) -> Robot:
        return self.graph["robot"]

    @property
    def dim(self) -> int:
        return self.graph["dim"]

    @property
    def axis_length(self) -> float:
        return self.graph["axis_length"]

    @property
    def node_ids(self) -> List[str]:
        return list(self.nodes())

    # ------------------------------------------------------------------
    # Realization / matrix views (unchanged)
    # ------------------------------------------------------------------
    def realization(self, joint_angles: Dict[str, float]) -> nx.DiGraph:
        T_all = self.robot.get_all_poses(joint_angles)
        return self.from_pos(self._pose_goal(T_all))

    def distance_matrix(self) -> ArrayLike:
        return distance_matrix_from_graph(self)

    def distance_matrix_from_joints(self, joint_angles: Dict[str, float]) -> ArrayLike:
        return distance_matrix_from_graph(self.realization(joint_angles))

    def adjacency_matrix(self) -> ArrayLike:
        return adjacency_matrix_from_graph(self)

    def from_pos(self, P: Dict, dist: bool = True, overwrite: bool = False) -> nx.DiGraph:
        G = self.to_directed()
        for name, pos in P.items():
            if name in G:
                G.nodes[name][POS] = pos
        if dist:
            G = graph_complete_edges(G, overwrite=overwrite)
        return G

    def from_pose(self, T_goal) -> nx.DiGraph:
        if not isinstance(T_goal, dict):
            T_goal = {self.robot.end_effectors[0]: T_goal}
        return self.from_pos(self._pose_goal(T_goal))

    # ------------------------------------------------------------------
    # Anchor / obstacle APIs (unchanged from the previous base class)
    # ------------------------------------------------------------------
    def add_anchor_node(self, name: str, data: Dict[str, Any]):
        if POS not in data:
            raise KeyError("Node needs to have a position to be added.")
        self.add_nodes_from([(name, data)])
        for nname, ndata in self.nodes(data=True):
            if POS in ndata and nname != name:
                dist = la.norm(ndata[POS] - data[POS])
                self.add_edge(nname, name)
                self[nname][name][DIST] = dist
                self[nname][name][LOWER] = dist
                self[nname][name][UPPER] = dist
                self[nname][name][BOUNDED] = []

    def add_spherical_obstacle(self, name: str, position: ArrayLike, radius: float):
        self.add_anchor_node(name, {POS: position, TYPE: [OBSTACLE]})
        for node, node_type in self.nodes(data=TYPE):
            if (
                ROBOT in node_type
                and BASE not in node_type
                and node[0] == MAIN_PREFIX
            ):
                self.add_edge(node, name)
                self[node][name][BOUNDED] = [BELOW]
                self[node][name][LOWER] = radius
                self[node][name][UPPER] = 100

    def clear_obstacles(self):
        node_types = nx.get_node_attributes(self, TYPE)
        obstacles = [n for n, t in node_types.items() if t == [OBSTACLE]]
        self.remove_nodes_from(obstacles)

    def check_distance_limits(self, G: nx.DiGraph, tol=1e-10) -> List[Dict[str, List[Any]]]:
        typ = nx.get_node_attributes(self, name=TYPE)
        broken_limits = []
        for u, v, data in self.edges(data=True):
            bounds = data.get(BOUNDED, [])
            if BELOW in bounds or ABOVE in bounds:
                if G[u][v][DIST] < data[LOWER] - tol:
                    bl = {}
                    if (ROBOT in typ[u] and OBSTACLE in typ[v]) or (
                        OBSTACLE in typ[u] and ROBOT in typ[v]
                    ):
                        bl["edge"] = (u, v)
                        bl["value"] = G[u][v][DIST] - data[LOWER]
                        bl["type"] = OBSTACLE
                        bl["side"] = LOWER
                        broken_limits += [bl]
                    if ROBOT in typ[u] and ROBOT in typ[v]:
                        bl["edge"] = (u, v)
                        bl["value"] = G[u][v][DIST] - data[LOWER]
                        bl["type"] = "joint"
                        bl["side"] = LOWER
                        broken_limits += [bl]
                if G[u][v][DIST] > data[UPPER] + tol:
                    bl = {}
                    if (ROBOT in typ[u] and OBSTACLE in typ[v]) or (
                        OBSTACLE in typ[u] and ROBOT in typ[v]
                    ):
                        bl["edge"] = (u, v)
                        bl["value"] = G[u][v][DIST] - data[UPPER]
                        bl["type"] = OBSTACLE
                        bl["side"] = UPPER
                        broken_limits += [bl]
                    if ROBOT in typ[u] and ROBOT in typ[v]:
                        bl["edge"] = (u, v)
                        bl["value"] = G[u][v][DIST] - data[UPPER]
                        bl["type"] = "joint"
                        bl["side"] = UPPER
                        broken_limits += [bl]
        return broken_limits

    def distance_bound_matrices(self) -> Tuple[ArrayLike, ArrayLike]:
        n_nodes = self.number_of_nodes()
        L = np.zeros([n_nodes, n_nodes])
        U = np.zeros([n_nodes, n_nodes])
        for e1, e2, data in self.edges(data=True):
            if BOUNDED in data:
                udx = self.node_ids.index(e1)
                vdx = self.node_ids.index(e2)
                bounds = data[BOUNDED]
                if BELOW in bounds:
                    L[udx, vdx] = data[LOWER] ** 2
                    L[vdx, udx] = L[udx, vdx]
                if ABOVE in bounds:
                    U[udx, vdx] = data[UPPER] ** 2
                    U[vdx, udx] = U[udx, vdx]
        return L, U

    # ------------------------------------------------------------------
    # Dim-aware base + structure subgraph construction
    # ------------------------------------------------------------------
    def _anchor_positions(self) -> Dict[str, np.ndarray]:
        """Anchor (base) node positions — preserves the existing 2D and 3D conventions."""
        L = self.axis_length
        if self.dim == 2:
            return {
                "p0": np.array([0, 0]),
                "x": np.array([-1, 0]),
                "y": np.array([0, 1]),
            }
        else:
            return {
                "p0": np.array([0, 0, 0]),
                "x": np.array([L, 0, 0]),
                "y": np.array([0, -L, 0]),
                "q0": np.array([0, 0, L]),
            }

    def _anchor_types(self) -> Dict[str, List[str]]:
        """Per-anchor TYPE lists — preserves existing 2D and 3D orderings exactly."""
        if self.dim == 2:
            return {
                "p0": [BASE, ROBOT],
                "x": [BASE],
                "y": [BASE],
            }
        else:
            return {
                "p0": [ROBOT, BASE],
                "x": [BASE],
                "y": [BASE],
                "q0": [ROBOT, BASE],
            }

    def _build_base_subgraph(self) -> nx.DiGraph:
        pos = self._anchor_positions()
        types = self._anchor_types()
        if self.dim == 2:
            edges = [("p0", "x"), ("p0", "y"), ("x", "y")]
        else:
            edges = [
                ("p0", "x"), ("p0", "y"), ("p0", "q0"),
                ("x", "y"), ("y", "q0"), ("q0", "x"),
            ]
        base = nx.DiGraph(edges)
        for name in pos:
            base.add_node(name, **{POS: pos[name], TYPE: types[name]})
        for u, v in base.edges():
            d = la.norm(base.nodes[u][POS] - base.nodes[v][POS])
            base[u][v][DIST] = d
            base[u][v][LOWER] = d
            base[u][v][UPPER] = d
            base[u][v][BOUNDED] = []
        return base

    def _joint_nodes(self, cur: str) -> List[Tuple[str, np.ndarray]]:
        """List of (node_name, T0_position) pairs for the given joint."""
        T0 = self.robot.nodes[cur]["T0"]
        if self.dim == 2:
            return [(cur, T0[:2, 2])]
        else:
            trans_z = trans_axis(self.axis_length, "z")
            aux_name = AUX_PREFIX + cur[1:]
            return [
                (cur, T0[:3, 3]),
                (aux_name, (T0 @ trans_z)[:3, 3]),
            ]

    def structure_graph(self) -> nx.DiGraph:
        """Standalone copy of the robot structure subgraph (exact-distance
        edges only, no joint-limit bound edges). Public API used by the SDP
        solvers; equivalent to the pre-unification ``structure_graph()``."""
        return self._build_structure_subgraph()

    def _build_structure_subgraph(self) -> nx.DiGraph:
        structure = nx.empty_graph(create_using=nx.DiGraph)
        end_effectors = self.robot.end_effectors
        for ee in end_effectors:
            k_map = self.robot.kinematic_map[ROOT][ee]
            for idx in range(len(k_map)):
                cur = k_map[idx]
                cur_nodes = self._joint_nodes(cur)

                joint_type = [ROBOT]
                if cur == ROOT:
                    joint_type = joint_type + [BASE]
                if cur in end_effectors:
                    joint_type = joint_type + [END_EFFECTOR]

                for name, pos in cur_nodes:
                    structure.add_node(name, **{POS: pos, TYPE: list(joint_type)})

                # Intra-joint edge (only meaningful in 3D: p_i — q_i)
                for i in range(len(cur_nodes)):
                    for j in range(i + 1, len(cur_nodes)):
                        u_name, u_pos = cur_nodes[i]
                        v_name, v_pos = cur_nodes[j]
                        d = la.norm(u_pos - v_pos)
                        structure.add_edge(
                            u_name, v_name,
                            **{DIST: d, LOWER: d, UPPER: d, BOUNDED: []},
                        )

                # Inter-joint complete-bipartite edges to predecessor's nodes
                if idx != 0:
                    pred = k_map[idx - 1]
                    pred_nodes = self._joint_nodes(pred)

                    # 2D-only: propagate END_EFFECTOR tag to predecessor's single node.
                    # (3D's existing convention does NOT propagate, so this branch is dim-specific.)
                    if self.dim == 2 and cur in end_effectors:
                        pred_name = pred_nodes[0][0]
                        if END_EFFECTOR not in structure.nodes[pred_name][TYPE]:
                            structure.nodes[pred_name][TYPE].append(END_EFFECTOR)

                    for u_name, u_pos in pred_nodes:
                        for v_name, v_pos in cur_nodes:
                            d = la.norm(u_pos - v_pos)
                            structure.add_edge(
                                u_name, v_name,
                                **{DIST: d, LOWER: d, UPPER: d, BOUNDED: []},
                            )

        # Strip POS from structure nodes (it was used only to compute distances)
        for u in structure.nodes:
            if POS in structure.nodes[u]:
                del structure.nodes[u][POS]

        return structure

    # ------------------------------------------------------------------
    # Distance bounds — dim-dispatched
    # ------------------------------------------------------------------
    def root_angle_limits(self):
        if self.dim == 2:
            self._root_angle_limits_2d()
        else:
            self._root_angle_limits_3d()

    def set_limits(self):
        if self.dim == 2:
            self._set_limits_2d()
        else:
            self._set_limits_3d()

    # --- 2D bodies (verbatim from the old ProblemGraphPlanar) ---
    def _root_angle_limits_2d(self):
        ax = "x"
        S = self.structure
        l1 = la.norm(self.nodes[ax][POS])
        for node in S.successors(ROOT):
            if DIST in S[ROOT][node]:
                l2 = S[ROOT][node][DIST]
                lb = self.robot.lb[node]
                ub = self.robot.ub[node]
                lim = max(abs(ub), abs(lb))
                self.add_edge(ax, node)
                self[ax][node][UPPER] = l1 + l2
                self[ax][node][LOWER] = sqrt(
                    l1 ** 2 + l2 ** 2 - 2 * l1 * l2 * cos(pi - lim)
                )
                self[ax][node][BOUNDED] = [BELOW]

    def _set_limits_2d(self):
        S = self.structure
        for u in S:
            for v in (suc for suc in S.successors(u) if suc):
                self[u][v][UPPER] = S[u][v][DIST]
                self[u][v][LOWER] = S[u][v][DIST]
            for v in (des for des in level2_descendants(S, u) if des):
                ids = self.robot.kinematic_map[u][v]
                l1 = self.robot.l[ids[1]]
                l2 = self.robot.l[ids[2]]
                lb = self.robot.lb[ids[2]]
                ub = self.robot.ub[ids[2]]
                lim = max(abs(ub), abs(lb))
                self.add_edge(u, v)
                self[u][v][UPPER] = l1 + l2
                self[u][v][LOWER] = sqrt(
                    l1 ** 2 + l2 ** 2 - 2 * l1 * l2 * cos(pi - lim)
                )
                self[u][v][BOUNDED] = [BELOW]

    # --- 3D bodies (verbatim from the old ProblemGraphRevolute) ---
    def _root_angle_limits_3d(self):
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
                self[base_node][node][BOUNDED] = _bounded_tags(limit)
                self[base_node][node][UPPER] = d_max
                self[base_node][node][LOWER] = d_min

    def _set_limits_3d(self):
        S = self.structure
        robot = self.robot
        kinematic_map = self.robot.kinematic_map
        T_axis = trans_axis(self.axis_length, "z")
        end_effectors = self.robot.end_effectors
        upper_limits = self.robot.ub

        limited_joints = []
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
                    self[ids[0]][ids[1]][BOUNDED] = _bounded_tags(limit)
                    self[ids[0]][ids[1]][UPPER] = d_max
                    self[ids[0]][ids[1]][LOWER] = d_min

        self.limited_joints = limited_joints

    # ------------------------------------------------------------------
    # Goal-pose ↔ position-dict conversion (dim-dispatched)
    # ------------------------------------------------------------------
    def _pose_goal(self, T_goal: Dict[str, np.ndarray]) -> Dict[str, ArrayLike]:
        if self.dim == 2:
            pos = {}
            for u, T_goal_u in T_goal.items():
                for v in self.structure.predecessors(u):
                    if DIST in self[v][u]:
                        d = self[v][u][DIST]
                        z = T_goal_u[:2, 0]
                        pos[u] = T_goal_u[:2, 2]
                        pos[v] = T_goal_u[:2, 2] - z * d
            return pos
        else:
            pos = {}
            for u, T_goal_u in T_goal.items():
                v = AUX_PREFIX + u[1:]
                pos[u] = T_goal_u[:3, 3]
                pos[v] = (T_goal_u @ trans_axis(self.axis_length, "z"))[:3, 3]
            return pos

    # ------------------------------------------------------------------
    # Joint variables from realization (dim-dispatched)
    # ------------------------------------------------------------------
    def joint_variables(
        self,
        G: nx.Graph,
        T_final: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, float]:
        if self.dim == 2:
            joint_variables = {}
            R_, t_ = best_fit_transform(
                np.vstack((G.nodes[ROOT][POS], G.nodes["x"][POS], G.nodes["y"][POS])),
                np.vstack(([0, 0], [-1, 0], [0, 1])),
            )
            R = {ROOT: np.eye(2)}
            for u, v, dat in self.structure.edges(data=DIST):
                if dat:
                    diff_uv = R_ @ (G.nodes[v][POS] - G.nodes[u][POS])
                    len_uv = np.linalg.norm(diff_uv)
                    sol = R[u].T @ (diff_uv / len_uv)
                    theta_idx = atan2(sol[1], sol[0])
                    joint_variables[v] = wraptopi(theta_idx)
                    Rz = R2(theta_idx)
                    R[v] = R[u] @ Rz
            return joint_variables
        else:
            tol = 1e-10
            axis_length = self.axis_length
            end_effectors = self.robot.end_effectors
            kinematic_map = self.robot.kinematic_map

            T = {ROOT: self.robot.T_base}

            x_hat = G.nodes["x"][POS] - G.nodes["p0"][POS]
            y_hat = G.nodes["y"][POS] - G.nodes["p0"][POS]
            z_hat = G.nodes["q0"][POS] - G.nodes["p0"][POS]

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

    # ------------------------------------------------------------------
    # Pose lookup (dim-dispatched)
    # ------------------------------------------------------------------
    def get_pose(
        self,
        joint_angles: Dict[str, float],
        query_node: Union[List[str], str],
    ) -> np.ndarray:
        T = self.robot.pose(joint_angles, query_node)
        if self.dim == 3 and isinstance(query_node, str) and query_node[0] == AUX_PREFIX:
            T_trans = trans_axis(self.axis_length, "z")
            T = T @ T_trans
        return T

    # ------------------------------------------------------------------
    # 3D-only sampling helper (keep dim-agnostic — works for any dim)
    # ------------------------------------------------------------------
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
