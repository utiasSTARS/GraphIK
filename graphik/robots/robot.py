#!/usr/bin/env python3
"""Unified Robot class parameterized by dim in {2, 3}."""
from math import pi
from typing import Any, Dict, List, Tuple, Union

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike
from pymlg.numpy import SE2, SE3

from graphik.utils import (
    list_to_variable_dict,
    flatten,
    fk_3d,
    modified_fk_3d,
    fk_tree_2d,
)
from graphik.utils.constants import ROOT, TRANSFORM, MAIN_PREFIX


class Robot(nx.DiGraph):
    """
    Kinematic-tree robot with revolute joints. Workspace dimension is `dim ∈ {2, 3}`;
    SE(2) is used for 2D, SE(3) for 3D. Construction sources:
      - explicit `T_zero` dict;
      - `link_lengths` (2D-only);
      - DH params `(a, d, alpha, theta, modified_dh)` (3D-only).
    """

    def __init__(self, params: Dict):
        super().__init__()
        if "dim" not in params:
            raise KeyError("dim must be specified in params")
        self.dim = params["dim"]
        if self.dim not in (2, 3):
            raise ValueError(f"dim must be 2 or 3, got {self.dim}")
        self._SE = SE2 if self.dim == 2 else SE3

        self.lambdified = False
        self.params = params
        self.n = params["num_joints"]

        # Topology: tree of joints. If parents not provided, assume a chain.
        if "parents" in params:
            topology = nx.DiGraph(params["parents"])
        else:
            topology = nx.path_graph(
                [f"p{idx}" for idx in range(self.n + 1)], nx.DiGraph
            )
        self.add_nodes_from(topology.nodes())
        self.add_edges_from(topology.edges())

        self.kinematic_map = dict(nx.all_pairs_shortest_path(self))

        self.lb = params.get("joint_limits_lower", self.n * [-pi])
        self.ub = params.get("joint_limits_upper", self.n * [pi])
        nx.set_node_attributes(self, values=self.lb, name="lb")
        nx.set_node_attributes(self, values=self.ub, name="ub")

        # Build T_zero (frame poses at zero config)
        if "T_zero" in params:
            T_zero = params["T_zero"]
        elif "link_lengths" in params:
            T_zero = self.from_params()
        elif all(k in params for k in ("a", "d", "alpha", "theta", "modified_dh")):
            T_zero = self.from_dh_params(params)
        else:
            raise Exception("Robot description not provided.")

        nx.set_node_attributes(self, values=T_zero, name="T0")
        self.set_geometric_attributes()

    # ------------------------------------------------------------------
    # Properties (unchanged behaviour from the previous Robot base class)
    # ------------------------------------------------------------------
    @property
    def kinematic_map(self) -> dict:
        return self._kinematic_map

    @kinematic_map.setter
    def kinematic_map(self, kinematic_map: dict):
        self._kinematic_map = kinematic_map

    @property
    def joint_ids(self) -> List[str]:
        try:
            return self._joint_ids
        except AttributeError:
            self._joint_ids = list(self.kinematic_map.keys())
            return self._joint_ids

    @property
    def end_effectors(self) -> List:
        if not hasattr(self, "_end_effectors"):
            self._end_effectors = [x for x in self.nodes() if self.out_degree(x) == 0]
        return self._end_effectors

    @property
    def T_base(self) -> np.ndarray:
        try:
            return self._T_base
        except AttributeError:
            self._T_base = self.nodes[ROOT]["T0"]
        return self._T_base

    @property
    def limited_joints(self) -> List[str]:
        return self._limited_joints

    @limited_joints.setter
    def limited_joints(self, lim: List[str]):
        self._limited_joints = lim

    @property
    def ub(self) -> Dict[str, Any]:
        return self._ub

    @ub.setter
    def ub(self, ub: dict):
        self._ub = ub if type(ub) is dict else list_to_variable_dict(flatten([ub]))

    @property
    def lb(self) -> Dict[str, Any]:
        return self._lb

    @lb.setter
    def lb(self, lb: dict):
        self._lb = lb if type(lb) is dict else list_to_variable_dict(flatten([lb]))

    @property
    def spherical(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def random_configuration(self) -> Dict[str, float]:
        q = {}
        for key in self.joint_ids:
            if key != ROOT:
                q[key] = self.lb[key] + (self.ub[key] - self.lb[key]) * np.random.rand()
        return q

    def zero_configuration(self) -> Dict[str, float]:
        q = {}
        for key in self.joint_ids:
            if key != ROOT:
                q[key] = 0
        return q

    def get_all_poses(self, joint_angles: Dict[str, Any]) -> Dict[str, np.ndarray]:
        T = {ROOT: self.T_base}
        for ee in self.end_effectors:
            for node in self.kinematic_map[ROOT][ee][1:]:
                T[node] = self.pose(joint_angles, node)
        return T

    def end_effector_pos(self, q: Dict[str, float]) -> Dict[str, ArrayLike]:
        goals = {}
        for ee in self.end_effectors:
            for node in ee:
                goals[node] = self.pose(q, node)[:-1, -1]
        return goals

    # ------------------------------------------------------------------
    # Dim-aware kinematics
    # ------------------------------------------------------------------
    def _screw_axis_from_T0(self, T0: np.ndarray) -> np.ndarray:
        """Return the screw axis (twist coords) for a revolute joint with frame T0."""
        if self.dim == 2:
            omega = np.array([0, 0, 1])
            q = np.hstack((T0[:2, 2], 0))
            return np.hstack((np.cross(-omega, q), omega))[[5, 0, 1]]
        else:
            omega = T0[:3, 2]
            q = T0[:3, 3]
            return np.hstack((omega, np.cross(-omega, q)))

    def set_geometric_attributes(self):
        for ee in self.end_effectors:
            k_map = self.kinematic_map[ROOT][ee]
            for idx in range(len(k_map)):
                cur = k_map[idx]
                self.nodes[cur]["S"] = self._screw_axis_from_T0(self.nodes[cur]["T0"])
                if idx != 0:
                    pred = k_map[idx - 1]
                    self[pred][cur][TRANSFORM] = (
                        self._SE.inverse(self.nodes[pred]["T0"])
                        @ self.nodes[cur]["T0"]
                    )

    def from_params(self) -> Dict[str, np.ndarray]:
        """Construct T_zero from link lengths. 2D-only today (no fk_tree_3d helper)."""
        assert self.dim == 2, "from_params (link_lengths) is 2D-only"
        self.l = self.params["link_lengths"]
        T = {ROOT: np.eye(3)}
        q0 = self.zero_configuration()
        kmap = self.kinematic_map
        for ee in self.end_effectors:
            for node in kmap[ROOT][ee][1:]:
                path_nodes = kmap[ROOT][node][1:]
                T[node] = fk_tree_2d(self.l, q0, q0, path_nodes)
        return T

    def from_dh_params(self, params: Dict) -> Dict[str, np.ndarray]:
        """Construct T_zero from DH params. 3D-only."""
        assert self.dim == 3, "DH parameterization is 3D-only"
        a, d, al, th, modified_dh = (
            params["a"], params["d"], params["alpha"], params["theta"], params["modified_dh"],
        )
        a = a if type(a) is dict else list_to_variable_dict(flatten([a]))
        d = d if type(d) is dict else list_to_variable_dict(flatten([d]))
        al = al if type(al) is dict else list_to_variable_dict(flatten([al]))
        th = th if type(th) is dict else list_to_variable_dict(flatten([th]))

        T = {ROOT: np.eye(4)}
        kmap = self.kinematic_map
        for ee in self.end_effectors:
            for node in kmap[ROOT][ee][1:]:
                path_nodes = kmap[ROOT][node][1:]
                q = np.asarray([0 for _ in path_nodes])
                a_ = np.asarray([a[n] for n in path_nodes])
                alpha_ = np.asarray([al[n] for n in path_nodes])
                th_ = np.asarray([th[n] for n in path_nodes])
                d_ = np.asarray([d[n] for n in path_nodes])
                if not modified_dh:
                    T[node] = fk_3d(a_, alpha_, d_, q + th_)
                else:
                    T[node] = modified_fk_3d(a_, alpha_, d_, q + th_)
        return T

    def pose(self, joint_angles: Dict[str, float], query_node: str) -> np.ndarray:
        """Forward kinematics: returns (dim+1)x(dim+1) homogeneous transform."""
        kmap = self.kinematic_map[ROOT][query_node]
        T = self.nodes[ROOT]["T0"]
        for idx in range(len(kmap) - 1):
            pred, cur = kmap[idx], kmap[idx + 1]
            T = T @ self._SE.Exp(self.nodes[pred]["S"] * joint_angles[cur])
        T = T @ self.nodes[query_node]["T0"]
        return T

    def jacobian(
        self,
        joint_angles: Dict[str, float],
        query_nodes: Union[List[str], str],
    ) -> Dict[str, ArrayLike]:
        """Body-frame twist Jacobian. Twist dim is 3 for SE(2), 6 for SE(3)."""
        kmap = self.kinematic_map[ROOT]
        if query_nodes is None:
            query_nodes = self.end_effectors

        twist_dim = 3 if self.dim == 2 else 6
        J = {}
        for node in query_nodes:
            path = kmap[node]
            T = self.nodes[ROOT]["T0"]
            J[node] = np.zeros([twist_dim, self.n])
            for idx in range(len(path) - 1):
                pred, cur = path[idx], path[idx + 1]
                if idx == 0:
                    J[node][:, idx] = self.nodes[pred]["S"]
                else:
                    ppred = list(self.predecessors(pred))[0]
                    T = T @ self._SE.Exp(self.nodes[ppred]["S"] * joint_angles[pred])
                    Ad = self._SE.adjoint(T)
                    J[node][:, idx] = Ad @ self.nodes[pred]["S"]
        return J

    def jacobian_geometric(
        self,
        joint_angles: Dict[str, float],
        nodes: Union[List[str], str],
        Ts: Dict[str, np.ndarray] = None,
    ) -> Dict[str, ArrayLike]:
        """Geometric Jacobian (linear+angular) for end-effector p-nodes. 3D-only."""
        assert self.dim == 3, "jacobian_geometric is 3D-only"
        kmap = self.kinematic_map[ROOT]

        if nodes is None:
            nodes = []
            for ee in self.end_effectors:
                if ee[0][0] == MAIN_PREFIX:
                    nodes += [ee[0]]
                elif ee[1][0] == MAIN_PREFIX:
                    nodes += [ee[1]]

        if Ts is None:
            Ts = self.get_all_poses(joint_angles)

        J = {}
        for node in nodes:
            path = kmap[node][1:]
            p_ee = Ts[node][:3, 3]
            J[node] = np.zeros([6, self.n])
            for idx, joint in enumerate(path):
                T_0_i = Ts[list(self.parents.predecessors(joint))[0]]
                z_hat_i = T_0_i[:3, 2]
                p_i = T_0_i[:3, 3]
                J[node][:3, idx] = np.cross(z_hat_i, p_ee - p_i)
                J[node][3:, idx] = z_hat_i
        return J
