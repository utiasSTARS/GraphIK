"""Chirality detection for distance-geometric IK.

Distance-only constraints determine point positions up to global rigid motion
*and reflection*. For each set of four mutually-connected graph nodes whose
Cayley-Menger determinant is non-zero (i.e. they form a genuine tetrahedron),
the distance constraints admit two mirror solutions. We pick the correct one
by computing the signed volume from a forward-kinematics reference and
penalizing the opposite sign during optimization.

For typical revolute graphs in GraphIK there are two structural sources of
chiral tetrahedra:

* The base tetrahedron ``{p0, x, y, q0}`` set by ``base_subgraph()``. Its
  signed volume is constant by construction and pins the ambient handedness.
* Per-link tetrahedra ``{p<i-1>, q<i-1>, p<i>, q<i>}`` formed by two points
  on each of two consecutive joint axes. Volume is zero when the axes are
  coplanar (intersecting or parallel) — the case for UR10/KUKA — and non-zero
  for offset axes — the case for Panda's joints 3, 4, and 6.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from graphik.utils.constants import AUX_PREFIX, POS, ROOT


def signed_volume(P: NDArray) -> NDArray:
    """Signed volume of a tetrahedron with corners along the last two axes.

    ``P`` has shape ``(..., 4, 3)``. Returns shape ``(...)``. Sign flips under
    reflection of the ambient frame.
    """
    P = np.asarray(P)
    a, b, c, d = P[..., 0, :], P[..., 1, :], P[..., 2, :], P[..., 3, :]
    return np.einsum("...i,...i->...", np.cross(b - a, c - a), d - a) / 6.0


def link_tetrahedra(graph) -> List[Tuple[str, str, str, str]]:
    """Return the canonical 4-tuples of node names whose chirality matters.

    Always includes the base tetrahedron when present, plus one tuple per
    consecutive joint pair along every kinematic chain. Duplicates across
    branches of a tree are removed.
    """
    ids = set(graph.node_ids)
    tuples: List[Tuple[str, str, str, str]] = []

    base_tet = (ROOT, "x", "y", AUX_PREFIX + ROOT[1:])
    if all(n in ids for n in base_tet):
        tuples.append(base_tet)

    seen = set()
    for ee in graph.robot.end_effectors:
        kmap = graph.robot.kinematic_map[ROOT][ee]
        for idx in range(1, len(kmap)):
            pred, cur = kmap[idx - 1], kmap[idx]
            key = (pred, cur)
            if key in seen:
                continue
            seen.add(key)
            tet = (
                pred, AUX_PREFIX + pred[1:],
                cur, AUX_PREFIX + cur[1:],
            )
            if all(n in ids for n in tet):
                tuples.append(tet)
    return tuples


def chirality_reference(
    graph,
    joint_angles: Optional[Dict[str, float]] = None,
    threshold: float = 1e-6,
    eps_scale: float = 0.5,
    weight: float = 1.0,
) -> Optional[Dict[str, NDArray]]:
    """Build the chirality side-information for a problem graph.

    Realizes the graph at ``joint_angles`` (zero config by default), computes
    the signed volume of every link tetrahedron, drops those below
    ``threshold`` (effectively coplanar — distance constraints already
    determine them), and returns the row-index tuples, target signs, and
    per-tuple gap ``eps_t = eps_scale * |V_t_ref|``.

    Returns ``None`` when no chiral tuples remain, signalling that the
    distance formulation alone suffices for this robot.
    """
    if joint_angles is None:
        joint_angles = {n: 0.0 for n in graph.robot.random_configuration().keys()}

    G = graph.realization(joint_angles)
    ids = graph.node_ids
    name_to_idx = {n: i for i, n in enumerate(ids)}
    P = np.stack([G.nodes[n][POS] for n in ids])

    rows: List[List[int]] = []
    signs: List[float] = []
    refs: List[float] = []
    names: List[Tuple[str, str, str, str]] = []

    for tet in link_tetrahedra(graph):
        idx = [name_to_idx[n] for n in tet]
        v = float(signed_volume(P[idx]))
        if abs(v) < threshold:
            continue
        rows.append(idx)
        signs.append(1.0 if v > 0 else -1.0)
        refs.append(v)
        names.append(tet)

    if not rows:
        return None

    refs_arr = np.asarray(refs, dtype=float)
    return {
        "tuples": np.asarray(rows, dtype=np.int64),
        "signs": np.asarray(signs, dtype=float),
        "ref_volumes": refs_arr,
        "eps": np.abs(refs_arr) * eps_scale,
        "weight": float(weight),
        "names": names,
    }
