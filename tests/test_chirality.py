"""Sanity tests for the chirality penalty.

Covers:

* ``signed_volume`` sign flip under reflection,
* finite-difference check on ``chirality_cost_grad``,
* detection: Panda yields chiral link tetrahedra at zero config; UR10 does not.
"""
from __future__ import annotations

import unittest

import numpy as np

from graphik.solvers.chirality_cost import chirality_cost_grad
from graphik.solvers.riemannian_solver import RiemannianSolver
from graphik.utils.chirality import (
    chirality_reference,
    link_tetrahedra,
    signed_volume,
)
from graphik.utils.dgp import (
    adjacency_matrix_from_graph,
    bound_smoothing,
    distance_matrix_from_graph,
)
from graphik.utils.roboturdf import load_panda, load_ur10


class SignedVolumeTests(unittest.TestCase):
    def test_reflection_flips_sign(self):
        rng = np.random.default_rng(0)
        P = rng.standard_normal((4, 3))
        v = signed_volume(P)
        Pm = P * np.array([1.0, 1.0, -1.0])
        self.assertAlmostEqual(float(signed_volume(Pm)), -float(v), places=12)

    def test_known_unit_tetrahedron(self):
        P = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
        self.assertAlmostEqual(float(signed_volume(P)), 1.0 / 6.0, places=12)


class ChirGradFiniteDiffTests(unittest.TestCase):
    def test_gradient_matches_finite_diff(self):
        rng = np.random.default_rng(1)
        N, K = 12, 3
        Y = rng.standard_normal((N, 3))
        tuples = np.stack(
            [rng.choice(N, 4, replace=False) for _ in range(K)]
        ).astype(np.int64)
        signs = rng.choice([-1.0, 1.0], size=K)
        # pick eps small enough that some tuples are active and some aren't
        Vref = signed_volume(Y[tuples])
        eps = 0.5 * np.abs(Vref) + 1e-3
        weight = 0.7

        c0, g = chirality_cost_grad(Y, tuples, signs, eps, weight)

        eps_fd = 1e-6
        g_fd = np.zeros_like(Y)
        for i in range(N):
            for j in range(3):
                Yp = Y.copy(); Yp[i, j] += eps_fd
                Ym = Y.copy(); Ym[i, j] -= eps_fd
                cp, _ = chirality_cost_grad(Yp, tuples, signs, eps, weight)
                cm, _ = chirality_cost_grad(Ym, tuples, signs, eps, weight)
                g_fd[i, j] = (cp - cm) / (2 * eps_fd)
        self.assertTrue(
            np.allclose(g, g_fd, atol=1e-6),
            f"max |g - g_fd| = {np.abs(g - g_fd).max():.3e}",
        )


class DetectionTests(unittest.TestCase):
    def test_panda_has_chiral_link_tetrahedra(self):
        _, graph = load_panda()
        chir = chirality_reference(graph, threshold=1e-6)
        self.assertIsNotNone(chir, "expected at least one chiral tuple on Panda")
        # We empirically observed three: (p2,q2,p3,q3), (p3,q3,p4,q4),
        # (p5,q5,p6,q6), plus the base tetrahedron {p0, x, y, q0}.
        names = chir["names"]
        self.assertIn(("p0", "x", "y", "q0"), names)
        link_quads = [n for n in names if "x" not in n and "y" not in n]
        self.assertGreaterEqual(len(link_quads), 3)
        self.assertEqual(chir["tuples"].shape[1], 4)
        self.assertEqual(chir["signs"].shape[0], chir["tuples"].shape[0])
        self.assertTrue(np.all(np.isin(chir["signs"], [-1.0, 1.0])))

    def test_ur10_has_only_base_tetrahedron(self):
        _, graph = load_ur10()
        chir = chirality_reference(graph, threshold=1e-6)
        # All UR10 link tetrahedra are coplanar — only the base tetrahedron
        # survives the threshold.
        self.assertIsNotNone(chir)
        link_quads = [
            n for n in chir["names"] if "x" not in n and "y" not in n
        ]
        self.assertEqual(link_quads, [])

    def test_link_tetrahedra_listing_is_unique(self):
        _, graph = load_panda()
        tets = link_tetrahedra(graph)
        self.assertEqual(len(tets), len(set(tets)))


class PandaIntegrationTests(unittest.TestCase):
    """End-to-end check: with the chirality penalty enabled, the recovered
    Panda configuration matches the reference signs on all chiral tuples.
    Without the penalty, one or more tuples typically flip — but that's not
    guaranteed, so we only assert the positive direction.
    """

    def test_chirality_penalty_preserves_reference_signs(self):
        np.random.seed(7)
        robot, graph = load_panda()
        chir = chirality_reference(graph, threshold=1e-6)
        self.assertIsNotNone(chir)

        q_true = robot.random_configuration()
        T_goal = robot.pose(q_true, robot.end_effectors[0])
        G = graph.from_pose(T_goal)
        D_goal = distance_matrix_from_graph(G)
        omega = adjacency_matrix_from_graph(G)
        lb, ub = bound_smoothing(G)

        solver = RiemannianSolver(graph, chirality=chir)
        res = solver.solve(D_goal, omega, use_limits=True, bounds=(lb, ub))
        Y = res["x"]

        # Distance loss should still converge.
        self.assertLess(res["f(x)"], 1e-6, msg=f"f(x)={res['f(x)']}")

        sv = signed_volume(Y[chir["tuples"]])
        flips = int(np.sum(np.sign(sv) != chir["signs"]))
        self.assertEqual(flips, 0, msg=f"signed volumes: {sv}")


if __name__ == "__main__":
    unittest.main()
