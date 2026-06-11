"""Stale kwargs from removed APIs must raise TypeError."""
import unittest

from graphik.solvers import JointAngleSolver, RiemannianSolver, ScipySolver


class TestRemovedKwargs(unittest.TestCase):
    def test_riemannian_rejects_stale_kwargs(self):
        with self.assertRaisesRegex(TypeError, "jit"):
            RiemannianSolver(None, jit=True)
        with self.assertRaisesRegex(TypeError, "cost_type"):
            RiemannianSolver(None, cost_type="dense")

    def test_scipy_rejects_stale_kwargs(self):
        with self.assertRaisesRegex(TypeError, "jit"):
            ScipySolver(None, jit=True)
        with self.assertRaisesRegex(TypeError, "cost_type"):
            ScipySolver(None, cost_type="dense")

    def test_joint_angle_rejects_old_params_positional(self):
        with self.assertRaises(TypeError):
            JointAngleSolver(None, {})


if __name__ == "__main__":
    unittest.main()
