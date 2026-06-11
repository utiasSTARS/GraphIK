"""Removed-kwarg guards on the solver constructors.

Both solver __init__s setattr unknown kwargs onto self, so without an
explicit guard a stale ``jit=True`` from a pre-removal caller would be
silently swallowed instead of erroring.
"""
import unittest

from graphik.solvers.nonlinear_solver import NonlinearSolver
from graphik.solvers.riemannian_solver import RiemannianSolver


class TestRemovedKwargs(unittest.TestCase):
    def test_riemannian_rejects_jit(self):
        with self.assertRaisesRegex(TypeError, "jit"):
            RiemannianSolver(None, jit=True)

    def test_riemannian_rejects_cost_type(self):
        with self.assertRaisesRegex(TypeError, "cost_type"):
            RiemannianSolver(None, cost_type="dense")

    def test_nonlinear_rejects_jit(self):
        with self.assertRaisesRegex(TypeError, "jit"):
            NonlinearSolver(None, jit=True)

    def test_nonlinear_rejects_cost_type(self):
        with self.assertRaisesRegex(TypeError, "cost_type"):
            NonlinearSolver(None, cost_type="dense")


if __name__ == "__main__":
    unittest.main()
