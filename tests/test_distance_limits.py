"""Distance-limit invariants on the constructed problem graph.

For any random configuration with joint angles in [lb, ub], the realized
graph (positions computed from forward kinematics) must satisfy:

  1. Every edge with a DIST attribute is *rigid* — the realized distance
     between its endpoints equals DIST exactly (within float tolerance).
     This covers base K_{dim+1}, intra-joint (p_i, q_i) for 3D, and the
     inter-joint complete-bipartite edges between consecutive joints.

  2. Every edge whose BOUNDED list contains BELOW or ABOVE — i.e., a
     level-2 set_limits or root_angle_limits edge whose distance varies
     with joint rotation — has realized distance within [LOWER, UPPER].
     graph.check_distance_limits returns the violations.

These invariants exercise the graph-construction math (link-length
computation at zero config, max/min-distance bounds from joint rotation)
end-to-end against the runtime kinematics. A construction bug that
mis-computes a fixed distance or a level-2 bound surfaces here as a
violation; a forward-kinematics bug surfaces as both BOUNDED and DIST
violations across many edges.
"""
import unittest
from math import pi

import numpy as np
import graphik
from graphik.graphs import ProblemGraph
from graphik.robots import Robot
from graphik.utils import list_to_variable_dict
from graphik.utils.constants import DIST
from graphik.utils.roboturdf import RobotURDF


N_TRIALS = 50
SEED = 42
TOL = 1e-6


class TestDistanceLimits(unittest.TestCase):

    def _check_invariants(self, graph, robot, n_trials=N_TRIALS, seed=SEED, tol=TOL):
        """Run n_trials random in-limit configurations and assert (1)+(2)."""
        np.random.seed(seed)
        for trial in range(n_trials):
            q = robot.random_configuration()
            G = graph.realization(q)

            # (2) BOUNDED edges respect their LOWER/UPPER
            broken = graph.check_distance_limits(G, tol=tol)
            self.assertEqual(
                broken,
                [],
                msg=(
                    f"trial {trial}: in-limit configuration violated "
                    f"{len(broken)} bound(s); first 3: {broken[:3]}"
                ),
            )

            # (1) Exact-DIST edges match the realization
            for u, v, data in graph.edges(data=True):
                if DIST not in data:
                    continue
                expected = data[DIST]
                actual = G[u][v][DIST]
                if not np.isclose(actual, expected, atol=tol, rtol=tol):
                    self.fail(
                        f"trial {trial}: edge ({u},{v}) DIST mismatch — "
                        f"expected {expected:.8f}, got {actual:.8f}"
                    )

    def test_revolute_ur10(self):
        fname = graphik.__path__[0] + "/robots/urdfs/ur10_mod.urdf"
        urdf_robot = RobotURDF(fname)
        n = urdf_robot.n_q_joints
        ub = pi * np.ones(n)
        lb = -ub
        robot = urdf_robot.make_Revolute3d(ub, lb)
        graph = ProblemGraph(robot)
        self._check_invariants(graph, robot)

    def test_revolute_panda(self):
        # Different URDF, different DH and link geometry — guards against
        # UR10-specific coincidences.
        fname = graphik.__path__[0] + "/robots/urdfs/panda_arm.urdf"
        urdf_robot = RobotURDF(fname)
        n = urdf_robot.n_q_joints
        ub = pi * np.ones(n)
        lb = -ub
        robot = urdf_robot.make_Revolute3d(ub, lb)
        graph = ProblemGraph(robot)
        self._check_invariants(graph, robot)

    def test_planar_chain(self):
        # 2D chain with non-trivial joint limits (0.8*pi) so set_limits
        # produces non-degenerate bounds.
        n = 6
        a = list_to_variable_dict(np.ones(n))
        th = list_to_variable_dict(np.zeros(n))
        ub = list_to_variable_dict(0.8 * pi * np.ones(n))
        lb = list_to_variable_dict(-0.8 * pi * np.ones(n))
        params = {
            "link_lengths": a,
            "theta": th,
            "joint_limits_upper": ub,
            "joint_limits_lower": lb,
            "num_joints": n,
        }
        robot = Robot({**params, "dim": 2})
        graph = ProblemGraph(robot)
        self._check_invariants(graph, robot)


if __name__ == "__main__":
    unittest.main()
