import cvxik
from cvxik.utils.roboturdf import RobotURDF
import numpy as np
from numpy import pi
import networkx as nx
import pickle

from cvxik.robots.revolute import Revolute3dChain
from cvxik.graphs.graph_base import SphericalRobotGraph, Revolute3dRobotGraph
from cvxik.utils.utils import list_to_variable_dict, make_save_string
from cvxik.utils.experiments import run_multiple_experiments, process_experiment


if __name__ == "__main__":
    # Experiment params
    dim = 3
    seed = 11235813 #8675309
    np.random.seed(seed)
    local_algorithms_unbounded = ["trust-exact"]
    local_algorithms_bounded = ["trust-constr"]
    n_goals = 3000  # Number of goals
    n_init = 1  # Number of initializations to try (should be 1 for zero_init = True and for bound_smoothing = True)
    zero_init = True  # True makes the angular solvers MUCH better w
    sdp_random_init = True  # Whether to use a random initialization for the SDP solver (vs. zero_init like the others)
    use_limits = False  # Whether to use angular limits for all the solvers
    do_sdp = True
    do_jacobian = False  # Jacobian doesn't work well for zero_init (need a more local starting point)
    do_fabrik = False  # Fabrik not supported for generic revolute solvers
    fabrik_only = (
        False  # Only run the FABRIK solver (messy utility for re-running after the bug)
    )
    if fabrik_only:
        do_jacobian = False
    if fabrik_only:
        local_algorithms = []
    else:
        local_algorithms = (
            local_algorithms_bounded if use_limits else local_algorithms_unbounded
        )

    # Solver params
    verbosity = (
        0  # Needs to be 2 for Riemannian solver at the moment TODO: make it smarter!!
    )
    maxiter = 2000  # Most algs never max it (Riemannian ConjugateGradient often does)
    riemannian_maxiter = 2000
    tol = 1e-9  # This is the key parameter, will be worth playing with (used for gtol except for SLSQP)
    initial_tr_radius = 1.0  # This is a key parameter for trust-constr and trust-exact.
    trigsimp = False  # Not worth setting to True for n_init = 1
    if fabrik_only:
        riemannian_algorithms = []
    else:
        # riemannian_algorithms = ["TrustRegions", "ConjugateGradient"]
        riemannian_algorithms = ["TrustRegions"]
    solver_params = {
        "solver": "BFGS",
        "maxiter": maxiter,
        "tol": tol,
        "initial_tr_radius": initial_tr_radius,
    }
    bound_smoothing = True  # Riemannian algs will do with and without bound smoothing when this is True
    riemannian_alg1 = riemannian_algorithms[0] if not fabrik_only else "TrustRegions"
    riemann_params = {
        "solver": riemannian_alg1,
        "logverbosity": verbosity,
        "mingradnorm": tol,
        "maxiter": riemannian_maxiter,
    }
    jacobian_params = {
        "tol": tol,
        "maxiter": maxiter,
        "dt": 1e-3,
        "method": "dls_inverse",
    }
    fabrik_tol = 1e-9
    fabrik_max_iter = (
        maxiter  # FABRIK is faster per iteration, might be worth changing this around
    )

    # Save string setup
    save_string_properties = [
        ("bounded", use_limits),
        ("tol", tol),
        ("maxiter", maxiter),
        ("n_goals", n_goals),
        ("n_init", n_init),
        ("zero_init", zero_init),
        ("sdp_rand_init", sdp_random_init)
    ]

    # ### UR10
    n = 6
    ub = np.minimum(np.random.rand(n)*(pi/2) + pi/2, pi)
    lb = -ub
    fname = cvxik.__path__[0] + "/robots/urdfs/ur10_mod.urdf"
    urdf_robot = RobotURDF(fname)

    robot = urdf_robot.make_Revolute3d(ub, lb)  # make the Revolute class from a URDF
    print(robot.structure.nodes())
    graph = Revolute3dRobotGraph(robot)

    save_string = "results/ur10_" + make_save_string(save_string_properties)

    print("Running experiments")
    results = run_multiple_experiments(
        graph,
        n_goals,
        n_init,
        zero_init,
        solver_params,
        riemann_params,
        jacobian_params,
        use_limits,
        verbosity,
        bound_smoothing,
        local_algorithms,
        riemannian_algorithms,
        fabrik_max_iter,
        trigsimp=trigsimp,
        do_jacobian=do_jacobian,
        do_fabrik=do_fabrik,
        pose_goals=True,
        do_sdp=do_sdp,
        sdp_random_init=sdp_random_init
    )
    results.robot = robot
    results.seed = seed
    pickle.dump(results, open(save_string + "full_results.p", "wb"))
    process_experiment(results)
