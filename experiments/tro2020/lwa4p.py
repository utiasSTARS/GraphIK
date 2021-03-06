import graphik
from graphik.utils.roboturdf import RobotURDF
import numpy as np
from numpy import pi
import pickle

from graphik.graphs.graph_base import RobotRevoluteGraph
from graphik.utils.utils import make_save_string
from graphik.utils.experiments import run_multiple_experiments, process_experiment


if __name__ == "__main__":
    # Experiment params
    dim = 3
    seed = 8675309
    np.random.seed(seed)

    # Keep whichever algorithms you want to run ('trust-exact', 'Newton-CG', and 'trust-constr' are the best)
    # local_algorithms_unbounded = [
    #     "BFGS",
    #     "CG",
    #     "Newton-CG",
    #     "trust-exact"
    # ]
    # local_algorithms_bounded = [
    #     "L-BFGS-B",
    #     "TNC",
    #     "SLSQP",
    #     "trust-constr"
    # ]
    local_algorithms_unbounded = ["trust-exact"]
    local_algorithms_bounded = ["trust-constr"]
    n_goals = 20  # Number of goals
    n_init = 1  # Number of initializations to try (should be 1 for zero_init = True and for bound_smoothing = True)
    zero_init = True  # True makes the angular solvers MUCH better w
    use_limits = True  # Whether to use angular limits for all the solvers
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
    ]

    # ### UR10 DH
    n = 6
    ub = np.minimum(np.random.rand(n) * (pi / 2) + pi / 2, pi)
    lb = -ub
    fname = graphik.__path__[0] + "/robots/urdfs/lwa4p.urdf"
    urdf_robot = RobotURDF(fname)

    robot = urdf_robot.make_Revolute3d(ub, lb)  # make the Revolute class from a URDF
    graph = RobotRevoluteGraph(robot)

    save_string = "results/lwa4p_" + make_save_string(save_string_properties)

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
    )
    # results.robot = robot
    # results.seed = seed
    # pickle.dump(results, open(save_string + "full_results.p", "wb"))
    # process_experiment(results)
