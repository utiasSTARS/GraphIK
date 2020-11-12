import numpy as np
import networkx as nx
import pickle

from graphik.robots.revolute import Revolute2dTree
from graphik.graphs.graph_base import SphericalRobotGraph
from graphik.utils.utils import list_to_variable_dict, make_save_string
from graphik.utils.experiments import run_multiple_experiments, process_experiment


if __name__ == "__main__":
    # Experiment params
    dim = 2
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
    local_algorithms_unbounded = [
        "trust-exact"
    ]
    local_algorithms_bounded = [
        "trust-constr"
    ]
    n_goals = 1000  # Number of goals
    n_init = 1  # Number of initializations to try (should be 1 for zero_init = True and for bound_smoothing = True)
    zero_init = True  # True makes the angular solvers MUCH better w
    use_limits = False  # Whether to use angular limits for all the solvers
    do_jacobian = False  # Jacobian doesn't work well for zero_init (need a more local starting point)
    fabrik_only = False  # Only run the FABRIK solver (messy utility for re-running after the bug)
    symbolic = False
    if fabrik_only:
        do_jacobian = False
    if fabrik_only:
        local_algorithms = []
    else:
        local_algorithms = local_algorithms_bounded if use_limits else local_algorithms_unbounded
    # Solver params
    verbosity = 2  # Needs to be 2 for Riemannian solver at the moment TODO: make it smarter!!
    maxiter = 2000  # Most algs never max it (Riemannian ConjugateGradient often does)
    tol = 1e-9  # This is the key parameter, will be worth playing with (used for gtol except for SLSQP)
    initial_tr_radius = 1.  # This is a key parameter for trust-constr and trust-exact.
    trigsimp = False  # Not worth setting to True for n_init = 1
    if fabrik_only:
        riemannian_algorithms = []
    else:
        # riemannian_algorithms = ["TrustRegions", "ConjugateGradient"]
        riemannian_algorithms = ["TrustRegions"]
    solver_params = {"solver": "BFGS", "maxiter": maxiter, "tol": tol, "initial_tr_radius": initial_tr_radius}
    bound_smoothing = True  # Riemannian algs will do with and without bound smoothing when this is True
    riemannian_alg1 = riemannian_algorithms[0] if not fabrik_only else "TrustRegions"
    riemann_params = {
        "solver": riemannian_alg1,
        "logverbosity": verbosity,
        "mingradnorm": tol,
        "maxiter": maxiter,
    }
    jacobian_params = {
        "tol": tol,
        "maxiter": maxiter,
        "dt": 1e-3,
        "method": "dls_inverse"
    }
    fabrik_tol = 1e-9
    fabrik_max_iter = maxiter  # FABRIK is faster per iteration, might be worth changing this around

    # Form the robot and graph object (any robot you want here)
    height = 4
    gen = nx.balanced_tree(2, height, create_using=nx.DiGraph)
    gen = nx.relabel_nodes(gen, {node: f"p{node}" for node in gen})
    n = gen.number_of_edges()
    dof = n
    print("Number of DOF: {:}".format(n))
    parents = nx.to_dict_of_lists(gen)
    a = list_to_variable_dict(np.ones(n))
    th = list_to_variable_dict(np.zeros(n))
    if use_limits:
        lim = np.minimum(np.random.rand(n)*np.pi + 0.2, np.pi)
    else:
        lim = np.pi * np.ones(n)
    lim_u = list_to_variable_dict(lim)
    lim_l = list_to_variable_dict(-lim)
    params = {
        "a": a,
        "theta": th,
        "parents": parents,
        "joint_limits_upper": lim_u,
        "joint_limits_lower": lim_l,
    }

    # Save string setup
    save_string_properties = [
        ("dof", dof),
        ("bounded", use_limits),
        ("tol", tol),
        ("maxiter", maxiter),
        ("n_goals", n_goals),
        ("n_init", n_init),
        ("zero_init", zero_init)
    ]

    if fabrik_only:
        save_string = "results/FABRIK_only_planar_tree_" + make_save_string(save_string_properties)
    else:
        save_string = "results/planar_tree_" + make_save_string(save_string_properties)
    # Robot params
    robot = Revolute2dTree(params)
    # robot.lambdify_get_pose()
    graph = SphericalRobotGraph(robot)

    results = run_multiple_experiments(graph, n_goals, n_init, zero_init, solver_params,
                             riemann_params, jacobian_params, use_limits, verbosity,
                             bound_smoothing, local_algorithms, riemannian_algorithms,
                             fabrik_max_iter, use_symbolic=symbolic, trigsimp=trigsimp, do_jacobian=do_jacobian,
                                       pose_goals=True)
    results.robot = robot
    results.seed = seed
    pickle.dump(results, open(save_string + "full_results.p", "wb"))
    process_experiment(results)
