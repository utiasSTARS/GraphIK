import cvxpy as cp

try:
    import mosek
    MOSEK_AVAILABLE = True
except ImportError:
    mosek = None
    MOSEK_AVAILABLE = False


def _default_sdp_solver():
    if MOSEK_AVAILABLE:
        return cp.MOSEK
    installed = cp.installed_solvers()
    if cp.CLARABEL in installed:
        return cp.CLARABEL
    if cp.SCS in installed:
        return cp.SCS
    return cp.CVXOPT


DEFAULT_SDP_SOLVER = _default_sdp_solver()


class SdpSolverParams:
    """
    Parameters for cvxpy's SDP solvers (MOSEK, CVXOPT, SCS, CLARABEL).
    """
    def __init__(self, solver=None, abstol=1e-6, reltol=1e-6, feastol=1e-6, max_iters=1000, refinement_steps=1,
                 kkt_solver='chol', alpha=1.8, scale=5.0, normalize=True, use_indirect=True, qcp=False,
                 mosek_params=None, feasibility=False, cost_function=None, verbose=False):
        self.solver = DEFAULT_SDP_SOLVER if solver is None else solver
        # Common
        self.abstol = abstol
        self.reltol = reltol
        self.feastol = feastol
        self.max_iters = max_iters
        self.qcp = qcp
        self.feasibility = feasibility  # Whether to perform a feasibility program (i.e., zero cost)
        self.verbose = verbose
        # CVXOPT
        self.refinement_steps = refinement_steps
        self.kkt_solver = kkt_solver  # 'chol' or 'robust'
        # SCS
        self.alpha = alpha  # Relaxation parameter for SCS
        self.scale = scale
        self.normalize = normalize  # Whether to preconditon data matrix
        self.use_indirect = use_indirect
        self.cost_function = cost_function
        # MOSEK
        if mosek_params is None and MOSEK_AVAILABLE:
            self.mosek_params = {'MSK_IPAR_INTPNT_MAX_ITERATIONS': max_iters,
                                 'MSK_DPAR_INTPNT_TOL_PFEAS': abstol,
                                 'MSK_DPAR_INTPNT_TOL_DFEAS': abstol,
                                 'MSK_DPAR_INTPNT_TOL_REL_GAP': reltol,
                                 'MSK_DPAR_INTPNT_TOL_INFEAS': feastol,
                                 'MSK_IPAR_INFEAS_REPORT_AUTO': True,
                                 'MSK_IPAR_INFEAS_REPORT_LEVEL': 10,
                                 mosek.iparam.intpnt_scaling: mosek.scalingtype.free,
                                 mosek.iparam.intpnt_solve_form: mosek.solveform.primal,
                                 mosek.iparam.ana_sol_print_violated: mosek.onoffkey.on,
                                 mosek.dparam.intpnt_co_tol_near_rel: 1e5
                                 }
        else:
            self.mosek_params = mosek_params


def solve_sdp(prob, solver_params, **extra_kwargs):
    """Solve an SDP problem using the configured solver, passing mosek-specific
    kwargs only when MOSEK is selected. Caller-provided kwargs (e.g. verbose,
    warm_start) are forwarded as-is."""
    solver = solver_params.solver
    kwargs = dict(extra_kwargs)
    if solver == cp.MOSEK and solver_params.mosek_params is not None:
        kwargs.setdefault('mosek_params', solver_params.mosek_params)
    prob.solve(solver=solver, **kwargs)
    return prob
