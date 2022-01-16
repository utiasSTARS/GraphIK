import cvxpy as cp
import mosek


class SdpSolverParams:
    """
    Parameters for cvxpy's SDP solvers (MOSEK, CVXOPT, and SCS).
    """
    def __init__(self, solver=cp.MOSEK, abstol=1e-6, reltol=1e-6, feastol=1e-6, max_iters=1000, refinement_steps=1,
                 kkt_solver='chol', alpha=1.8, scale=5.0, normalize=True, use_indirect=True, qcp=False,
                 mosek_params=None, feasibility=False, cost_function=None, verbose=False):
        self.solver = solver  # cp.CVXOPT #cp.MOSEK #cp.SCS
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
        if mosek_params is None:
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
