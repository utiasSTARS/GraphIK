import sympy as sp
import cvxpy as cp
import numpy as np
import mosek
import networkx as nx
from matplotlib import pyplot as plt


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


# class SdpFullSolution:
#     """
#     Full solution for SDPs.
#     """
#     def __init__(self):
#         pass
#

class QcqpToSdpRelaxation:

    def __init__(self, constraints: list, extra_var_sets: list = None, verbose: bool = False,
                 force_dense: bool = False):
        """
        Create an object that produces cvxpy SDP instances that are relaxations of QCQPs.
        Main Features:
        - Separation of 'fixed' and 'per-instance' constraints for fast computation
            - Similarly takes in different cost functions for a fixed set of constraints
        - Automatically exploit chordal sparsity present in constraints
        - Maps solution to cvxpy problem instance back to original variables

        # TODO: use cvxpy parameter feature to define cost (faster experimental setup)

        :param constraints: list of quadratic polynomial sympy equality constraints
        :param verbose: turn on for helpful print statements
        :param force_dense: for testing the dense SDP relaxation even when chordal structure is present
        """
        # Store the fixed constraints
        self.constraints = []
        # Extra variables that may appear in extra constraints
        self.extra_var_sets = extra_var_sets
        # Set of variables used in constraints and cost function
        self.variable_set = set()
        # Create a constraint graphs
        self.constraint_graph = nx.Graph()
        # Create mapping of constraints (by index) that each clique's variables covers (allows overlapping enforcement)
        self.clique_to_constraint_index = {}
        # Create a dictionary mapping variables to an integer ID
        self.variable_mapping = {}
        # Each clique needs an SDP variable (or a single one for non-chordal case)
        self.sdp_variables = []
        # This is a list of list of tuples: each list corresponds to one clique, each tuple is one constraint
        self.sdp_constraints = []
        # Make equality constraints for equivalent variables that appear in multiple cliques (chordal only)
        self.equality_constraints = []
        self.verbose = verbose
        self.force_dense = force_dense
        # Store the cvxpy-ready fixed constraints here
        self.fixed_constraints_cvxpy = []

        # Determine which constraints are inequalities (of the form f(x) <= g(x))
        self.are_constraints_inequalities = []
        for idx, cons in enumerate(constraints):
            is_inequality = type(cons) == sp.relational.LessThan
            self.are_constraints_inequalities.append(is_inequality)
            # Turn the inequality into an equality so that it can be parsed uniformly
            # if is_inequality:
            #     self.constraints[idx] = sp.Eq(cons.args[0], cons.args[1])
            self.constraints.append(sp.Eq(cons.args[0], cons.args[1], evaluate = False))

        # Populate the variable set and constraint graphs
        for eq in self.constraints:
            self.variable_set = self.variable_set.union(eq.free_symbols)
            symbol_list = list(eq.free_symbols)
            for idx, symb1 in enumerate(symbol_list):
                for symb2 in symbol_list[idx + 1:]:
                    self.constraint_graph.add_edge(symb1, symb2)

        # Populate the variable set and constraint graphs with variables that WILL be used in extra constraints
        if self.extra_var_sets is not None:
            for var_set in self.extra_var_sets:
                self.variable_set = self.variable_set.union(var_set)
                symbol_list = list(eq.free_symbols)
                for idx, symb1 in enumerate(symbol_list):
                    for symb2 in symbol_list[idx + 1:]:
                        self.constraint_graph.add_edge(symb1, symb2)

        if self.verbose:
            print("Variable set: {:}".format(self.variable_set))
            print("Constraint graphs: {:}".format(self.constraint_graph.edges))
            print("Chordal: {:}".format(nx.is_chordal(self.constraint_graph)))
            nx.draw(self.constraint_graph)
            plt.show()

        # Convert equality constraints into polynomial objects for processing
        self.constraints_polynomials = [QcqpToSdpRelaxation.equality_to_polynomial(cons) for cons in self.constraints]
        self.chordal = nx.is_chordal(self.constraint_graph)
        if self.chordal and not self.force_dense:
            self.max_cliques = nx.chordal_graph_cliques(self.constraint_graph)
            if self.verbose:
                print("Max cliques: {:}".format(self.max_cliques))
            self.create_chordal_sparse_sdp_constraints()
        else:
            if not self.chordal:
                print("INFO: Constraint graphs not chordal!!")
            # self.max_cliques = None
            # For compatibility when searching for inequality constraints
            self.max_cliques = [self.variable_set]
            self.create_sdp_constraints()

        # Create cvxypy compatible constraints with the polynomials in constraints_polynomials
        self.set_fixed_constraints()

    @staticmethod
    def equality_to_polynomial(cons):
        """
        Convert a SymPy equality constraint to a sympy.polys.polytoos.Poly object that is constrained to be zero.
        :param cons: sympy.core.relational.Equality representing some polynomial constraint
        :return: a sympy.polys.polytoos.Poly object representing the same polynomial constraint (when equal to zero)
        """
        # return (cons.args[0] - cons.args[1]).as_poly()
        return cons.rewrite(sp.Add).as_poly()

    @staticmethod
    def extract_rank_1_homogeneous_solution(Z: np.ndarray, tol=1e-6, eig_index=0, include_homog=False):
        """
        Extract a rank-1 solution from an SDP variable Z whose final index is a homogenizing variable (i.e., x**2 = 1).

        :param Z: (n+1,n+1) np.ndarray representing the solution to an
        :param tol: float representing the rank tolerance
        :param eig_index: zero-indexed ordering of eigenvalue (0 is greatest, n-1 is least) whose eigenvector to use as
        the solution.
        :return: (n,) np.ndarray representing the vector whose self-outer product produces Z; None if rank(Z) != 1
        :return: int rank of Z
        """
        u, s, v = np.linalg.svd(Z, full_matrices=False)
        Z = s[eig_index] * np.outer(u.T[eig_index], v[eig_index])
        Z_rank = np.linalg.matrix_rank(Z, tol=tol, hermitian=True)
        # Extract rank 1 solution easily using final column (may need to mult. by -1?)
        if include_homog:
            return Z[0:, -1], Z_rank
        else:
            return Z[0:-1, -1], Z_rank

    def create_sdp_constraints(self):
        self.index_variables()
        sdp_variable, sdp_constraints_clique = self.create_sdp_constraints_from_qcqp_constraints(self.variable_set)
        self.sdp_variables.append(sdp_variable)
        self.sdp_constraints.append(sdp_constraints_clique)

    def create_chordal_sparse_sdp_constraints(self):
        """
        Method for converting a generic set of equality constraints (sympy.Eq) into sparse SDP constraints for use with
        cvxpy.

        Creates constraints in tuples (A, c) such that tr(A*Z) = c for some maximal clique SDP variable Z. That is, to
        use with cvxpy we'd need to write:
                                        cp.trace(A@Z) == c
        for each constraint.
        The linear constraints that equate redundant variables introduced by the chordal decomposition are already in
        equation form.
        :return: None
        """
        # Populate the index of constraints that each clique is involved in
        for clique in self.max_cliques:
            constraints_in_clique = []
            for idx, eq in enumerate(self.constraints):
                if eq.free_symbols.issubset(clique):
                    constraints_in_clique.append(idx)
            self.clique_to_constraint_index[clique] = tuple(constraints_in_clique)
        self.index_variables()
        # Transform the quadratic constraints into linear constraints on SDP variables
        for clique in self.max_cliques:
            sdp_variable, sdp_constraints_clique = self.create_sdp_constraints_from_qcqp_constraints(clique)
            self.sdp_variables.append(sdp_variable)
            self.sdp_constraints.append(sdp_constraints_clique)
        self.create_sparse_equality_constraints()

    def index_variables(self):
        """
        Populate a mapping from each variable name (via sympy) to an integer index.
        :return:
        """
        for idx, var_name in enumerate(self.variable_set):
            self.variable_mapping[var_name] = idx
        if self.verbose:
            print("Variable name to int mapping: {:}".format(self.variable_mapping))

    @staticmethod
    def poly_to_sdp(clique_vars: list, poly_constraint) -> (np.ndarray, float):
        """
        Convert a quadratic equality poly_constraint into an SDP constraint in the variables of clique_vars.
        :param clique_vars: list of variables
        :param poly_constraint: polynomial equality constraint
        :return:
        """
        n = len(clique_vars)
        c = -poly_constraint.coeff_monomial(1)
        A = np.zeros((n + 1, n + 1))
        constraint_vars = poly_constraint.free_symbols
        # Fill in all constraints
        for idx in range(n):
            if clique_vars[idx] in constraint_vars:
                # Add the quadratic monomial term for this variable
                try:
                    A[idx, idx] = poly_constraint.coeff_monomial(clique_vars[idx] ** 2)
                except ValueError:
                    print("idx: {:}".format(idx))
                    print(clique_vars[idx])
                    print("Clique vars: {:}".format(clique_vars))
                    print("Poly constraint: {:}".format(poly_constraint))
                    print("Type: {:}".format(type(clique_vars[idx])))
                # Add the linear term for this variable
                A[idx, -1] = 0.5 * poly_constraint.coeff_monomial(clique_vars[idx])
                A[-1, idx] = A[idx, -1]
                # Add cross-term monomials
                for jdx in range(idx + 1, n):
                    if clique_vars[jdx] in constraint_vars:
                        A[idx, jdx] = 0.5 * poly_constraint.coeff_monomial(clique_vars[idx] * clique_vars[jdx])
                        A[jdx, idx] = A[idx, jdx]
        return A, c

    def create_sdp_constraints_from_qcqp_constraints(self, clique: frozenset) -> (cp.Variable, list):
        """
        This accepts a clique of variables by sympy name (or all variables in the non-chordally-sparse case) and creates
        SDP constraints for all QCQP equality constraints associated with variables within that clique.
        This method relies on self.clique_to_constraint_index having been computed earlier.
        :param clique:
        :return:
        """
        n = len(clique)
        sdp_constraints_clique = []
        # TODO: test that this preserves ordering
        clique_vars = list(clique)
        # Add an extra variable for the homogenizing bit
        sdp_variable = cp.Variable((n + 1, n + 1), PSD=True)
        if self.chordal and not self.force_dense:
            clique_constraints_ind = self.clique_to_constraint_index[clique]
        else:
            clique_constraints_ind = range(len(self.constraints))
        # Add homogenizing variable's constraint for this clique
        A_homog = np.zeros((n + 1, n + 1))
        A_homog[-1, -1] = 1.0
        c_homog = 1.0
        sdp_constraints_clique.append((A_homog, c_homog))
        # Iterate over each
        for ind in clique_constraints_ind:
            poly_constraint = self.constraints_polynomials[ind]
            A, c = QcqpToSdpRelaxation.poly_to_sdp(clique_vars, poly_constraint)
            sdp_constraints_clique.append((A, c))
        return sdp_variable, sdp_constraints_clique

    def create_sparse_equality_constraints(self):
        """
        For a chordally sparse SDP relaxation, the overlapping elements of cliques mean that we have to equate the
        members of different maximal cliques that represent the same variable.
        :return:
        """
        max_clique_list = list(self.max_cliques)
        # Maintain the outer idx, jdx indexing cliques for getting the correct SDP variable
        for idx, clique_idx in enumerate(max_clique_list):
            clique_idx_list = list(clique_idx)
            for jdx in range(idx + 1, len(max_clique_list)):
                clique_jdx_list = list(max_clique_list[jdx])
                # Maintain inner kdx, ldx indexing variables for the correct SDP variable index
                vars_idx = []
                vars_jdx = []
                for kdx, var_k in enumerate(clique_idx_list):
                    for ldx, var_l in enumerate(clique_jdx_list):
                        if var_k == var_l:
                            vars_idx.append(kdx)
                            vars_jdx.append(ldx)
                            # # Make the 'linear' components equal
                            # self.equality_constraints.append(self.sdp_variables[idx][kdx, -1] ==
                            #                                  self.sdp_variables[jdx][ldx, -1])
                            # # Make the 'quadratic' diagonal components equal
                            # self.equality_constraints.append(self.sdp_variables[idx][kdx, kdx] ==
                            #                                  self.sdp_variables[jdx][ldx, ldx])
                            # # TODO: do we need all pairwise (between cliques) mixed quadratic equalities??
                            # if self.verbose:
                            #     print(var_k)
                            #     print(var_l)
                            #     print("Equality constraint: {:}".format(self.equality_constraints[-1]))
                            # # Is symmetry enforcement needed? Might help as redundant constraints tend to?
                            # # self.equality_constraints.append(self.sdp_variables[idx][-1, kdx] ==
                            # #                                  self.sdp_variables[jdx][-1, ldx])
                for kdx in range(0, len(vars_idx)):
                    idx_kdx = vars_idx[kdx]
                    jdx_kdx = vars_jdx[kdx]
                    # Append the diagonal and constant (from homogenized variable) equalities
                    self.equality_constraints.append(self.sdp_variables[idx][idx_kdx, idx_kdx] ==
                                                     self.sdp_variables[jdx][jdx_kdx, jdx_kdx])
                    self.equality_constraints.append(self.sdp_variables[idx][-1, idx_kdx] ==
                                                     self.sdp_variables[jdx][-1, jdx_kdx])
                    for ldx in range(kdx+1, len(vars_idx)):
                        idx_ldx = vars_idx[ldx]
                        jdx_ldx = vars_jdx[ldx]
                        # Append the equalities involving two variables
                        self.equality_constraints.append(self.sdp_variables[idx][idx_kdx, idx_ldx] ==
                                                         self.sdp_variables[jdx][jdx_kdx, jdx_ldx])

    def set_fixed_constraints(self):
        """
        This populates a list (self.fixed_constraints_cvxpy) with the 'fixed' constraints for a problem in the form
        needed by cvxpy. These are combined with any auxiliary constraints for a particular problem instance.
        :return:
        """
        for idx, clique in enumerate(self.max_cliques):
            constraint_list = self.sdp_constraints[idx]
            Z = self.sdp_variables[idx]
            if self.chordal and not self.force_dense:
                # TODO: this ordering should be fine, but need some strict unit test
                constraint_map = self.clique_to_constraint_index[clique]
            for jdx, (A, c) in enumerate(constraint_list):
                if self.chordal and not self.force_dense:
                    # The first constraint (jdx == 0) is the homogenizing variable, it is always equality
                    is_inequality = jdx > 0 and self.are_constraints_inequalities[constraint_map[jdx-1]]
                else:
                    # The first constraint (jdx == 0) is the homogenizing variable, it is always equality
                    is_inequality = jdx > 0 and self.are_constraints_inequalities[jdx-1]
                if is_inequality:
                    self.fixed_constraints_cvxpy.append(cp.trace(A @ Z) <= c)
                else:
                    self.fixed_constraints_cvxpy.append(cp.trace(A @ Z) == c)
            # self.fixed_constraints_cvxpy += [cp.trace(A@Z) == c for (A, c) in constraint_list]

        # The equality constraints are only needed for overlapping elements in the chordally sparse case
        if self.chordal and not self.force_dense:
            self.fixed_constraints_cvxpy += self.equality_constraints

    def get_extra_constraints(self, extra_constraints: list) -> list:
        """
        Get a list of cvxpy constraints using this instance's SDP variables corresponding to the SymPy quadratic
        equality constraints in extra_constraints
        :param extra_constraints:
        :return: list of cvxpy constraints
        """
        cvx_constraints = []
        is_inequality = [type(cons) == sp.relational.LessThan for cons in extra_constraints]
        if self.chordal and not self.force_dense:
            for idx, cons in enumerate(extra_constraints):
                if is_inequality[idx]:
                    cons = sp.Eq(cons.args[0], cons.args[1], evaluate = False)
                var_set = cons.free_symbols
                for idx, clique in enumerate(self.max_cliques):
                    if var_set.issubset(clique):
                        Z = self.sdp_variables[idx]
                        poly_constraint = QcqpToSdpRelaxation.equality_to_polynomial(cons)
                        A, c = QcqpToSdpRelaxation.poly_to_sdp(list(clique), poly_constraint)
                        if is_inequality[idx]:
                            cvx_constraints.append(cp.trace(A@Z) <= c)
                        else:
                            cvx_constraints.append(cp.trace(A@Z) == c)
                        break
                else:
                    print("ERROR: this constraint does not respect sparsity: {:}".format(cons))
        else:
            full_var_set = list(self.variable_set)
            # For non-sparse case there is only one SDP variable
            Z = self.sdp_variables[0]
            for idx, cons in enumerate(extra_constraints):
                if is_inequality[idx]:
                    cons = sp.Eq(cons.args[0], cons.args[1], evaluate = False)
                poly_constraint = QcqpToSdpRelaxation.equality_to_polynomial(cons)
                A, c = QcqpToSdpRelaxation.poly_to_sdp(full_var_set, poly_constraint)
                if is_inequality[idx]:
                    cvx_constraints.append(cp.trace(A @ Z) <= c)
                else:
                    cvx_constraints.append(cp.trace(A @ Z) == c)
        return cvx_constraints

    def get_cost_function(self, cost):
        cost_terms = []
        if self.chordal and not self.force_dense:
            c = 0.0
            try:
                terms = cost.as_expr().as_terms()
            except AttributeError:
                assert np.isscalar(c), "Cost function {:} is messed up!".format(c)
                terms = ([c],)
            for term in terms[0]:
                term = term[0]
                if not term.is_constant():
                    for idx, clique in enumerate(self.max_cliques):
                        if term.free_symbols.issubset(clique):
                            A, const_part = QcqpToSdpRelaxation.poly_to_sdp(list(clique), term.as_poly())
                            Z = self.sdp_variables[idx]
                            cost_terms.append(cp.trace(A@Z))
                            cost_terms.append(-const_part)
                            break
                    else:
                        print("ERROR: this cost function's terms do not respect the sparsity: ")
                else:
                    c += term
            # Add in the constant term (for checking gap tightness, etc.)
            A_const = np.zeros(self.sdp_variables[0].shape)
            A_const[-1, -1] = c
            cost_terms.append(cp.trace(A_const @ self.sdp_variables[0]))
        else:
            Z = self.sdp_variables[0]
            A, c = QcqpToSdpRelaxation.poly_to_sdp(list(self.variable_set), cost.as_poly())
            cost_terms.append(cp.trace(A@Z) - c)

        return sum(cost_terms)

    def report_solution_info(self, tol=1e-6):
        """
        TODO Note that the last indexed row/column of Z should ALWAYS be the homogenizing variable!! Check this!!
        :param tol:
        :return:
        """
        if not self.force_dense and self.chordal:
            sol_list = []
            for idx, clique in enumerate(self.max_cliques):
                Z = self.sdp_variables[idx].value
                Z_eigs = np.linalg.eigvalsh(Z)
                print("Clique {:}".format(idx))
                print(clique)
                print("Z for clique {:}: ".format(idx))
                print(Z)
                print("Z_eigs for clique {:}: ".format(idx))
                print(Z_eigs)
                print("Eig ratio: {:}".format(Z_eigs[-1] / Z_eigs[-2]))
                x_list = []
                eig_list = list(Z_eigs)
                eig_list.reverse()
                for eig_idx, eig in enumerate(eig_list):
                    if np.abs(eig) > tol:
                        x, _ = QcqpToSdpRelaxation.extract_rank_1_homogeneous_solution(Z, eig_index=eig_idx,
                                                                                       include_homog=True)
                        print("{:}th largest eigenvalue's eigenvector solution in block {:}: ".format(eig_idx, idx))
                        print(x)
                        x_list.append(x)
                sol_list.append((Z, Z_eigs, x_list))
            return sol_list
        else:
            Z = self.sdp_variables[0].value
            Z_eigs = np.linalg.eigvalsh(Z)
            print("Z for dense solution: ")
            print(Z)
            print("Z_eigs: ")
            print(Z_eigs)
            print("Eig ratio: {:}".format(Z_eigs[-1] / Z_eigs[-2]))
            x_list = []
            eig_list = list(Z_eigs)
            eig_list.reverse()
            for eig_idx, eig in enumerate(eig_list):
                if np.abs(eig) > tol:
                    x, _ = QcqpToSdpRelaxation.extract_rank_1_homogeneous_solution(Z, eig_index=eig_idx,
                                                                                   include_homog=True)
                    print("{:}th largest eigenvalue's eigenvector solution: ".format(eig_idx))
                    print(x)
                    x_list.append(x)
            return [(Z, Z_eigs, x_list)]

    def extract_solution(self):
        """
        After solve() has been called on a cvxpy.Problem instance involving the variables in self.sdp_variables, attempt
        to extract a rank-1 solution (from each sparse sub-variable in the sparse case) if it exists.
        :return: tuple with dictionary mapping SymPy variable names to values (order independent this way) followed by
        a list of ranks (or single rank in dense case).
        """
        solution_dict = {}
        # For the chordal case, go through the entire set of variables
        if self.chordal and not self.force_dense:
            ranks = []
            for var in self.variable_set:
                if var not in solution_dict.keys():
                    # Go through each SDP grouping looking for this variable
                    for idx, clique in enumerate(self.max_cliques):
                        if var in clique:
                            Z = self.sdp_variables[idx].value
                            x_idx, rank_idx = QcqpToSdpRelaxation.extract_rank_1_homogeneous_solution(Z)
                            print("Full x: {:}".format(x_idx))
                            if x_idx is not None:
                                # Copy any currently unseen variables solved in this block
                                for jdx, var_jdx in enumerate(clique):
                                    if var_jdx not in solution_dict.keys():
                                        solution_dict[var_jdx] = x_idx[jdx]
                            if self.verbose:
                                Z_eigs = np.linalg.eigvalsh(Z)
                                print("Clique {:}".format(idx))
                                print(clique)
                                print("Z for clique {:}: ".format(idx))
                                print(Z)
                                print("Z_eigs for clique {:}: ".format(idx))
                                print(Z_eigs)
                                print("Eig ratio: {:}".format(Z_eigs[-1]/Z_eigs[-2]))
                            ranks.append(rank_idx)
        else:
            Z = self.sdp_variables[0].value
            # print("Z matrix: {:}".format(Z))
            x, rank = QcqpToSdpRelaxation.extract_rank_1_homogeneous_solution(Z)
            if x is not None:
                for idx, var in enumerate(self.variable_set):
                    # print("Idx, var: {:}, {:}".format(idx, var))
                    if var not in solution_dict.keys():
                        solution_dict[var] = x[idx]
            ranks = [rank]

        return solution_dict, ranks

    def solve_sdp(self, constraints: list, cost, params: SdpSolverParams = None):
        if params is None:
            params = SdpSolverParams()
            print("-------------------------------------------")
            print("INFO: solve_sdp using default solver params")
            print("-------------------------------------------")
        if params.feasibility:
            params.cost_function = 0.0
        if params.cost_function is not None:
            prob = cp.Problem(cp.Minimize(params.cost_function), constraints)
        else:
            try:
                prob = cp.Problem(cp.Minimize(cost), constraints)
            except TypeError:
                print("Cost: {:}".format(cost))
                print("Constraints: {:}".format(constraints))
        solved_successfully = False
        # try:
        if params.solver == cp.MOSEK:
            prob.solve(verbose=False, qcp=params.qcp, solver=params.solver, mosek_params=params.mosek_params)
        elif params.solver == cp.CVXOPT:
            prob.solve(verbose=True, qcp=params.qcp, solver=params.solver, max_iters=params.max_iters,
                       abstol=params.abstol, reltol=params.reltol,
                       feastol=params.feastol, refinement=params.refinement_steps, kkt_solver=params.kkt_solver)
        elif params.solver == cp.SCS:
            prob.solve(verbose=True, qcp=params.qcp, solver=params.solver, max_iters=params.max_iters,
                       eps=params.abstol, alpha=params.alpha,
                       scale=params.scale, normalize=params.normalize, use_indirect=params.use_indirect)
        solved_successfully = True
        # except cp.error.SolverError as e:
        #     print("{:} failed, let's investigate".format(params.solver))
        #     print(e)

        # TODO: use solved_successfully to determine what to do with the extraction
        if not solved_successfully:
            print("-------------------------------------------")
            print("WARNING: failed to solve the SDP relaxation")
            print("-------------------------------------------")

        solution_dict, ranks = self.extract_solution()
        if self.verbose:
            self.report_solution_info()
        return solution_dict, ranks, prob
