from graphik.graphs.graph_base import ProblemGraph
from graphik.solvers.sdp_formulations import SdpSolverParams, QcqpToSdpRelaxation
from graphik.solvers.solver_base import GraphProblemSolver
from graphik.solvers.constraints import angular_constraints, archimedean_constraint, constraints_from_graph, all_symbols


class SdpRelaxationSolver(GraphProblemSolver):
    """
    Lightweight wrapper for QcqpToSdpRelaxation that interacts with a Graph object's symbolic constraints.
    TODO: fast cvxpy parameterized cost functions that Filip mentioned
    TODO: add specific extra-constraints stuff as well for fast resolve with different constraints/anchors
    """
    def __init__(self, params=None, verbose=False, force_dense=False):
        if params is None:
            params = SdpSolverParams()
        super(SdpRelaxationSolver, self).__init__(params)
        self.sdp_relaxation = None  # Needs to be initialized with a first solve (or explicit assignment)
        self.verbose = verbose
        self.force_dense = force_dense
        self.cost = None

    @property
    def sdp_relaxation(self) -> QcqpToSdpRelaxation:
        return self._sdp_relaxation

    @sdp_relaxation.setter
    def sdp_relaxation(self, sdp_relaxation: QcqpToSdpRelaxation):
        self._sdp_relaxation = sdp_relaxation

    @property
    def verbose(self) -> bool:
        return self._verbose

    @verbose.setter
    def verbose(self, verbose: bool):
        self._verbose = verbose

    @property
    def force_dense(self) -> bool:
        return self._force_dense

    @force_dense.setter
    def force_dense(self, force_dense: bool):
        self._force_dense = force_dense

    @property
    def params(self) -> SdpSolverParams:
        return self._solver_params

    @params.setter
    def params(self, params: SdpSolverParams):
        self._solver_params = params

    def solve(self, graph: ProblemGraph, problem_params: dict = None):
        assert problem_params is not None, "Need a dictionary with end effector positions, etc."
        end_effector_assignment = problem_params["end_effector_assignment"]
        angular_limits = problem_params["angular_limits"] if "angular_limits" in problem_params else None
        angular_offsets = problem_params["angular_offsets"] if "angular_offsets" in problem_params else None
        as_equality = problem_params["as_equality"] if "as_equality" in problem_params else False
        archimedean = problem_params["archimedean"] if "archimedean" in problem_params else None

        cost = 0.0 if self.cost is None else self.cost
        constraints = constraints_from_graph(graph, end_effector_assignment)  #graph.symbolic_constraints()
        if angular_limits is not None:
            constraints += angular_constraints(graph, angular_limits, end_effector_assignment,
                                               angular_offsets, as_equality)
        if archimedean is not None:
            all_vars = all_symbols(constraints)
            constraints += archimedean_constraint(all_vars, [archimedean]*len(all_vars))
        self.sdp_relaxation = QcqpToSdpRelaxation(constraints, verbose=self.verbose, force_dense=self.force_dense)
        solution_dict, ranks, prob = self.sdp_relaxation.solve_sdp(self.sdp_relaxation.fixed_constraints_cvxpy,
                                                                   self.sdp_relaxation.get_cost_function(cost),
                                                                   self.params)

        return solution_dict, ranks, prob, constraints
