import numpy as np
import sympy as sp
import time
from scipy.optimize import minimize, show_options
from graphik.solvers.solver_base import GraphProblemSolver
from graphik.graphs.graph_base import RobotGraph
from graphik.utils.utils import (
    norm_sq,
    list_to_variable_dict,
    variable_dict_to_list,
    list_to_variable_dict_spherical,
    flatten,
)


# TODO: build a better options interface with, e.g., show_options(solver='minimize', method='L-BFGS-B')
class LocalSolver(GraphProblemSolver):
    """
    Uses a built in Python local solver on joint variable formulations
    """

    def __init__(self, params=None, verbose=False, force_dense=False):
        if params is None:
            # TODO: try L-BFGS and Trust Region, etc. for unconstrained case!
            params = {"solver": "L-BFGS-B", "tol": 1e-6, "maxiter": 1000}
        super(LocalSolver, self).__init__(params)
        # self.cost = None
        self.f_cost = None
        self.grad = None
        self.hess = None

    @property
    def params(self):
        return self._solver_params

    @params.setter
    def params(self, params):
        self._solver_params = params

    @staticmethod
    def gradient(f, v):
        return sp.Matrix([f]).jacobian(v)

    @staticmethod
    def symbolic_cost_function(
        robot,
        end_effector_assignment,
        variable_angles,
        use_trigsimp: bool = False,
        use_grad: bool = True,
        use_hess: bool = True,
        pose_cost: bool = False,
    ):
        dZ = np.zeros(robot.dim)
        dZ[-1] = 1.0
        # sym_vars = sp.symbols(list(variable_angles.keyset()))
        sym_vars = {}
        if not robot.spherical:
            for var in variable_angles:
                sym_vars[var] = sp.symbols(var)
        else:
            for var in variable_angles:
                sym_vars[var] = sp.symbols([var + "_1", var + "_2"])
        cost_sym = 0.0
        for query_node in end_effector_assignment:

            if pose_cost:
                T_query = end_effector_assignment[query_node]
                T_ee = robot.get_pose(sym_vars, query_node)
                cost_sym += norm_sq(
                    T_ee.rot.as_matrix() @ dZ
                    + T_ee.trans
                    - T_query.rot.as_matrix() @ dZ
                    - T_query.trans
                )
                cost_sym += norm_sq(T_ee.trans - T_query.trans)
            else:
                cost_sym += norm_sq(
                    robot.get_pose(sym_vars, query_node).trans
                    - end_effector_assignment[query_node].trans
                )

        if use_trigsimp:
            cost_sym = sp.trigsimp(cost_sym)

        if not robot.spherical:

            def sym_cost(angles):
                angles_dict_str = list_to_variable_dict(angles)
                angles_dict = {}
                for key in angles_dict_str:
                    angles_dict[sp.symbols(key)] = angles_dict_str[key]
                return cost_sym.subs(angles_dict)

        else:

            def sym_cost(angles):
                angles_dict_str = list_to_variable_dict_spherical(angles)
                angles_dict = {}
                for key in angles_dict_str:
                    angles_dict[sp.symbols(key)] = angles_dict_str[key]
                return cost_sym.subs(angles_dict)

        if robot.spherical:
            arg_list = flatten(sym_vars.values())
        else:
            arg_list = list(sym_vars.values())

        if use_grad:
            # Define sym gradient and hessian
            cost_sym_grad = LocalSolver.gradient(cost_sym, arg_list)
            if use_trigsimp:
                cost_sym_grad = sp.trigsimp(cost_sym_grad)

            if not robot.spherical:

                def sym_grad(angles):
                    angles_dict_str = list_to_variable_dict(angles)
                    angles_dict = {}
                    for key in angles_dict_str:
                        angles_dict[sp.symbols(key)] = angles_dict_str[key]
                    return cost_sym_grad.subs(angles_dict)

            else:

                def sym_grad(angles):
                    angles_dict_str = list_to_variable_dict_spherical(angles)
                    angles_dict = {}
                    for key in angles_dict_str:
                        angles_dict[sp.symbols(key)] = angles_dict_str[key]
                    return cost_sym_grad.subs(angles_dict)

        else:
            sym_grad = None

        if use_hess:
            cost_sym_hess = sp.hessian(cost_sym, arg_list)
            if use_trigsimp:
                cost_sym_hess = sp.trigsimp(cost_sym_hess)

            if not robot.spherical:

                def sym_hess(angles):
                    angles_dict_str = list_to_variable_dict(angles)
                    angles_dict = {}
                    for key in angles_dict_str:
                        angles_dict[sp.symbols(key)] = angles_dict_str[key]
                    return cost_sym_hess.subs(angles_dict)

            else:

                def sym_hess(angles):
                    angles_dict_str = list_to_variable_dict_spherical(angles)
                    angles_dict = {}
                    for key in angles_dict_str:
                        angles_dict[sp.symbols(key)] = angles_dict_str[key]
                    return cost_sym_hess.subs(angles_dict)

        else:
            sym_hess = None

        return sym_cost, sym_grad, sym_hess

    def set_symbolic_cost_function(
        self,
        robot,
        end_effector_assignment,
        variable_angles,
        use_trigsimp: bool = False,
        use_grad: bool = True,
        use_hess: bool = True,
        pose_cost: bool = False,
    ):
        f, grad, hess = LocalSolver.symbolic_cost_function(
            robot,
            end_effector_assignment,
            variable_angles,
            use_trigsimp=use_trigsimp,
            use_grad=use_grad,
            use_hess=use_hess,
            pose_cost=pose_cost,
        )
        self.f_sym = f
        self.grad_sym = grad
        self.hess_sym = hess

        # Lambdify the symbolic functions
        if not robot.spherical:
            x = sp.symarray("x", (len(variable_angles),))
        else:
            x = sp.symarray("x", (len(variable_angles) * 2,))
        f_lam = sp.lambdify([x], f(x), "scipy")

        grad_lam = sp.lambdify([x], grad(x), "scipy") if use_grad else None
        hess_lam = sp.lambdify([x], hess(x), "scipy") if use_hess else None

        self.f_cost = f_lam
        self.grad = lambda x: grad_lam(x).squeeze() if use_grad else None
        self.hess = hess_lam

    def set_procedural_cost_function(
        self, robot, end_effector_assignment, pose_cost=False, do_grad_and_hess=False
    ):
        if pose_cost:

            def cost_func(angles):
                node_inputs = list_to_variable_dict(angles)
                cost = 0.0
                dZ = np.zeros(robot.dim)
                dZ[-1] = 1.0
                for query_node in end_effector_assignment:

                    T_query = robot.get_pose(node_inputs, query_node)
                    T_goal = end_effector_assignment[query_node]
                    cost += norm_sq(T_query.trans - T_goal.trans)
                    cost += norm_sq(
                        T_query.rot.as_matrix() @ dZ
                        + T_query.trans
                        - T_goal.rot.as_matrix() @ dZ
                        - T_goal.trans
                    )
                return cost

        elif robot.spherical:

            def cost_func(angles):
                node_inputs = list_to_variable_dict_spherical(angles)
                cost = 0.0
                for query_node in end_effector_assignment:
                    cost += norm_sq(
                        robot.get_pose(node_inputs, query_node).trans
                        - end_effector_assignment[query_node].trans
                    )
                return cost

        else:

            def cost_func(angles):
                node_inputs = list_to_variable_dict(angles)
                cost = 0.0
                for query_node in end_effector_assignment:
                    cost += norm_sq(
                        robot.get_pose(node_inputs, query_node).trans
                        - end_effector_assignment[query_node].trans
                    )
                return cost

            if do_grad_and_hess:

                def grad_function(angles):
                    return robot.jacobian_cost(
                        list_to_variable_dict(angles), end_effector_assignment
                    )

                def hess_function(angles):
                    return robot.hessian_cost(
                        list_to_variable_dict(angles), end_effector_assignment
                    )

                self.grad = grad_function
                self.hess = hess_function

        self.f_cost = cost_func

    def set_revolute_cost_function(
        self, robot, end_effector_assignment, variable_angles, pose_cost=False
    ):
        """
        TODO: Finish this function
        :param robot:
        :param end_effector_assignment:
        :param variable_angles:
        :return:
        """
        # Use the generic, lambdified method to set self.f_cost (and maybe self.grad?)
        # self.set_symbolic_cost_function(robot, end_effector_assignment, variable_angles,
        #                                 use_trigsimp=False,
        #                                 use_grad=False,
        #                                 use_hess=False)
        self.set_procedural_cost_function(robot, end_effector_assignment, pose_cost)
        dZ = np.zeros(robot.dim)
        dZ[-1] = 1.0
        if pose_cost:

            def grad_function(angles):
                angles_dict = list_to_variable_dict(angles)
                J = robot.jacobian(angles_dict)
                J_z = robot.jacobian(angles_dict, pose_term=True)
                grad = np.zeros(len(angles_dict))
                for ee in end_effector_assignment:
                    T_goal = end_effector_assignment[ee]
                    T_ee = robot.get_pose(angles_dict, ee)
                    grad = grad + J[ee].T @ (T_ee.trans - T_goal.trans)
                    grad = grad + J_z[ee].T @ (
                        T_ee.rot.as_matrix() @ dZ
                        + T_ee.trans
                        - T_goal.rot.as_matrix() @ dZ
                        - T_goal.trans
                    )
                return 2.0 * grad.astype("float")

        else:

            def grad_function(angles):
                angles_dict = list_to_variable_dict(angles)
                J = robot.jacobian(
                    angles_dict, ee_keys=list(end_effector_assignment.keys())
                )
                grad = np.zeros(len(angles_dict))
                for ee in end_effector_assignment:
                    pose_ee = robot.get_pose(angles_dict, ee).trans
                    grad = grad + J[ee].T @ (
                        pose_ee - end_effector_assignment[ee].trans
                    )
                return 2.0 * grad.astype("float")

        self.grad = grad_function

        if pose_cost:

            def hess_function(angles):
                angles_dict = list_to_variable_dict(angles)
                J = robot.jacobian(angles_dict)
                J_z = robot.jacobian(angles_dict, pose_term=True)
                K = robot.hessian_linear_symb(angles_dict, J)
                K_z = robot.hessian_linear_symb(angles_dict, J_z, pose_term=True)
                r = {}
                r_z = {}
                for ee in end_effector_assignment:
                    T_ee = robot.get_pose(angles_dict, ee)
                    T_goal = end_effector_assignment[ee]
                    r[ee] = end_effector_assignment[ee].trans - T_ee.trans
                    r_z[ee] = (
                        T_goal.rot.as_matrix() @ dZ
                        + T_goal.trans
                        - T_ee.rot.as_matrix() @ dZ
                        - T_ee.trans
                    )
                H = robot.euclidean_cost_hessian(J, K, r)
                H_z = robot.euclidean_cost_hessian(J_z, K_z, r_z)
                return 2 * (H.astype("float") + H_z.astype("float"))

        else:

            def hess_function(angles):
                angles_dict = list_to_variable_dict(angles)
                J = robot.jacobian(
                    angles_dict, ee_keys=list(end_effector_assignment.keys())
                )
                K = robot.hessian_linear_symb(
                    angles_dict, J, ee_keys=list(end_effector_assignment.keys())
                )
                r = {}
                for ee in end_effector_assignment:
                    r[ee] = (
                        end_effector_assignment[ee].trans
                        - robot.get_pose(angles_dict, ee).trans
                    )
                H = robot.euclidean_cost_hessian(J, K, r)
                return 2 * H.astype("float")

        self.hess = hess_function

    def solve(self, graph: RobotGraph, problem_params: dict = None):
        assert problem_params is not None, "Need a dictionary with initial guess, etc."
        angular_limits = (
            problem_params["angular_limits"]
            if "angular_limits" in problem_params
            else list_to_variable_dict(graph.robot.n * [np.pi])
        )
        angular_offsets = (
            problem_params["angular_offsets"]
            if "angular_offsets" in problem_params
            else list_to_variable_dict(graph.robot.n * [0.0])
        )
        initial_guess = (
            problem_params["initial_guess"]
            if "initial_guess" in problem_params
            else list_to_variable_dict(graph.robot.n * [0.0])
        )
        # Form the x0 vector from the dictionary
        x0 = np.array(variable_dict_to_list(initial_guess))
        if graph.robot.spherical:
            x0 = np.array(flatten(x0))

        tol_other = 1e-9  # Default tolerance (for non gtol)
        if self.params["solver"] in ("L-BFGS-B", "TNC", "SLSQP"):

            if self.params["solver"] == "L-BFGS-B":
                options = {
                    "ftol": tol_other,
                    "gtol": self.params["tol"],
                    "maxiter": self.params["maxiter"],
                    "maxfun": 1e9,
                }
            elif self.params["solver"] == "TNC":
                options = {
                    "ftol": tol_other,
                    "xtol": tol_other,
                    "gtol": self.params["tol"],
                    "maxiter": self.params["maxiter"],
                }
            else:
                options = {"ftol": tol_other, "maxiter": self.params["maxiter"]}
            # Form the angular bounds (symmetrical for now)
            bounds = [
                (-angular_limits[node], angular_limits[node]) for node in angular_limits
            ]
            # The first angular variable is unbounded each time
            if graph.robot.spherical:
                new_bounds = []
                for b in bounds:
                    new_bounds.append((-np.inf, np.inf))
                    new_bounds.append(b)
                bounds = new_bounds
            time_start = time.time()
            results = minimize(
                self.f_cost,
                x0,
                method=self.params["solver"],
                jac=self.grad,
                bounds=bounds,
                options=options,
            )
            results.runtime = time.time() - time_start
        elif self.params["solver"] in ("BFGS", "CG"):
            options = {"gtol": self.params["tol"], "maxiter": self.params["maxiter"]}
            time_start = time.time()
            results = minimize(
                self.f_cost,
                x0,
                method=self.params["solver"],
                jac=self.grad,
                options=options,
            )
            results.runtime = time.time() - time_start
        elif self.params["solver"] in ("trust-constr"):
            options = {
                "xtol": tol_other,
                "barrier_tol": tol_other,
                "gtol": self.params["tol"],
                "maxiter": self.params["maxiter"],
                "initial_tr_radius": self.params["initial_tr_radius"]
                if "initial_tr_radius" in self.params
                else 1.0,
            }
            # Form the angular bounds (symmetrical for now)
            # bounds = [
            #     (-angular_limits[node], angular_limits[node]) for node in angular_limits
            # ]
            bounds = []
            for node in angular_limits:
                limit = angular_limits[node]
                bounds += [(-limit, limit)] if limit != np.pi else [(None, None)]
            # The first angular variable is unbounded each time
            if graph.robot.spherical:
                new_bounds = []
                for b in bounds:
                    new_bounds.append((-np.inf, np.inf))
                    new_bounds.append(b)
                bounds = new_bounds
            time_start = time.time()
            results = minimize(
                self.f_cost,
                x0,
                method=self.params["solver"],
                jac=self.grad,
                hess=self.hess,
                bounds=bounds,
                options=options,
            )
            results.runtime = time.time() - time_start
        elif self.params["solver"] in ("Newton-CG", "dogleg", "trust-exact"):
            if self.params["solver"] == "Newton-CG":
                options = {"xtol": tol_other, "maxiter": self.params["maxiter"]}
            else:
                options = {
                    "gtol": self.params["tol"],
                    "initial_tr_radius": self.params["initial_tr_radius"]
                    if "initial_tr_radius" in self.params
                    else 1.0,
                }
            time_start = time.time()
            results = minimize(
                self.f_cost,
                x0,
                method=self.params["solver"],
                jac=self.grad,
                hess=self.hess,
                options=options,
            )
            results.runtime = time.time() - time_start
        else:
            raise (
                ValueError,
                "Solver {:} not supported.".format(self.params["solver"]),
            )

        # TODO: Extract/return the solution
        return results
