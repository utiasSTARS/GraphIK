import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

import math
import time

from graphik.robots.robot_base import Robot
from graphik.utils.utils import generate_rotation_matrix, list_to_variable_dict

from liegroups.numpy import SO2
from liegroups.numpy import SO3
from liegroups.numpy import SE2
from liegroups.numpy import SE3


class solver_fabrik:
    def __init__(self, params):
        self.params = params

        self.N = params["N"]

        self.r = params["r"]
        self.angle_limit = params["angle_limit"]
        self.parents = params["parents"]

        self.goal_index = params["goal_index"]
        self.goal_position = params["goal_position"]

        self.dim = params["dim"]

    def generate_random_configuration(self, returnAllPositions=False):
        positions = np.zeros((self.N, 3))

        for i in range(1, self.N):
            parent_index = self.parents[i]

            phi = (2 * np.random.random() - 1) * self.angle_limit[i]
            theta = np.random.random() * 2 * math.pi

            if self.parents[parent_index] == -1:
                if self.dim == 2:
                    axis = np.array([1, 0, 0])
                else:
                    axis = np.array([0, 0, 1])
            else:
                axis = (
                    positions[parent_index, :]
                    - positions[self.parents[parent_index], :]
                )
                axis = axis / la.norm(axis)

            if self.dim == 2:
                relative_direction = generate_rotation_matrix(
                    phi, np.array([0, 0, 1])
                ).dot(axis.reshape(3, 1))
            else:
                perp = np.array([0, 0, 0])
                f = np.random.randint(0, 3)
                if axis[f] == 0:
                    perp[f] = 1
                else:
                    perp = np.array([1, 1, 1])
                    perp[f] = 0
                    perp[f] = -perp.dot(axis) / axis[f]
                    perp = perp / la.norm(perp)

                relative_direction = generate_rotation_matrix(theta, axis).dot(
                    generate_rotation_matrix(phi, perp).dot(axis.reshape(3, 1))
                )

            positions[i, :] = (
                positions[parent_index, :]
                + relative_direction.reshape(
                    3,
                )
                * self.r[parent_index]
            )

        output = []

        for i in range(len(self.goal_index)):
            output += [positions[self.goal_index[i], :]]

        if returnAllPositions:
            return positions
        else:
            return output

    def get_parents(self, n):
        parents = self.parents

        ind = [n]
        r = parents[n]
        while r != -1:
            ind = [r] + ind
            r = parents[r]

        return ind

    def iterate(self, p, isFirst=False, isStuck=False, perturbation=2):
        r = self.r
        angle_limit = self.angle_limit

        N = self.N

        p1 = np.zeros((N, 3))
        p2 = np.zeros((N, 3))

        n1 = np.zeros((N, 1))

        # Backward
        for k in range(len(self.goal_index)):
            index = self.goal_index[k]
            n1[index] = -1
            p1[index, :] = np.reshape(self.goal_position[k], (3,))

        for k in range(len(self.goal_index)):
            index = self.goal_index[k]
            chain = self.get_parents(index)

            orientation = np.zeros((3,))
            for i in range(len(chain) - 2, -1, -1):
                j1 = chain[i]
                j2 = chain[i + 1]

                if n1[j1] == -1:
                    continue

                relative_direction = np.array(
                    (p[j1, :] - p1[j2, :]) / la.norm(p[j1, :] - p1[j2, :])
                )

                if np.any(np.isnan(relative_direction)):
                    print("Testing NaN values")

                if np.all((orientation == 0)):
                    orientation = relative_direction
                else:
                    angle = math.acos(
                        max(min(1, orientation.dot(relative_direction)), -1)
                    )
                    limit = angle_limit[j2]

                    if angle > limit:
                        axis = np.cross(relative_direction, orientation)
                        axis = axis / la.norm(axis)

                        if np.any(np.isnan(axis)):
                            print("Testing NaN values")

                        R = generate_rotation_matrix(angle - limit, axis)

                        relative_direction = R.dot(relative_direction)

                        angle = math.acos(
                            min(1, max(-1, orientation.dot(relative_direction)))
                        )

                new_pos = relative_direction * r[j1] + p1[j2, :]

                # TODO: Check this change
                orientation = relative_direction

                p1[j1, :] = (p1[j1, :] * n1[j1] + new_pos) / (n1[j1] + 1)
                n1[j1] += 1

                if False:
                    _, ax_handle = plt.subplots()
                    plt.plot(p1[:, 0], p1[:, 1], "b-o")
                    plt.plot(self.goal_position[0][0], self.goal_position[0][1], "gx")
                    plt.plot(self.goal_position[1][0], self.goal_position[1][1], "gx")
                    plt.grid()
                    plt.title(
                        "After Backward Pass for Goal {:} and Node {:}".format(k, j1)
                    )
                    ax_handle.set_aspect("equal", adjustable="box")
                    plt.show()

        # Forward
        p2[0, :] = np.zeros((3,))
        orientation = np.zeros((N, 3))

        angles = np.zeros((N,))

        isLinear = False
        for i in range(len(self.goal_index)):
            goal = self.goal_position[i]
            if (goal[1] == 0) and (goal[2] == 0):
                isLinear = True
                break

        for k in range(1, N):
            chain = self.get_parents(k)

            j1 = chain[-1]
            j2 = chain[-2]

            relative_direction = (p1[j1, :] - p2[j2, :]) / la.norm(
                p1[j1, :] - p2[j2, :]
            )

            if np.all((orientation[j2, :] == 0)):
                if self.dim == 2:
                    orientation[j2, :] = [1, 0, 0]
                else:
                    orientation[j2, :] = [0, 0, 1]

            angle = math.acos(
                max(-1, min(1, orientation[j2, :].dot(relative_direction)))
            )
            limit = angle_limit[j1]

            if (angle > limit) or (isLinear and isFirst) or (isStuck):
                fix = 0
                if angle > limit:
                    axis = np.cross(relative_direction, orientation[j2, :])
                    axis = axis / la.norm(axis)

                    fix = angle - limit
                else:
                    if self.dim == 2:
                        axis = np.array([0, 0, 1])
                    else:
                        axis = np.array([1, 1, 1])
                    axis = axis / la.norm(axis)

                    # fix = max(min(limit+angle, (2*random.random() - 1) / 180 * math.pi * perturbation), limit-angle)
                    fix = (2 * np.random.random() - 1) / 180 * math.pi * perturbation

                R = generate_rotation_matrix(fix, axis)
                relative_direction = R.dot(relative_direction)

            angle_new = math.acos(
                max(min(orientation[j2, :].dot(relative_direction), 1), -1)
            )

            angles[k] = angle_new

            orientation[j1, :] = relative_direction
            p2[j1, :] = relative_direction * r[j2] + p2[j2, :]

            if angle_new > limit * 1.01:  # Repeat if the angular limit is not satisfied
                k = k - 1

            if False:
                _, ax_handle = plt.subplots()
                plt.plot(p2[:, 0], p2[:, 1], "b-o")
                plt.plot(self.goal_position[0][0], self.goal_position[0][1], "gx")
                plt.plot(self.goal_position[1][0], self.goal_position[1][1], "gx")
                plt.grid()
                plt.title("After Forward Pass for Goal {:}".format(k))
                ax_handle.set_aspect("equal", adjustable="box")
                plt.show()

        return p2, angles

    def individual_errors(self, p):
        return [
            la.norm(p[self.goal_index[i], :] - self.goal_position[i])
            for i in range(len(self.goal_index))
        ]

    def error(self, p):
        return sum(self.individual_errors(p))

    # Initial Guess: An Nx3 array containing the position of each joint in the initial guess;
    # Max-Iteration: The maximum number of iterations in the solver;
    # Error-threshold: The maximum tolerance for error to solve the problem;
    # sensitivity and sensitivity_range: If the fractional change in the error was less than sensitivity over the past sensitivity_range, the iterations halt and the final status is marked as Unsolved.
    def solve(
        self,
        initial_guess=None,
        max_iteration=65,
        error_threshold=0.001,
        sensitivity=0.00001,
        sensitivity_range=50,
        errorPause=False,
        showAnimation=False,
        frameNumber=10,
    ):
        status = "Unsolved"

        if initial_guess is None:
            initial_guess = self.generate_random_configuration(returnAllPositions=True)

        # Current position
        p = initial_guess

        # Iterations
        time_start = time.time()

        errors = [self.error(p)]
        max_errors = [max(self.individual_errors(p))]
        isStuck = False

        # Showing the iterations
        counter = 1
        solution = {"positions": p}
        self.print_solution(solution, show=showAnimation)

        if False:
            _, ax_handle = plt.subplots()
            plt.plot(p[:, 0], p[:, 1], "b-o")
            plt.plot(self.goal_position[0][0], self.goal_position[0][1], "gx")
            plt.plot(self.goal_position[1][0], self.goal_position[1][1], "gx")
            plt.grid()
            plt.title("Initial Guess")
            ax_handle.set_aspect("equal", adjustable="box")
            plt.show()

        for i in range(max_iteration):
            p, angles = self.iterate(p, isFirst=(i == 0), isStuck=isStuck)

            # Showing the iterations
            if i * frameNumber / counter > max_iteration:
                counter += 1
                solution = {"positions": p}
                self.print_solution(solution, show=showAnimation)

            e = self.error(p)
            e_max = max(self.individual_errors(p))
            max_errors.append(e_max)
            errors += [e]

            if e < error_threshold:
                status = "Solved"
                break

            if i > sensitivity_range:
                max_change = 0
                for j in range(i - sensitivity_range, i):
                    change = abs(errors[i] - errors[j]) / errors[i]
                    if max_change < change:
                        max_change = change

                if max_change < sensitivity:
                    # status = "Unsolved"
                    # break
                    isStuck = True
                else:
                    isStuck = False

        time_end = time.time()
        # Calculating the runtime of the algorithm
        runtime = time_end - time_start

        solution = {
            "positions": p,
            "angles": angles,
            "final_error": self.error(p),
            "initial_guess": initial_guess,
            "iterations": i + 1,
            "runtime": runtime,
            "error_per_iteration": errors,
            "status": status,
            "success": (status == "solved"),
            "max_error_per_iteration": max_errors,
        }

        if False:
            _, ax_handle = plt.subplots()
            plt.plot(p[:, 0], p[:, 1], "b-o")
            plt.plot(self.goal_position[0][0], self.goal_position[0][1], "gx")
            plt.plot(self.goal_position[1][0], self.goal_position[1][1], "gx")
            plt.grid()
            plt.title("Final configuration")
            ax_handle.set_aspect("equal", adjustable="box")
            plt.show()

        if errorPause:
            # if solution["status"] != "Solved":
            if solution["final_error"] > 0.1:
                print(solution["final_error"])
                print(initial_guess)
                print(self.parents)
                print(self.r)
                print(self.angle_limit)
                print(self.goal_index)
                print(self.goal_position)
                input()

        return solution

    def print_information(self, solution):
        solver = self

        joint_positions = solution["positions"]

        print("----------------")

        print("Problem stsatus: %s" % solution["status"])
        print("****************")
        print("Joint Locations")
        for i in range(np.shape(joint_positions)[0]):
            print(
                "%d: (%f, %f, %f)"
                % (
                    i,
                    joint_positions[i, 0],
                    joint_positions[i, 1],
                    joint_positions[i, 2],
                )
            )
        print("****************")
        print("Targets")
        for i in range(len(solver.goal_index)):
            print(
                "%d: (%f, %f, %f)"
                % (
                    solver.goal_index[i],
                    solver.goal_position[i][0],
                    solver.goal_position[i][1],
                    solver.goal_position[i][2],
                )
            )
        print("****************")
        print("Joint angles/Joint angle limits: ")
        print("(%f/%f)" % (solution["angles"][1], self.angle_limit[1]), end="")
        for i in range(2, len(self.angle_limit)):
            print(", (%f/%f)" % (solution["angles"][i], self.angle_limit[i]), end="")
        print()
        print("****************")
        print("Number of iterations: %d" % solution["iterations"])
        print("****************")
        print("Runtime of the algorithm: %f seconds" % solution["runtime"])
        print("****************")
        print("Final error: %f" % solution["final_error"])

        print("----------------")

        print()

        return True

    def print_solution(self, solution, print_enabled=False, show=False):
        solver = self

        if print_enabled:
            self.print_information(solution)

        if not show:
            return True

        joint_positions = solution["positions"]

        fig = plt.figure()
        if self.dim == 3:
            ax = fig.add_subplot(111, projection="3d")

            for k in range(len(solver.goal_position)):
                c = solver.get_parents(solver.goal_index[k])

                for i in range(len(c) - 1):

                    x = [joint_positions[c[i], 0], joint_positions[c[i + 1], 0]]
                    y = [joint_positions[c[i], 1], joint_positions[c[i + 1], 1]]
                    z = [joint_positions[c[i], 2], joint_positions[c[i + 1], 2]]

                    ax.plot3D(x, y, z, c="b")

            ax.scatter3D(
                joint_positions[:, 0],
                joint_positions[:, 1],
                joint_positions[:, 2],
                c="r",
            )

            for k in range(len(solver.goal_position)):
                ax.scatter3D(
                    solver.goal_position[k][0],
                    solver.goal_position[k][1],
                    solver.goal_position[k][2],
                    c="g",
                    marker="o",
                    s=50,
                )

            ax.set_xlim3d(-4, 4)
            ax.set_ylim3d(-4, 4)
            ax.set_zlim3d(-4, 4)

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
        else:
            ax = fig.add_subplot(111)
            plt.grid(True)

            for k in range(len(solver.goal_position)):
                c = solver.get_parents(solver.goal_index[k])

                for i in range(len(c) - 1):

                    x = [joint_positions[c[i], 0], joint_positions[c[i + 1], 0]]
                    y = [joint_positions[c[i], 1], joint_positions[c[i + 1], 1]]

                    ax.plot(x, y, c="b")

            ax.scatter(joint_positions[:, 0], joint_positions[:, 1], c="r")

            for k in range(len(solver.goal_position)):
                ax.scatter(
                    solver.goal_position[k][0],
                    solver.goal_position[k][1],
                    c="g",
                    marker="o",
                    s=50,
                )

            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)

            ax.set_xlabel("x")
            ax.set_ylabel("y")

        plt.show()

        return True
