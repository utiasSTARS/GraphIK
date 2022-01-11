# graphIK
A library for solving inverse kinematics by modelling robots as geometric graphs and using ideas from distance geometry.

<img src="https://raw.githubusercontent.com/utiasSTARS/GraphIK/main/assets/graph_ik_logo.png" width="250px"/>


## Dependencies
GraphIK is implemented in Python 3. See [requirements.txt](https://github.com/utiasSTARS/graphIK/blob/main/requirements.txt) for a full list of dependencies.


## Usage
Use of GraphIK can be summarized into 4 key steps which we'll walk through below. First, we'll need a few imports:

```
import graphik
import numpy as np
```

### 1. Define a Robot
In this example, we'll parse a [URDF file](https://industrial-training-master.readthedocs.io/en/melodic/_source/session3/Intro-to-URDF.html) describing a [Schunk LWA4P manipulator](https://github.com/marselap/schunk_lwa4p). 

```
from graphik.robots.robot_base import RobotRevolute
from graphik.utils.roboturdf import RobotURDF
fname = graphik.__path__[0] + "/robots/urdfs/ur10.urdf"
urdf_robot = RobotURDF(fname)
robot = urdf_robot.make_Revolute3d(ub, lb)  # make the Revolute class from a URDF
```

### 2. Instantiate a Graph Object
GraphIK's interface between robot models and IK solvers is the abstract [`Graph`](https://github.com/utiasSTARS/graphIK/blob/main/graphik/graphs/graph_base.py) class. For the LWA4P, we'll use `RobotRevoluteGraph`, a subclass of `Graph`.
```
from graphik.graphs.graph_base import RobotRevoluteGraph
graph = RobotRevoluteGraph(robot)
```
### 3. Select and Instantiate a Solver
The main purpose of our graphical interpretation of robot kinematics is the development of distance geometric IK solvers. One example is the [Riemannian optimization-based solver](https://arxiv.org/abs/2011.04850) implemented in [`RiemannianSolver`](https://github.com/utiasSTARS/graphIK/blob/main/graphik/solvers/riemannian_solver.py). 
```
from graphik.solvers.riemannian_solver import RiemannianSolver
solver = RiemannianSolver(graph)
```

### 4. Solve for a Goal Pose
For simplicity, we can use a random configuration to get a goal pose that we know is reachable. From this configuration, we can define the objects needed by `RiemannianSolver`.
```
from graphik.utils.utils import list_to_variable_dict, trans_axis
from graphik.utils.dgp import pos_from_graph, adjacency_matrix_from_graph, bound_smoothing, graph_from_pos
q_goal = robot.random_configuration()
G_goal = graph.realization(q_goal)
X_goal = pos_from_graph(G_goal)
D_goal = graph.distance_matrix_from_joints(q_goal)
T_goal = robot.get_pose(list_to_variable_dict(q_goal), f"p{n}")
goals = {
    f"p{n}": T_goal.trans,
    f"q{n}": T_goal.dot(trans_axis(axis_len, "z")).trans,
}
G = graph.complete_from_pos(goals)
omega = adjacency_matrix_from_graph(G)
```

This allows us to run our solver and extract our solution points and the runtime:
```
lb, ub = bound_smoothing(G)
sol_info = solver.solve(D_goal, omega, use_limits=False, bounds=(lb, ub))
Y = sol_info["x"]
t_sol = sol_info["time"]
```
The solution points `Y` are correct up to a [Euclidean transformation](https://en.wikipedia.org/wiki/Rigid_transformation), so we must solve an [orthogonal Procrustes problem](https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem) to align these points to our robot's base and the end-effector goal in our task space with [`best_fit_transform`](https://github.com/utiasSTARS/graphIK/blob/main/graphik/utils/utils.py):
```
from graphik.utils.utils import best_fit_transform
align_ind = list(np.arange(graph.dim + 1))
for name in goals.keys():
    align_ind.append(graph.node_ids.index(name))

R, t = best_fit_transform(Y[align_ind, :], X_goal[align_ind, :])
P_e = (R @ Y.T + t.reshape(3, 1)).T
```
Which can be used to finally extract the angular configuration `q_sol` from our point-based solution.
```
G_sol = graph_from_pos(P_e, graph.node_ids)
T_g = {f"p{n}": T_goal}
q_sol = robot.joint_variables(G_sol, T_g)
```

See [experiments/simple_ik_examples/](https://github.com/utiasSTARS/graphIK/tree/main/experiments/simple_ik_examples) for further examples on other types of robots, including planar and spherical manipulators.

For an example using a convex optimization-based approach and incorporating spherical obstacles, please see [experiments/cidgik_example.py](https://github.com/utiasSTARS/graphIK/blob/main/experiments/cidgik_example.py).

## Publications and Related Work
If you use any of this code in your research, kindly cite any relevant publications listed here.

### Riemannian Optimization 
arXiv: [Inverse Kinematics as Low-Rank Euclidean Distance Matrix Completion](https://arxiv.org/abs/2011.04850)

arXiv: [Riemannian Optimization for Distance-Geometric Inverse Kinematics](https://arxiv.org/abs/2108.13720)

```bibtex
@misc{marić2021riemannian,
      title={Riemannian Optimization for Distance-Geometric Inverse Kinematics}, 
      author={Filip Marić and Matthew Giamou and Adam W. Hall and Soroush Khoubyarian and Ivan Petrović and Jonathan Kelly},
      year={2021},
      eprint={2108.13720},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

### Semidefinite Programming (SDP) Relaxations

#### CIDGIK
arXiv: [Convex Iteration for Distance-Geometric Inverse Kinematics](https://arxiv.org/abs/2109.03374)
```bibtex
@misc{giamou2022convex,
      title={Convex Iteration for Distance-Geometric Inverse Kinematics}, 
      author={Matthew Giamou and Filip Marić and David M. Rosen and Valentin Peretroukhin and Nicholas Roy and Ivan Petrović and Jonathan Kelly},
      year={2022},
      eprint={2109.03374},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

#### Sparse Sum-of-Squares Optimization for Planar and Spherical IK 
arXiv: [Inverse Kinematics for Serial Kinematic Chains via Sum of Squares Optimization](https://arxiv.org/abs/1909.09318)

MATLAB Code: https://github.com/utiasSTARS/sos-ik

```bibtex
@inproceedings{maric2020inverse,
  title={Inverse Kinematics for Serial Kinematic Chains via Sum of Squares Optimization},
  author={Mari{\'c}, Filip and Giamou, Matthew and Khoubyarian, Soroush and Petrovi{\'c}, Ivan and Kelly, Jonathan},
  booktitle={2020 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={7101--7107},
  year={2020},
  organization={IEEE}
}
```
