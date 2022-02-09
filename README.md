# graphIK
A library for solving inverse kinematics by modelling robots as geometric graphs and using ideas from distance geometry.

<img src="https://raw.githubusercontent.com/utiasSTARS/GraphIK/main/assets/graph_ik_logo.png" width="250px"/>


## Dependencies
GraphIK is implemented in Python 3. See [setup.py](https://github.com/utiasSTARS/graphIK/blob/main/setup.py) for a full list of dependencies.


## Usage
Use of GraphIK can be summarized into 4 key steps which we'll walk through below (see the scripts in [experiments/](https://github.com/utiasSTARS/graphik-internal/tree/main/experiments) for more details).

### 1. Load a Robot
In this example, we'll parse a [URDF file](https://industrial-training-master.readthedocs.io/en/melodic/_source/session3/Intro-to-URDF.html) describing a [Schunk LWA4P manipulator](https://github.com/marselap/schunk_lwa4p). 

```
from graphik.utils.roboturdf import load_schunk_lwa4d
robot, graph = load_schunk_lwa4d()
```
GraphIK's interface between robot models and IK solvers is the abstract [`ProblemGraph`](https://github.com/utiasSTARS/graphIK/blob/main/graphik/graphs/graph_base.py) class. For the LWA4P, we'll use `ProblemGraphRevolute`, a subclass of `ProblemGraph` that can represent 3D robots with revolute joints.

### 2. Instantiate a ProblemGraph Object with Obstacles
If you are considering an environment with spherical obstacles, you can include constraints that prevent collisions with them. In this example, we will use a set of spheres that approximate a table: 
```
from graphik.utils.utils import table_environment
obstacles = table_environment()
# This loop is not needed if you are not using obstacle avoidance constraints 
for idx, obs in enumerate(obstacles):
    graph.add_spherical_obstacle(f"o{idx}", obs[0], obs[1])
```

### 3. Specify a Goal Pose
Interfaces to our solvers require a goal pose defined by the [`lieroups`](https://github.com/utiasSTARS/liegroups) library. For this simple example, using the robot's forward kinematics is the fastest way to get a sample goal pose:
```
q_goal = robot.random_configuration()
T_goal = robot.pose(q_goal, f"p{robot.n}")
```

### 4. Solve the IK Problem
The main purpose of our graphical interpretation of robot kinematics is the development of distance geometric IK solvers. One example is the [Riemannian optimization-based solver](https://arxiv.org/abs/2011.04850) implemented in [`RiemannianSolver`](https://github.com/utiasSTARS/graphIK/blob/main/graphik/solvers/riemannian_solver.py). 

```
from graphik.solvers.riemannian_solver import solve_with_riemannian
q_sol, solution_points = solve_with_riemannian(graph, T_goal)  # Returns None if infeasible or didn't solve
```

For a similar example using [`CIDGIK`](https://arxiv.org/abs/2109.03374), a convex optimization-based approach, please see [experiments/cidgik_example.py](https://github.com/utiasSTARS/graphIK/blob/main/experiments/cidgik_example.py).

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
