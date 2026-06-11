# GraphIK
GraphIK is a library for solving inverse kinematics problems by modelling robots as geometric graphs and using concepts from distance geometry.

<img src="https://raw.githubusercontent.com/utiasSTARS/GraphIK/main/assets/graph_ik_logo.png" width="250px"/>

## Installation

GraphIK is implemented in Python 3.11. The recommended setup uses conda for the Python environment and pip for runtime dependencies (declared in [`setup.py`](https://github.com/utiasSTARS/GraphIK/blob/main/setup.py)).

```bash
git clone https://github.com/utiasSTARS/GraphIK.git
cd GraphIK
conda env create -f environment.yml
conda activate graphik_public
```

This produces a working install on linux, macOS, and Windows by letting pip resolve runtime deps for your platform.

**macOS arm64 fallback.** If pip resolution against `setup.py` picks a bad combination on macOS arm64 (you already ran the primary recipe above and the install completed but doesn't work), reinstall from the known-good lockfile we ship:

```bash
pip install -r requirements-macos-arm64.txt
pip install -e . --no-deps
```

## SDP solvers (Mosek recommended)

GraphIK's SDP-relaxation solvers (`solve_with_cidgik` in `graphik/solvers/convex_iteration.py`, the SDP formulations in `graphik/solvers/sdp_*.py`) run out of the box on a free solver (Clarabel by default, falling back to SCS or CVXOPT) bundled with `cvxpy`. `experiments/cidgik_example.py` works without any extra setup.

[Mosek](https://www.mosek.com/) is **recommended**: it's faster and gives tighter precision than the free alternatives on these SDPs. It's a commercial solver with a free academic license. If installed, GraphIK auto-detects it and uses it as the default — no code changes required.

To enable the Mosek path:

1. Request an academic license at https://www.mosek.com/products/academic-licenses/.
2. Install the Mosek Python interface into the active env:

   ```bash
   pip install mosek
   ```

3. Place the license file (`mosek.lic`) at the path Mosek expects (typically `~/mosek/mosek.lic`).

The SDP tests in `tests/test_sdp_snl*.py` run without Mosek: constraint-construction tests are solver-free, and the end-to-end solves use whichever free solver cvxpy picks, with tolerances loose enough for it.

## Usage
Use of GraphIK can be summarized by four key steps, which we'll walk through below (see the scripts in [experiments/](https://github.com/utiasSTARS/GraphIK/tree/main/experiments) for more details).

### 1. Load a Robot
In this example, we'll parse a [URDF file](https://industrial-training-master.readthedocs.io/en/melodic/_source/session3/Intro-to-URDF.html) describing a [Schunk LWA4P manipulator](https://github.com/marselap/schunk_lwa4p). 

```python
from graphik.utils.roboturdf import load_schunk_lwa4d
robot, graph = load_schunk_lwa4d()
```
GraphIK's interface between robot models and IK solvers is the [`ProblemGraph`](https://github.com/utiasSTARS/GraphIK/blob/main/graphik/graphs/graph.py) class, a single concrete `nx.DiGraph` subclass that handles both 2D and 3D revolute robots (the workspace dimension is read from the supplied `Robot`).

### 2. Instantiate a ProblemGraph Object with Obstacles
If you are considering an environment with spherical obstacles, you can include constraints that prevent collisions. In this example, we will use a set of spheres that approximate a table: 

```python
from graphik.utils.utils import table_environment
obstacles = table_environment()
# This loop is not needed if you are not using obstacle avoidance constraints 
for idx, obs in enumerate(obstacles):
    graph.add_spherical_obstacle(f"o{idx}", obs[0], obs[1])
```

### 3. Specify a Goal Pose
Interfaces to our solvers require a goal pose defined as a 4×4 SE(3) numpy array (we use the [`pymlg`](https://github.com/decargroup/pymlg) library internally for SE(3) operations). For this simple example, using the robot's forward kinematics is the fastest way to get a sample goal pose:

```python
q_goal = robot.random_configuration()
T_goal = robot.pose(q_goal, f"p{robot.n}")
```

### 4. Solve the IK Problem
The main purpose of our graphical interpretation of robot kinematics is to develop distance-geometric IK solvers. One example is the [Riemannian optimization-based solver](https://arxiv.org/abs/2011.04850) implemented in [`RiemannianSolver`](https://github.com/utiasSTARS/GraphIK/blob/main/graphik/solvers/riemannian.py). 

```python
from graphik.solvers import RiemannianSolver

solver = RiemannianSolver(graph)
result = solver.solve(T_goal)
q_sol = result.q if result.feasible else None  # feasible == within joint limits
```

For a similar example using [`CIDGIK`](https://arxiv.org/abs/2109.03374), a convex optimization-based approach, please see [experiments/cidgik_example.py](https://github.com/utiasSTARS/GraphIK/blob/main/experiments/cidgik_example.py).

## Publications and Related Work
If you use any of this code in your research work, please kindly cite the relevant publications listed here.

### Riemannian Optimization 

IEEE Transactions on Robotics: [Riemannian Optimization for Distance-Geometric Inverse Kinematics](https://ieeexplore.ieee.org/document/9631368/)

```bibtex
@article{marić2022riemannian,
  author = {Filip Mari\'{c} and Matthew Giamou and Adam W. Hall and Soroush Khoubyarian and Ivan Petrović and Jonathan Kelly},
  journal = {{IEEE} Transactions on Robotics},
  month = {June},
  number = {3},
  pages = {1703--1722},
  title = {Riemannian Optimization for Distance-Geometric Inverse Kinematics},
  volume = {38},
  year = {2022}
}
```

arXiv: [Riemannian Optimization for Distance-Geometric Inverse Kinematics](https://arxiv.org/abs/2108.13720)

```bibtex
@misc{marić2021riemannian_arxiv,
  author={Filip Marić and Matthew Giamou and Adam W. Hall and Soroush Khoubyarian and Ivan Petrović and Jonathan Kelly},
  title={Riemannian Optimization for Distance-Geometric Inverse Kinematics}, 
  year={2021},
  eprint={2108.13720},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}
```
arXiv: [Inverse Kinematics as Low-Rank Euclidean Distance Matrix Completion](https://arxiv.org/abs/2011.04850)

### Semidefinite Programming (SDP) Relaxations

#### CIDGIK

IEEE Robotics & Automation Letters: [Convex Iteration for Distance-Geometric Inverse Kinematics](https://ieeexplore.ieee.org/document/9677911)

```bibtex
@article{giamou2022convex,
  author = {Matthew Giamou and Filip Marić and David M. Rosen and Valentin Peretroukhin and Nicholas Roy and Ivan Petrović and Jonathan Kelly},
  journal = {{IEEE} Robotics and Automation Letters},
  month = {April},
  number = {2},
  pages = {1952--1959},
  title = {Convex Iteration for Distance-Geometric Inverse Kinematics},
  volume = {7},
  year = {2022}
}
```

arXiv: [Convex Iteration for Distance-Geometric Inverse Kinematics](https://arxiv.org/abs/2109.03374)

```bibtex
@misc{giamou2022convex_arxiv,
  author={Matthew Giamou and Filip Marić and David M. Rosen and Valentin Peretroukhin and Nicholas Roy and Ivan Petrović and Jonathan Kelly},
  title={Convex Iteration for Distance-Geometric Inverse Kinematics}, 
  year={2022},
  eprint={2109.03374},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}
```

#### Sparse Sum-of-Squares Optimization for Planar and Spherical IK

IEEE ICRA 2020: [Inverse Kinematics for Serial Kinematic Chains via Sum of Squares Optimization](https://ieeexplore.ieee.org/document/9196704)

```bibtex
@inproceedings{marić2020inverse,
  address = {Paris, France},
  author = {Filip Marić and Matthew Giamou and Soroush Khoubyarian and Ivan Petrović and Jonathan Kelly},
  booktitle = {Proceedings of the {IEEE} International Conference on Robotics and Automation {(ICRA})},
  pages = {7101--7107},
  title = {Inverse Kinematics for Serial Kinematic Chains via Sum of Squares Optimization},
  year = {2020}
}
```

arXiv: [Inverse Kinematics for Serial Kinematic Chains via Sum of Squares Optimization](https://arxiv.org/abs/1909.09318)

```bibtex
@misc{marić2022convex_arxiv,
  author={Filip Marić and Matthew Giamou and Soroush Khoubyarian and Ivan Petrović and Jonathan Kelly},
  title={Inverse Kinematics for Serial Kinematic Chains via Sum of Squares Optimization}, 
  year={2020},
  eprint={1909.09318},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}
```

MATLAB Code: https://github.com/utiasSTARS/sos-ik
