from setuptools import setup, find_packages

# TODO: see https://github.com/pymanopt/pymanopt/blob/master/setup.py for mmore later
setup(
    name="graphIK",
    version="0.01",
    description="Graph-based kinematics library",
    author="Filip Maric, Matthew Giamou",
    author_email="filip.maric@robotics.utias.utoronto.ca, matthew.giamou@robotics.utias.utoronto.ca",
    license="MIT",
    url="https://github.com/utiasSTARS/graphIK",
    packages=find_packages(),
    package_data={
        'graphik': ['robots/urdfs/*', 'robots/urdfs/meshes/**/*'],
    },
    include_package_data=True,
    install_requires=[
        "numpy >= 1.16",
        "scipy >= 1.3.0",
        "sympy >= 1.5",
        "matplotlib >= 3.1",
        "cvxpy >= 1.1.0a1",
        "networkx >= 3.3",
        # "networkx <= 2.8.7",
        "pymanopt == 0.2.5",
        "progress",
        "urdfpy @ git+ssh://git@github.com/utiasSTARS/urdfpy@master#egg=urdfpy",
        "trimesh",
        "numba",
        "liegroups @ git+ssh://git@github.com/utiasSTARS/liegroups@generative_ik#egg=liegroups",
        "pandas >= 0.24.2",
        "pytest",
    ],
    python_requires=">=3.10",
)
