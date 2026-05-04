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
        "numpy >= 1.16, < 2",        # numpy 2 is step 5 of the dep ladder
        "scipy >= 1.17",
        "sympy >= 1.14",
        "matplotlib >= 3.1",
        "cvxpy >= 1.6, < 1.7",       # cvxpy 1.7+ requires numpy 2; deferred to step 5
        "networkx >= 3.6",
        # "networkx <= 2.8.7",
        "pymanopt == 0.2.5",         # pymanopt 2.x port is step 4
        "progress",
        "urdfpy @ git+ssh://git@github.com/utiasSTARS/urdfpy@master#egg=urdfpy",
        "trimesh",
        "numba >= 0.65",
        "pymlg @ git+https://github.com/decargroup/pymlg@8e6dc5ea61327ddfc2c8c1d16f276ae829f22db8#egg=pymlg",
        "pandas >= 0.24.2",
        "pytest",
    ],
    python_requires=">=3.10",
)
