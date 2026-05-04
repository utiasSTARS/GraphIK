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
        "numpy >= 2",
        "scipy >= 1.17",
        "sympy >= 1.14",
        "matplotlib >= 3.1",
        "cvxpy >= 1.8",
        "networkx >= 3.6",
        # "networkx <= 2.8.7",
        "pymanopt >= 2.2.1",
        "progress",
        "yourdfpy >= 0.0.60",
        "trimesh >= 4.6.1",
        "numba >= 0.65",
        "pymlg @ git+https://github.com/decargroup/pymlg@8e6dc5ea61327ddfc2c8c1d16f276ae829f22db8#egg=pymlg",
        "pandas >= 0.24.2",
        "pytest",
    ],
    python_requires=">=3.10",
)
