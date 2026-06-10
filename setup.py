from setuptools import setup, find_packages

setup(
    name="graphIK",
    version="0.2.0",
    description="Graph-based kinematics library",
    author="Filip Maric, Matthew Giamou",
    author_email="filip.maric@robotics.utias.utoronto.ca, matthew.giamou@robotics.utias.utoronto.ca",
    license="MIT",
    url="https://github.com/utiasSTARS/GraphIK",
    packages=find_packages(include=["graphik", "graphik.*"], exclude=["tests*", "experiments*"]),
    package_data={
        'graphik': ['robots/urdfs/*', 'robots/urdfs/meshes/**/*'],
    },
    include_package_data=True,
    install_requires=[
        "numpy >= 2",
        "scipy >= 1.17",
        "matplotlib >= 3.1",
        "cvxpy >= 1.8",
        "networkx >= 3.6",
        "yourdfpy >= 0.0.60",
        "trimesh >= 4.6.1",
        "numba >= 0.65",
        "pymlg @ git+https://github.com/decargroup/pymlg@8e6dc5ea61327ddfc2c8c1d16f276ae829f22db8#egg=pymlg",
    ],
    extras_require={
        "dev": ["pytest"],
    },
    python_requires=">=3.11",
)
