from setuptools import setup, find_packages

# TODO: see https://github.com/pymanopt/pymanopt/blob/master/setup.py for mmore later
setup(
    name='cvxik',
    version='0.1',
    description='Convex optimization for inverse kinematics',
    author='Filip Maric, Matthew Giamou',
    author_email='filip.maric@robotics.utias.utoronto.ca, matthew.giamou@robotics.utias.utoronto.ca',
    license='MIT',
    url='https://github.com/utiasSTARS/cvxik',
    packages=find_packages(),
    python_requires='>=3.7',
)
