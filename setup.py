from setuptools import setup, find_packages

# TODO: see https://github.com/pymanopt/pymanopt/blob/master/setup.py for mmore later
setup(
    name='graphIK',
    version='0.01',
    description='Graph-based kinematics library',
    author='Filip Maric, Matthew Giamou',
    author_email='filip.maric@robotics.utias.utoronto.ca, matthew.giamou@robotics.utias.utoronto.ca',
    license='MIT',
    url='https://github.com/utiasSTARS/graphIK',
    packages=find_packages(),
    python_requires='>=3.7',
)
