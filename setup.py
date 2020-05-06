from distutils.core import setup

setup(
    name='MPyDATA',
    version='0.0.0',
    packages=[
        'MPyDATA',
        'MPyDATA/formulae',
        'MPyDATA/arakawa_c',
        'MPyDATA/arakawa_c/boundary_condition',
    ],
    license='GPL v3',
    long_description='Numba-accelerated Pythonic implementation of MPDATA with Jupyter examples'
)
