from distutils.core import setup

setup(
    name='PyMPDATA',
    version='0.0.0',
    packages=[
        'PyMPDATA',
        'PyMPDATA/formulae',
        'PyMPDATA/arakawa_c',
        'PyMPDATA/arakawa_c/boundary_condition',
    ],
    license='GPL v3',
    long_description='Numba-accelerated Pythonic implementation of MPDATA with Jupyter examples'
)
