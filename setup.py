from distutils.core import setup
from platform import system, architecture

install_requires = []
if architecture()[0] == '64bit' and system() in ('Linux', 'Windows'): 
  install_requires += ['tbb>=2020.3.254']
# TODO: intel-openmp for OSX, icc_rt

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
    long_description='Numba-accelerated Pythonic implementation of MPDATA with Jupyter examples',
    install_requires=install_requires
)
