from distutils.core import setup
from platform import system, architecture

install_requires = []
if architecture()[0] == '64bit' and system() == 'Windows': # TODO: also available for Linux but hangs GithubActions!
  install_requires += ['tbb>=2020.3.254']
if architecture()[0] == '64bit' and system() == 'Linux': 
  install_requires += ['icc_rt>=2020.0.133', 'intel-openmp>=2020.0.133']
if architecture()[0] == '64bit' and system() == 'Darwin': 
  install_requires += ['intel-openmp>=2019.0']

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
