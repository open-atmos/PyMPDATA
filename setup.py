from setuptools import setup, find_packages


def get_long_description():
    with open("README.md", "r", encoding="utf8") as file:
        long_description = file.read()
    return long_description


setup(
    name='PyMPDATA',
    description='Numba-accelerated Pythonic implementation of MPDATA with Jupyter examples',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    install_requires=['numba==0.53.1',
                      'numpy>=1.20.2',
                      'ghapi'],
    author='https://github.com/atmos-cloud-sim-uj/PyMPDATA/graphs/contributors',
    license="GPL-3.0",
    packages=find_packages(include=['PyMPDATA', 'PyMPDATA.*']),
    long_description=get_long_description(),
    long_description_content_type = "text/markdown",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development :: Libraries'
    ],
    keywords='atmospheric-modelling, numba, numerical-integration, advection, pde-solver, advection-diffusion'
)