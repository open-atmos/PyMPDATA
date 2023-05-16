"""
the magick behind ``pip install ...``
"""
import os

from setuptools import find_packages, setup


def get_long_description():
    """returns contents of README.md file"""
    with open("README.md", "r", encoding="utf8") as file:
        long_description = file.read()
    return long_description


setup(
    name="PyMPDATA",
    description="Numba-accelerated Pythonic implementation of MPDATA "
    "with examples in Python, Julia and Matlab",
    use_scm_version={"local_scheme": lambda _: "", "version_scheme": "post-release"},
    setup_requires=["setuptools_scm"],
    install_requires=[
        "numba" + ("==0.56.4" if "CI" in os.environ else ""),
        "numpy" + ("==1.21.6" if "CI" in os.environ else ""),
        "pystrict",
    ],
    extras_require={
        "tests": [
            "matplotlib>=3.2.2",
            "scipy==1.7.3",
            "jupyter-core<5.0.0",
            "ipywidgets!=8.0.3",
            "ghapi",
            "pytest",
            "pytest-benchmark",
            # this is a PyMPDATA-examples dependency, as of time of writing
            # the pip package depends on deprecated distutils, which cause
            # a warning on Python 3.10, to be removed after joblib release
            "joblib @ git+https://github.com/joblib/joblib@3d80506#egg=joblib",
        ]
    },
    author="https://github.com/open-atmos/PyMPDATA/graphs/contributors",
    author_email="sylwester.arabas@uj.edu.pl",
    license="GPL-3.0",
    packages=find_packages(include=["PyMPDATA", "PyMPDATA.*"]),
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries",
    ],
    keywords="atmospheric-modelling, numba, numerical-integration, "
    "advection, pde-solver, advection-diffusion",
    project_urls={
        "Tracker": "https://github.com/open-atmos/PyMPDATA/issues",
        "Documentation": "https://open-atmos.github.io/PyMPDATA",
        "Source": "https://github.com/open-atmos/PyMPDATA",
    },
)
