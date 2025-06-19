"""
the magick behind ``pip install ...``
"""

import os
import platform
import sys

from setuptools import find_packages, setup


def get_long_description():
    """returns contents of README.md file"""
    with open("README.md", "r", encoding="utf8") as file:
        long_description = file.read()
        long_description = long_description.replace(
            "pympdata_logo.svg", "pympdata_logo.png"
        )
    return long_description


CI = "CI" in os.environ
_32bit = platform.architecture()[0] == "32bit"

setup(
    name="pympdata",
    description="Numba-accelerated Pythonic implementation of MPDATA "
    "with examples in Python, Julia, Rust and Matlab",
    install_requires=[
        "numba"
        + (
            {
                8: "<0.57.0",
                9: "<0.57.0",
                10: "<0.57.0",
                11: "==0.57.0",
                12: "==0.57.0",
                13: "==0.57.0",
            }[sys.version_info.minor]
            if CI and not _32bit
            else ""
        ),
        "numpy"
        + (
            {
                8: "==1.24.4",
                9: "==1.24.4",
                10: "==1.24.4",
                11: "==1.24.4",
                12: "==1.26.4",
                13: "==2.2.5",
            }[sys.version_info.minor]
            if CI
            else ""
        ),
        "pystrict",
    ],
    extras_require={
        "tests": [
            "PyMPDATA-examples",
            "matplotlib" + (">=3.2.2" if CI else ""),
            "scipy"
            + (
                {
                    8: "==1.10.1",
                    9: "==1.10.1",
                    10: "==1.10.1",
                    11: "==1.10.1",
                    12: "==1.13.0",
                    13: "==1.15.3",
                }[sys.version_info.minor]
                if CI and not _32bit
                else ""
            ),
            "jupyter-core" + ("<5.0.0" if CI else ""),
            "jupyter_client" + ("==8.6.3" if CI else ""),
            "ipywidgets" + ("==8.1.7" if CI else ""),
            "ipykernel" + ("==7.0.0a1" if CI else ""),
            "ghapi",
            "pytest",
            "pytest-benchmark",
            "joblib" + ("==1.4.0" if CI else ""),
            "imageio",
            "nbformat",
        ]
    },
    author="https://github.com/open-atmos/PyMPDATA/graphs/contributors",
    author_email="sylwester.arabas@agh.edu.pl",
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
