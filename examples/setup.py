""" the magick behind ``pip install ...`` """

import os

from setuptools import find_packages, setup


def get_long_description():
    """returns contents of README.md file"""
    with open("README.md", "r", encoding="utf8") as file:
        long_description = file.read()
    return long_description


CI = "CI" in os.environ

setup(
    name="PyMPDATA-examples",
    description="PyMPDATA usage examples reproducing results from literature"
    " and depicting how to use PyMPDATA in Python from Jupyter notebooks",
    use_scm_version={
        "local_scheme": lambda _: "",
        "version_scheme": "post-release",
        "root": "..",
    },
    setup_requires=["setuptools_scm"],
    install_requires=[
        "PyMPDATA",
        "open-atmos-jupyter-utils",
        "pystrict",
        "matplotlib",
        "ipywidgets" + "<8.0.3" if CI else "",
        "scipy",
        "pint",
        "joblib",
        "sympy",
        "h5py",
        "imageio",
    ],
    author="https://github.com/open-atmos/PyMPDATA/graphs/contributors",
    license="GPL-3.0",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["PyMPDATA_examples", "PyMPDATA_examples.*"]),
    package_data={"": ["*/*/*.txt"]},
    include_package_data=True,
    project_urls={
        "Tracker": "https://github.com/open-atmos/PyMPDATA/issues",
        "Documentation": "https://open-atmos.github.io/PyMPDATA",
        "Source": "https://github.com/open-atmos/PyMPDATA",
    },
)
