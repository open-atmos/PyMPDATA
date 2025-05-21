"""the magick behind ``pip install ...``"""

import os
import re

from setuptools import find_packages, setup


def get_long_description():
    """returns contents of the pdoc landing site with pdoc links converted into URLs"""
    with open("docs/pympdata_examples_landing.md", "r", encoding="utf8") as file:
        pdoc_links = re.compile(
            r"(`)([\w\d_-]*).([\w\d_-]*)(`)", re.MULTILINE | re.UNICODE
        )
        return pdoc_links.sub(
            r'<a href="https://open-atmos.github.io/PyMPDATA/\2/\3.html">\3</a>',
            file.read(),
        )


CI = "CI" in os.environ

setup(
    name="pympdata-examples",
    description="PyMPDATA usage examples reproducing results from literature"
    " and depicting how to use PyMPDATA in Python from Jupyter notebooks",
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
        "imageio",
        "meshio",
        "pandas",
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
