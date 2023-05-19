""" checks if all example Jupyter notebooks have file size less than a certain limit """
import os
import pathlib

import pint
import pytest

from tests.devops_tests.test_todos_annotated import findfiles

SI = pint.UnitRegistry()


@pytest.fixture(
    params=findfiles(
        (pathlib.Path(__file__).parent.parent.parent / "PyMPDATA-examples").absolute(),
        r".*\.(ipynb)$",
    ),
    name="notebook",
)
def _notebook(request):
    return request.param


def test_example_notebook_size(notebook):
    """returns True if a given notebook matches file size criterion"""
    assert os.stat(notebook).st_size * SI.byte < 12 * SI.megabyte  # TODO #392
