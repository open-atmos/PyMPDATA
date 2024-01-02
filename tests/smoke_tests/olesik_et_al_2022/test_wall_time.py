# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import os

import pytest
from PyMPDATA_examples.Olesik_et_al_2022 import wall_time


@pytest.mark.xfail(condition=os.name == "nt")
def test_wall_time():
    wall_time.test_wall_time()
