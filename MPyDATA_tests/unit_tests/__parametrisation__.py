"""
Created at 18.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import pytest
from .params import params


@pytest.fixture(params=[
    1
])
def halo(request):
    return request.param


@pytest.fixture(params=params)
def case(request):
    return request.param
