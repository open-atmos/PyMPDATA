from importlib import reload
from MPyDATA.utils import debug
from MPyDATA.fields import _scalar_field_1d, _scalar_field_2d, _vector_field_1d, _vector_field_2d, utils
from MPyDATA.formulae import antidiff, fct_utils, flux, upwind

packages = [
    _scalar_field_1d, _scalar_field_2d, _vector_field_1d, _vector_field_2d, utils,
    antidiff, fct_utils, flux, upwind
]


class DisabledNumba:
    def __enter__(*_):
        debug.DEBUG = True
        for package in packages:
            reload(package)

    def __exit__(*_):
        debug.DEBUG = False
        for package in packages:
            reload(package)
