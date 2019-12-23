import sys
from importlib import reload
from MPyDATA_tests.utils import debug


class DisabledNumba:
    def __enter__(*_):
        debug.DEBUG = True
        DisabledNumba.__reload()

    def __exit__(*_):
        debug.DEBUG = False
        DisabledNumba.__reload()

    @staticmethod
    def __reload():
        modules = [module for module in sys.modules.values() if "MPyDATA." in str(module)]
        modules.reverse()
        for module in modules:
            reload(module)
