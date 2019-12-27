import sys
from importlib import reload
from . import debug_flag


class DisabledNumba:
    def __enter__(*_):
        debug_flag.VALUE = True
        DisabledNumba.__reload()

    def __exit__(*_):
        debug_flag.VALUE = False
        DisabledNumba.__reload()

    @staticmethod
    def __reload():
        modules = [
            module for module in sys.modules.values() if "MPyDATA." in str(module) and ".debug_flag" not in str(module)
        ]
        modules.reverse()
        for module in modules:
            reload(module)
