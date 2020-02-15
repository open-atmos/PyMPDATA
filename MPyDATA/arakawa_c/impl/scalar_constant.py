from MPyDATA.utils import debug_flag

if debug_flag.VALUE:
    import MPyDATA.utils.fake_numba as numba
else:
    import numba


def make_scalar_constant(arg_value: float):
    @numba.jitclass([
        ('value', numba.float64),
    ])
    class ScalarConstant:
        def __init__(self, value):
            self.value = value

        def focus(self, _, __=-1):
            pass

        def set_axis(self, _):
            pass

        def at(self, _, __=-1):
            return self.value

    return ScalarConstant(arg_value)
