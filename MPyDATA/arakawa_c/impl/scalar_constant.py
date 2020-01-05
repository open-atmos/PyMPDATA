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

        # TODO: works only for 1D
        def focus(self, _):
            pass

    return ScalarConstant(arg_value)
