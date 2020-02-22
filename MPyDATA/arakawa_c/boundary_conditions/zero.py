from ...utils import debug_flag

if debug_flag.VALUE:
    import MPyDATA.utils.fake_numba as numba
else:
    import numba


class ZeroLeft:
    @staticmethod
    @numba.njit
    def scalar(impl, d):
        impl.data[impl.left_halo(d)] = 0

    @staticmethod
    @numba.njit
    def vector(impl, axis, comp):
        impl.data(comp)[impl.left_halo(axis, comp)] = 0


class ZeroRight:
    @staticmethod
    @numba.njit
    def scalar(impl, d):
        impl.data[impl.right_halo(d)] = 0

    @staticmethod
    @numba.njit
    def vector(impl, axis, comp):
        impl.data(comp)[impl.right_halo(axis, comp)] = 0
