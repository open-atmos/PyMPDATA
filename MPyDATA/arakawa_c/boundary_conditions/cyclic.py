from ...utils import debug_flag

if debug_flag.VALUE:
    import MPyDATA.utils.fake_numba as numba
else:
    import numba


class CyclicLeft:
    @staticmethod
    @numba.njit
    def scalar(impl, d):
        impl.left_halo(d)[:] = impl.right_edge(d)[:]

    @staticmethod
    @numba.njit
    def vector(impl, axis, comp):
        impl.left_halo(axis, comp)[:] = impl.right_edge(axis, comp)[:]


class CyclicRight:
    @staticmethod
    @numba.njit
    def scalar(impl, d):
        impl.right_halo(d)[:] = impl.left_edge(d)[:]

    @staticmethod
    @numba.njit
    def vector(impl, axis, comp):
        impl.right_halo(axis, comp)[:] = impl.left_edge(axis, comp)[:]
