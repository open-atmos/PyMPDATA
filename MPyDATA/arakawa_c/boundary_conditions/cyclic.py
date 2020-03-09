from ...utils import debug_flag

if debug_flag.VALUE:
    import MPyDATA.utils.fake_numba as numba
else:
    import numba


class CyclicLeft:
    @staticmethod
    @numba.njit
    def scalar(impl, d):
        a,b,c,e = impl.left_halo(d)
        f,g,h,j = impl.right_edge(d)
        impl.data[a:b, c:e][:] = impl.data[f:g, h:j][:]

    @staticmethod
    @numba.njit
    def vector(impl, axis, comp):
        data = impl.data(comp)
        data[impl.left_halo(axis, comp)][:] = data[impl.right_edge(axis, comp)][:]


class CyclicRight:
    @staticmethod
    @numba.njit
    def scalar(impl, d):
        a, b, c, e = impl.right_halo(d)
        f, g, h, j = impl.left_edge(d)
        impl.data[a:b, c:e][:] = impl.data[f:g, h:j][:]

    @staticmethod
    @numba.njit
    def vector(impl, axis, comp):
        data = impl.data(comp)
        data[impl.right_halo(axis, comp)][:] = data[impl.left_edge(axis, comp)][:]
