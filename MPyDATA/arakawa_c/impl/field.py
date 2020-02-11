from ..traversal import Traversal


class Field:
    class Impl:
        dimension: int
        halo: int
        axis: int

        def at(self, i: [int, float], j: [int, float] = -1, k: [int,float] = -1):
            raise NotImplementedError()

        def focus(self, i: int, j: int = -1, k: int = -1):
            raise NotImplementedError()

        def set_axis(self, d: int):
            raise NotImplementedError()

    _impl = None
    _halo_valid: bool = False

    def __init__(self, halo):
        self._halo = halo

    @property
    def halo(self):
        return self._halo

    @property
    def dimension(self):
        return self._impl.dimension

    def apply(self, traversal: Traversal, args, ext=0):
        assert ext < self.halo

        for arg in args:
            arg.fill_halos()

        if len(args) == 1:
            if traversal.operator == 'min':
                self._impl.min_1arg(traversal.logic, args[0]._impl, ext)
            elif traversal.operator == 'max':
                self._impl.max_1arg(traversal.logic, args[0]._impl, ext)
            elif traversal.operator == 'set':
                self._impl.set_1arg(traversal.logic, args[0]._impl, ext)
            elif traversal.operator == 'sum':
                self._impl.sum_1arg(traversal.logic, args[0]._impl, ext)
            else:
                raise NotImplementedError()
        elif len(args) == 2:
            if traversal.operator == 'sum':
                self._impl.sum_2arg(traversal.logic, args[0]._impl, args[1]._impl, ext)
            elif traversal.operator == 'set':
                self._impl.set_2arg(traversal.logic, args[0]._impl, args[1]._impl, ext)
            else:
                raise NotImplementedError()
        elif len(args) == 3:
            if traversal.operator == 'sum':
                self._impl.sum_3arg(traversal.logic, args[0]._impl, args[1]._impl, args[2]._impl, ext)
            elif traversal.operator == 'set':
                self._impl.set_3arg(traversal.logic, args[0]._impl, args[1]._impl, args[2]._impl, ext)
            else:
                raise NotImplementedError()
        elif len(args) == 4:
            if traversal.operator == 'set':
                self._impl.set_4arg(traversal.logic, args[0]._impl, args[1]._impl, args[2]._impl, args[3]._impl, ext)
            elif traversal.operator == 'sum':
                self._impl.sum_4arg(traversal.logic, args[0]._impl, args[1]._impl, args[2]._impl, args[3]._impl, ext)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

        self._halo_valid = False

    def swap_memory(self, other):
        self._impl, other._impl = other._impl, self._impl
        self._halo_valid, other._halo_valid = other._halo_valid, self._halo_valid

    def _fill_halos_impl(self):
        raise NotImplementedError()

    def fill_halos(self):
        if self._halo_valid or self.halo == 0:
            return
        self._fill_halos_impl()
        self._halo_valid = True
