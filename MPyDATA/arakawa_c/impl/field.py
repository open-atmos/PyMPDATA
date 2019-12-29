class Field:
    class Impl:
        dimension: int
        halo: int

        def at(self, i: [int, float], j: [int, float] = -1, k: [int,float] = -1):
            raise NotImplementedError()

        def focus(self, i: int, j: int = -1, k: int = -1):
            raise NotImplementedError()

        def set_axis(self, d: int):
            raise NotImplementedError()

    _impl = None
    _halo_valid: bool = False

    def __init__(self, halo, grid):
        self._halo = halo
        self._grid = grid

    @property
    def halo(self):
        return self._halo

    @property
    def grid(self):
        return self._grid

    @property
    def dimension(self):
        return self._impl.dimension

    def nd_sum(self, function, args, ext=0):
        assert ext < self.halo

        for arg in args:
            arg.fill_halos()

        if len(args) == 1:
            self._impl.apply_1arg(function, args[0]._impl, ext)
        elif len(args) == 2:
            self._impl.apply_2arg(function, args[0]._impl, args[1]._impl, ext)
        elif len(args) == 3:
            self._impl.apply_3arg(function, args[0]._impl, args[1]._impl, args[2]._impl, ext)
        elif len(args) == 4:
            self._impl.apply_4arg(function, args[0]._impl, args[1]._impl, args[2]._impl, args[3]._impl, ext)
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
