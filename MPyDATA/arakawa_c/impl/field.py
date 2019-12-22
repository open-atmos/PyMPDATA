class Field:
    class Impl:
        dimension: int
        halo: int

        def at(self, i, j=-1, k=-1):
            raise NotImplementedError()

        def focus(self, i, j=-1, k=-1):
            raise NotImplementedError()

        def set_axis(self, d:int):
            raise NotImplementedError()

    _impl = None

    @property
    def halo(self):
        return self._impl.halo

    @property
    def shape(self):
        return self._impl.shape

    @property
    def dimension(self):
        return self._impl.dimension

    def nd_sum(self, function, args, ext=0):
        assert ext < self.halo

        for arg in args:
            arg._impl.fill_halos()

        if len(args) == 1:
            self._impl.apply_1arg(function, args[0]._impl, ext)
        elif len(args) == 2:
            self._impl.apply_2arg(function, args[0]._impl, args[1]._impl, ext)
        elif len(args) == 2:
            self._impl._apply_3arg(function, args[0]._impl, args[1]._impl, args[2]._impl, ext)
        elif len(args) == 4:
            self._impl._apply_4arg(function, args[0]._impl, args[1]._impl, args[2]._impl, args[3]._impl, ext)
        else:
            raise NotImplementedError()

        self._impl.invalidate_halos()

    def swap_memory(self, other):
        self._impl, other._impl = other._impl, self._impl
