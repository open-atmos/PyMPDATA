class IField:
    halo: int
    axis: int
    shape: tuple
    dimension: int
    def at(self, i, j=-1, k=-1): raise NotImplementedError()
    def swap_memory(self, other): raise NotImplementedError()
    def fill_halos(self): raise NotImplementedError()
    def invalidate_halos(self): raise NotImplementedError()
    def apply(self, function, args, ext): raise NotImplementedError()
    def __init__(self): raise NotImplementedError()


class IScalarField(IField):
    def get(self): raise NotImplementedError()


class IVectorField(IField):
    def get_component(self, i: int): raise NotImplementedError()
