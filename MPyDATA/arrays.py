from .arakawa_c.vector_field import VectorField

class Arrays:
    def __init__(self,
        state,
        GC
    ):
        self.curr = state
        self.flux = VectorField.clone(GC)
        self.GC = GC
