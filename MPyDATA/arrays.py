from .arakawa_c.vector_field import VectorField
import numpy as np


class Arrays:
    def __init__(self,
                 advectee,
                 advector,
                 g_factor
                 ):
        self.curr = advectee
        self.flux = VectorField.clone(advector)
        self.GC = advector
        self.g_factor = g_factor if g_factor is not None else np.empty((0, 0))
