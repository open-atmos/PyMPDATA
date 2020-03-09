from .arakawa_c.scalar_field import ScalarField
from .arakawa_c.vector_field import VectorField
import numba

class Arrays:
    def __init__(self,
        state: ScalarField,
        GC_field: VectorField,
    ):
        self._curr = state
        self._prev = ScalarField.clone(state)
        self._flux = VectorField.clone(GC_field)
        self._GC_phys = GC_field
        self._halo = state.halo
        self.swaped = False

    @property
    def curr(self):
        return self._curr if not self.swaped else self._prev
