from .arakawa_c.scalar_field import ScalarField
from .arakawa_c.vector_field import VectorField
from .options import Options


# TODO: rename (data)
class Arrays:
    def __init__(self,
        state: ScalarField,
        g_factor: ScalarField,
        GC_field: VectorField,
        opts: Options
    ):
        # TODO: move to tests
        if state.dimension == 2:
            assert state._impl.data.shape[0] == GC_field._impl._data_0.shape[0] + 1
            assert state._impl.data.shape[1] == GC_field._impl._data_0.shape[1]
            assert GC_field._impl._data_0.shape[0] == GC_field._impl._data_1.shape[0] - 1
            assert GC_field._impl._data_0.shape[1] == GC_field._impl._data_1.shape[1] + 1
        # TODO: assert G.data.shape == state.data.shape (but halo...)
        # TODO assert halo

        self.G = g_factor

        self.curr = state
        self.prev = ScalarField.clone(state)

        self.GC_phys = GC_field
        self.GC_prev = VectorField.clone(GC_field)
        self.GC_curr = VectorField.clone(GC_field)

        self.halo = state.halo

        if opts.fct:
            self.psi_min = ScalarField.clone(state)
            self.psi_max = ScalarField.clone(state)
            self.beta_up = ScalarField.clone(state)
            self.beta_dn = ScalarField.clone(state)
        else:
            self.psi_min = None
            self.psi_max = None
            self.beta_up = None
            self.beta_dn = None
