from .formulae.antidiff import make_antidiff
from .formulae.flux import make_fluxes
from .formulae.upwind import make_upwind
from .formulae.laplacian import make_laplacian
from .formulae import fct_utils as fct
from .options import Options


class MPDATAFormulae:
    def __init__(self, opts: Options):
        # TODO: assert for numba decorators? (depending on value of utils.DEBUG)
        self.antidiff = make_antidiff(opts)
        self.flux = make_fluxes(opts)
        self.upwind = make_upwind(opts)
        self.fct_GC_mono = fct.fct_GC_mono
        self.laplacian = make_laplacian(opts)

