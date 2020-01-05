from .formulae.antidiff import make_antidiff
from .formulae.flux import make_fluxes
from .formulae.upwind import make_upwind
from .formulae.laplacian import make_laplacian
from .formulae.fct_utils import make_GC_mono


# TODO: rename (algorithms)
class Options:
    def __init__(self,
                 nug: bool = False,  # non-unit g_factor
                 iga: bool = False,  # infinite gauge
                 fct: bool = False,  # flux-corrected transport
                 dfl: bool = False,  # divergent flow
                 tot: bool = False,  # third-order terms
                 nzm: bool = False,  # non-zero mu
                 eps: float = 1e-8,
                 ):
        self._nug = nug
        self._iga = iga
        self._fct = fct
        self._dfl = dfl
        self._tot = tot
        self._eps = eps
        self._nzm = nzm

        # TODO: assert for numba decorators? (depending on value of utils.DEBUG)
        self._formulae = {
            "antidiff": make_antidiff(self),
            "flux": make_fluxes(self),
            "upwind": make_upwind(self),
            "GC_mono": make_GC_mono(),
            "laplacian": make_laplacian(self)
        }

    def clone(self):
        return self

    @property
    def formulae(self):
        return self._formulae

    @property
    def nug(self) -> bool:
        return self._nug

    @property
    def iga(self) -> bool:
        return self._iga

    @property
    def fct(self) -> bool:
        return self._fct

    @property
    def dfl(self) -> bool:
        return self._dfl

    @property
    def tot(self) -> bool:
        return self._tot

    @property
    def eps(self) -> float:
        return self._eps

    # for definition of mu (mesh Fourier number),
    # see eq. 20 in Sousa 2009, doi:10.1002/fld.1984
    @property
    def nzm(self) -> float:
        return self._nzm
