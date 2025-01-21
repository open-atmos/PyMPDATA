"""
MPDATA variants, iterations, data-type and jit-flags settings
"""

import numpy as np
from pystrict import strict


class HashableDict(dict):
    """serialization enabler"""

    def __hash__(self):
        return hash(tuple(sorted(self.items())))


@strict
class Options:
    """representation of MPDATA algorithm variant choice, for an overview of
    MPDATA options implemented in PyMPDATA, see
    [Olesik et al. 2020](https://doi.org/10.5194/gmd-15-3879-2022);
    equipped with meaningful `__str__` `__hash__`, `__eq__`.
    """

    def __init__(
        self,
        *,
        n_iters: int = 2,
        infinite_gauge: bool = False,
        divergent_flow: bool = False,
        nonoscillatory: bool = False,
        third_order_terms: bool = False,
        DPDC: bool = False,  # pylint: disable=invalid-name
        epsilon: float = 1e-15,
        non_zero_mu_coeff: bool = False,
        dimensionally_split: bool = False,
        dtype: [np.float32, np.float64] = np.float64
    ):
        self._values = HashableDict(
            {
                "n_iters": n_iters,
                "infinite_gauge": infinite_gauge,
                "epsilon": epsilon,
                "divergent_flow": divergent_flow,
                "nonoscillatory": nonoscillatory,
                "third_order_terms": third_order_terms,
                "non_zero_mu_coeff": non_zero_mu_coeff,
                "dimensionally_split": dimensionally_split,
                "dtype": dtype,
                "DPDC": DPDC,
            }
        )

        if (
            any(
                (
                    infinite_gauge,
                    divergent_flow,
                    nonoscillatory,
                    third_order_terms,
                    DPDC,
                )
            )
            and n_iters < 2
        ):
            raise ValueError()
        if n_iters < 1:
            raise ValueError()

    @property
    def dtype(self):
        """data type (e.g., np.float64)"""
        return self._values["dtype"]

    @property
    def n_iters(self) -> int:
        """Number of corrective iterations in the MPDATA algorithm + 1
        e.g. (1: upwind, 2: upwind + one corrective iteration, ...).
        Bigger values mean smaller error, but more computational cost.
        It does not change the order of the method.
        The order of the method depends on the variant of antidiffusive
        velocity used, see for example `third_order_terms` option.
        Note: not to confuse with n_steps in the Stepper."""
        return self._values["n_iters"]

    @property
    def infinite_gauge(self) -> bool:
        """flag enabling the infinite-gauge option, see e.g.:
        [Margolin & Shashkov, 2006](https://doi.org/10.1002/fld.1070),
        [Smolarkiewicz & Clark, 1986](https://doi.org/10.1016/0021-9991(86)90270-6)
        """
        return self._values["infinite_gauge"]

    @property
    def epsilon(self) -> float:
        """value of constant used to prevent from divisions by zero
        in statements such as (a - b)/(a + b + eps)"""
        return self._values["epsilon"]

    @property
    def divergent_flow(self) -> bool:
        """flag enabling the divergent-flow option, see e.g.:
        [Smolarkiewicz, 1984](https://doi.org/10.1016/0021-9991(84)90121-9),
        [Margolin & Smolarkiewicz, 1998](https://doi.org/10.1137/S106482759324700X)
        """
        return self._values["divergent_flow"]

    @property
    def nonoscillatory(self) -> bool:
        """flag enabling the nonoscillatory option, see
        [Smolarkiewicz & Grabowski 1990](https://doi.org/10.1016/0021-9991(90)90105-A)
        """
        return self._values["nonoscillatory"]

    @property
    def third_order_terms(self) -> bool:
        """flag enabling the third-order-terms option, see
        [Margolin & Smolarkiewicz 1998](https://doi.org/10.1137/S106482759324700X)"""
        return self._values["third_order_terms"]

    @property
    def DPDC(self) -> bool:  # pylint: disable=invalid-name
        """flag enabling the double-pass donor-cell option, see:
        [Beason & Margolin, 1988](https://osti.gov/biblio/7049237),
        [Margolin & Shashkov, 2006](https://doi.org/10.1002/fld.1070),
        [Margolin & Smolarkiewicz, 1998](https://doi.org/10.1137/S106482759324700X)
        """
        return self._values["DPDC"]

    @property
    def non_zero_mu_coeff(self) -> bool:
        """flag enabling handling of Fickian diffusion term"""
        return self._values["non_zero_mu_coeff"]

    @property
    def dimensionally_split(self) -> bool:
        """flag disabling cross-dimensional terms in antidiffusive velocities"""
        return self._values["dimensionally_split"]

    def __str__(self):
        return str(self._values)

    def __hash__(self):
        value = hash(self._values) + hash(self.jit_flags)
        return value

    def __eq__(self, other):
        return other.__hash__() == self.__hash__()

    @property
    def n_halo(self) -> int:
        """Halo extent for a given options set.
        The halo extent is the number of 'ghost layers' that need to be added
        to the outside of the domain to ensure that the MPDATA stencil operations can be
        applied to the edges of the domain.
        It is similar to
        [array padding](https://numpy.org/doc/stable/reference/generated/numpy.pad.html).
        The halo extent is determined by the options set."""
        if self.divergent_flow or self.nonoscillatory or self.third_order_terms:
            return 2
        return 1

    @property
    def jit_flags(self) -> HashableDict:
        """options passed [to numba.njit()](
        https://numba.pydata.org/numba-doc/dev/user/jit.html#compilation-options)"""
        return HashableDict(
            {
                "fastmath": True,
                "error_model": "numpy",
                "boundscheck": False,
            }
        )
