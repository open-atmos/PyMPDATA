"""antidiffusive velocity formulae incl. divergent-flow,
third-order-terms, DPDC and partially also infinite-gauge logic"""

import numba
import numpy as np

from PyMPDATA.impl.enumerations import MAX_DIM_NUM


def make_antidiff(non_unit_g_factor, options, traversals, last_pass=False):
    """returns njit-ted function for use with given traversals"""
    idx = traversals.indexers[traversals.n_dims]
    apply_vector = traversals.apply_vector()

    formulae_antidiff = tuple(
        (
            __make_antidiff(
                atv=idx.atv[i],
                ats=idx.ats[i],
                non_unit_g_factor=non_unit_g_factor,
                options=options,
                n_dims=traversals.n_dims,
                last_pass=last_pass,
            )
            if idx.ats[i] is not None
            else None
        )
        for i in range(MAX_DIM_NUM)
    )

    @numba.njit(**options.jit_flags)
    def apply(traversals_data, g_c_corr, psi, g_c_unco, g_factor):
        return apply_vector(
            *formulae_antidiff,
            *g_c_corr.field,
            *psi.field,
            psi.bc,
            *g_c_unco.field,
            g_c_unco.bc,
            *g_factor.field,
            g_factor.bc,
            traversals_data.buffer
        )

    return apply


# pylint: disable=too-many-locals
def __make_antidiff(*, atv, ats, non_unit_g_factor, options, n_dims, last_pass):
    infinite_gauge = options.infinite_gauge
    divergent_flow = options.divergent_flow
    third_order_terms = options.third_order_terms
    epsilon = options.epsilon
    dpdc = options.DPDC
    dimensionally_split = options.dimensionally_split

    @numba.njit(**options.jit_flags)
    def a_term(psi):
        """eq. 13 in [Smolarkiewicz 1984](https://doi.org/10.1016/0021-9991(84)90121-9);
        eq. 17a in [Smolarkiewicz & Margolin 1998](https://doi.org/10.1006/jcph.1998.5901)
        """
        result = ats(*psi, 1) - ats(*psi, 0)
        if infinite_gauge:
            return result / 2
        return result / (ats(*psi, 1) + ats(*psi, 0) + epsilon)

    @numba.njit(**options.jit_flags)
    def b_term(psi):
        """eq. 13 in [Smolarkiewicz 1984](https://doi.org/10.1016/0021-9991(84)90121-9);
        eq. 17b in [Smolarkiewicz & Margolin 1998](https://doi.org/10.1006/jcph.1998.5901)
        """
        result = ats(*psi, 1, 1) + ats(*psi, 0, 1) - ats(*psi, 1, -1) - ats(*psi, 0, -1)
        if infinite_gauge:
            return result / 4

        return result / (
            ats(*psi, 1, 1)
            + ats(*psi, 0, 1)
            + ats(*psi, 1, -1)
            + ats(*psi, 0, -1)
            + epsilon
        )

    @numba.njit(**options.jit_flags)
    def antidiff_basic(psi, g_c, _):
        """eq. 13 in [Smolarkiewicz 1984](https://doi.org/10.1016/0021-9991(84)90121-9)"""
        tmp = a_term(psi)
        result = (np.abs(atv(*g_c, 0.5)) - atv(*g_c, +0.5) ** 2) * tmp
        if dpdc and last_pass:
            val_1 = 1 / (1 - np.abs(tmp))
            val_2 = -(tmp * val_1) / (1 - tmp**2)
            result = result * (result * val_2 + val_1)
        if n_dims == 1 or dimensionally_split:
            return result
        return result - (
            0.5
            * atv(*g_c, 0.5)
            * 0.25
            * (
                atv(*g_c, 1.0, +0.5)
                + atv(*g_c, 0.0, +0.5)
                + atv(*g_c, 1.0, -0.5)
                + atv(*g_c, 0.0, -0.5)
            )
            * b_term(psi)
        )

    @numba.njit(**options.jit_flags)
    def antidiff_variants(psi, g_c, g_factor):
        """eq. 13 in [Smolarkiewicz 1984](https://doi.org/10.1016/0021-9991(84)90121-9)"""
        result = antidiff_basic(psi, g_c, g_factor)

        g_bar = (ats(*g_factor, 1) + ats(*g_factor, 0)) / 2 if non_unit_g_factor else 1

        # third-order terms
        if third_order_terms:
            # assert psi.dimension < 3  # TODO #96
            tmp = (
                3 * atv(*g_c, 0.5) * np.abs(atv(*g_c, 0.5)) / g_bar
                - 2 * atv(*g_c, 0.5) ** 3 / g_bar**2
                - atv(*g_c, 0.5)
            ) / 6

            tmp *= 2 * (ats(*psi, 2) - ats(*psi, 1) - ats(*psi, 0) + ats(*psi, -1))

            if infinite_gauge:
                tmp /= 1 + 1 + 1 + 1
            else:
                tmp /= (
                    ats(*psi, 2) + ats(*psi, 1) + ats(*psi, 0) + ats(*psi, -1) + epsilon
                )

            result += tmp

            if n_dims > 1:
                tmp = (
                    atv(*g_c, 1, 0.5)
                    + atv(*g_c, 0, 0.5)
                    + atv(*g_c, 1, -0.5)
                    + atv(*g_c, 0, -0.5)
                ) / 4
                tmp = (
                    tmp
                    / (2 * g_bar)
                    * (np.abs(atv(*g_c, 0.5)) - 2 * atv(*g_c, 0.5) ** 2 / g_bar)
                )

                tmp *= 2 * (
                    ats(*psi, 1, 1)
                    - ats(*psi, 0, 1)
                    - ats(*psi, 1, -1)
                    + ats(*psi, 0, -1)
                )

                if infinite_gauge:
                    tmp /= 1 + 1 + 1 + 1
                else:
                    tmp /= (
                        ats(*psi, 1, 1)
                        + ats(*psi, 0, 1)
                        + ats(*psi, 1, -1)
                        + ats(*psi, 0, -1)
                    )

                result += tmp

        # divergent flow option
        # eq.(30) in [Smolarkiewicz and Margolin 1998](https://doi.org/10.1006/jcph.1998.5901)
        if divergent_flow:
            # assert psi.dimension == 1
            tmp = -0.25 * atv(*g_c, 0.5) * (atv(*g_c, 1.5) - atv(*g_c, -0.5))
            if non_unit_g_factor:
                tmp /= g_bar
            if infinite_gauge:
                tmp *= 0.5 * ats(*psi, 1) + ats(*psi, 0)

            result += tmp
        return result

    return antidiff_variants if divergent_flow or third_order_terms else antidiff_basic
