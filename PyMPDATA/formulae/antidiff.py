import numba
import numpy as np
from ..arakawa_c.indexers import indexers
from ..arakawa_c.enumerations import MAX_DIM_NUM


def make_antidiff(non_unit_g_factor, options, traversals, last_pass=False):
    if options.n_iters <= 1:
        @numba.njit(**options.jit_flags)
        def apply(_1, _2, _3, _4, _5, _6, _7):
            return
    else:
        idx = indexers[traversals.n_dims]
        apply_vector = traversals.apply_vector()

        formulae_antidiff = tuple([
            __make_antidiff(idx.atv[i], idx.at[i],
                            non_unit_g_factor=non_unit_g_factor,
                            options=options,
                            n_dims=traversals.n_dims,
                            last_pass=last_pass)
            if idx.at[i] is not None else None
            for i in range(MAX_DIM_NUM)])

        @numba.njit(**options.jit_flags)
        def apply(GC_corr, psi, psi_bc, GC_unco, vec_bc, g_factor, g_factor_bc):
            return apply_vector(*formulae_antidiff, *GC_corr, *psi, *psi_bc, *GC_unco, *vec_bc,
                                *g_factor, *g_factor_bc)

    return apply


def __make_antidiff(atv, at, non_unit_g_factor, options, n_dims, last_pass):
    infinite_gauge = options.infinite_gauge
    divergent_flow = options.divergent_flow
    third_order_terms = options.third_order_terms
    epsilon = options.epsilon
    DPDC = options.DPDC
    dimensionally_split = options.dimensionally_split

    # TODO #225: does DPDC + other options combination make sense?

    # eq. 13 in Smolarkiewicz 1984; eq. 17a in Smolarkiewicz & Margolin 1998
    @numba.njit(**options.jit_flags)
    def A(psi):
        result = at(*psi, 1) - at(*psi, 0)
        if infinite_gauge:
            result /= 2
        else:
            result /= (at(*psi, 1) + at(*psi, 0) + epsilon)
        return result

    # eq. 13 in Smolarkiewicz 1984; eq. 17b in Smolarkiewicz & Margolin 1998
    @numba.njit(**options.jit_flags)
    def B(psi):
        result = (
                at(*psi, 1, 1) + at(*psi, 0, 1) -
                at(*psi, 1, -1) - at(*psi, 0, -1)
        )
        if infinite_gauge:
            result /= 4
        else:
            result /= (
                    at(*psi, 1, 1) + at(*psi, 0, 1) +
                    at(*psi, 1, -1) + at(*psi, 0, -1) +
                    epsilon
            )
        return result

    @numba.njit(**options.jit_flags)
    def antidiff_basic(psi, GC, _):
        # eq. 13 in Smolarkiewicz 1984
        tmp = A(psi)
        result = (np.abs(atv(*GC, .5)) - atv(*GC, +.5) ** 2) * tmp
        if DPDC and last_pass:  # TODO #225 n_dims > 1
            a = (1 / (1 - np.abs(tmp)))
            b = - (tmp*a)/(1 - tmp**2)
            result = result * (result * b + a) 
        if n_dims == 1 or dimensionally_split:
            return result
        else:
            result -= (
                0.5 * atv(*GC, .5) *
                0.25 * (atv(*GC, 1., +.5) + atv(*GC, 0., +.5) + atv(*GC, 1., -.5) + atv(*GC, 0., -.5)) *
                B(psi)
            )
        return result

    @numba.njit(**options.jit_flags)
    def antidiff_variants(psi, GC, G):
        # eq. 13 in Smolarkiewicz 1984
        result = antidiff_basic(psi, GC, G)

        G_bar = (at(*G, 1) + at(*G, 0)) / 2 if non_unit_g_factor else 1

        # third-order terms
        if third_order_terms:
            # assert psi.dimension < 3  # TODO #96
            tmp = (
              3 * atv(*GC, .5) * np.abs(atv(*GC, .5)) / G_bar
              - 2 * atv(*GC, .5) ** 3 / G_bar ** 2
              - atv(*GC, .5)
            ) / 6

            tmp *= 2 * (at(*psi, 2) - at(*psi, 1) - at(*psi, 0) + at(*psi, -1))

            if infinite_gauge:
                tmp /= (1 + 1 + 1 + 1)
            else:
                tmp /= at(*psi, 2) + at(*psi, 1) + at(*psi, 0) + at(*psi, -1) + epsilon

            result += tmp

            if n_dims > 1:
                GC1_bar = (
                                  atv(*GC, 1, .5) +
                                  atv(*GC, 0, .5) +
                                  atv(*GC, 1, -.5) +
                                  atv(*GC, 0, -.5)
                          ) / 4
                tmp = GC1_bar / (2 * G_bar) * (
                        np.abs(atv(*GC, .5)) - 2 * atv(*GC, .5) ** 2 / G_bar
                )

                tmp *= 2 * (at(*psi, 1, 1) - at(*psi, 0, 1) - at(*psi, 1, -1) + at(*psi, 0, -1))

                if infinite_gauge:
                    tmp /= (1 + 1 + 1 + 1)
                else:
                    tmp /= (at(*psi, 1, 1) + at(*psi, 0, 1) + at(*psi, 1, -1) + at(*psi, 0, -1))

                result += tmp

        # divergent flow option
        # eq.(30) in Smolarkiewicz_and_Margolin_1998
        if divergent_flow:
            # assert psi.dimension == 1  # TODO #94
            tmp = -.25 * atv(*GC, .5) * (atv(*GC, 1.5) - atv(*GC, -.5))
            if non_unit_g_factor:
                tmp /= G_bar
            if infinite_gauge:
                tmp *= .5 * at(*psi, 1) + at(*psi, 0)

            result += tmp
        return result
    return antidiff_variants if divergent_flow or third_order_terms else antidiff_basic
