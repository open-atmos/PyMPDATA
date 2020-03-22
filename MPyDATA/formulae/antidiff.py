import numba
import numpy as np
from .jit_flags import jit_flags


def make_antidiff(atv, at, non_unit_g_factor, options, n_dims, axis):
    infinite_gauge = options.infinite_gauge
    divergent_flow = options.divergent_flow
    third_order_terms = options.third_order_terms
    epsilon = options.epsilon

    # eq. 13 in Smolarkiewicz 1984; eq. 17a in Smolarkiewicz & Margolin 1998
    @numba.njit(**jit_flags)
    def A(psi):
        result = at(*psi, 1, 0) - at(*psi, 0, 0)
        if infinite_gauge:
            result /= 2
        else:
            result /= (at(*psi, 1, 0) + at(*psi, 0, 0) + epsilon)
        return result

    # eq. 13 in Smolarkiewicz 1984; eq. 17b in Smolarkiewicz & Margolin 1998
    @numba.njit(**jit_flags)
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

    @numba.njit(**jit_flags)
    def antidiff(psi, GC, G):
        # eq. 13 in Smolarkiewicz 1984
        result = (np.abs(atv(*GC, .5, 0)) - atv(*GC, +.5, 0) ** 2) * A(psi)
        if axis == 0 and n_dims == 1:
            return result
        else:
            result -= (
                0.5 * atv(*GC, .5, 0) *
                0.25 * (atv(*GC, 1, +.5) + atv(*GC, 0, +.5) + atv(*GC, 1, -.5) + atv(*GC, 0, -.5)) *
                B(psi)
            )

        if not third_order_terms and not divergent_flow:
            return result

        # G_bar = 1
        # if non_unit_g_factor:
        #     G_bar = (at(*G, 1, 0) + at(*G, 0, 0)) / 2

        #
        #     # third-order terms
        #     if third_order_terms:
        #         assert psi.dimension < 3  # TODO
        #         tmp = (
        #                       3 * GC.at(.5, 0) * np.abs(GC.at(.5, 0)) / G_bar
        #                       - 2 * GC.at(.5, 0) ** 3 / G_bar ** 2
        #                       - GC.at(.5, 0)
        #               ) / 6
        #
        #         tmp *= 2 * (psi.at(2, 0) - psi.at(1, 0) - psi.at(0, 0) + psi.at(-1, 0))
        #
        #         if infinite_gauge:
        #             tmp /= (1 + 1 + 1 + 1)
        #         else:
        #             tmp /= (psi.at(2, 0) + psi.at(1, 0) + psi.at(0, 0) + psi.at(-1, 0))
        #
        #         result += tmp
        #
        #         if psi.dimension > 1:
        #             GC1_bar = (
        #                               GC.at(1, .5) +
        #                               GC.at(0, .5) +
        #                               GC.at(1, -.5) +
        #                               GC.at(0, -.5)
        #                       ) / 4
        #             tmp = GC1_bar / (2 * G_bar) * (
        #                     np.abs(GC.at(.5, 0)) - 2 * GC.at(.5, 0) ** 2 / G_bar
        #             )
        #
        #             tmp *= 2 * (psi.at(1, 1) - psi.at(0, 1) - psi.at(1, -1) + psi.at(0, -1))
        #
        #             if infinite_gauge:
        #                 tmp /= (1 + 1 + 1 + 1)
        #             else:
        #                 tmp /= (psi.at(1, 1) + psi.at(0, 1) + psi.at(1, -1) + psi.at(0, -1))
        #
        #             result += tmp
        #
        #     # divergent flow option
        #     # eq.(30) in Smolarkiewicz_and_Margolin_1998
        #     if divergent_flow:
        #         assert psi.dimension == 1  # TODO!
        #         tmp = -.5 * GC.at(.5, 0) * (GC.at(1.5, 0) - GC.at(-.5, 0))
        #         if non_unit_g_factor:
        #             tmp /= 2 * G_bar
        #         if infinite_gauge:
        #             tmp *= .5 * (psi.at(1, 0) + psi.at(0, 0))
        #
        #         result += tmp
        return result
    return antidiff
