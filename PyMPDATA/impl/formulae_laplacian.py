"""logic for handling the Fickian term by modifying physical velocity"""

import numba

from ..impl.enumerations import MAX_DIM_NUM
from ..impl.traversals import Traversals
from ..options import Options


def make_laplacian(non_unit_g_factor: bool, options: Options, traversals: Traversals):
    """returns njit-ted function for use with given traversals"""
    if not options.non_zero_mu_coeff:

        @numba.njit(**options.jit_flags)
        def apply(_1, _2, _3):
            return

    else:
        idx = traversals.indexers[traversals.n_dims]
        apply_vector = traversals.apply_vector()

        formulae_laplacian = tuple(
            (
                __make_laplacian(
                    options.jit_flags, idx.ats[i], options.epsilon, non_unit_g_factor
                )
                if idx.ats[i] is not None
                else None
            )
            for i in range(MAX_DIM_NUM)
        )

        @numba.njit(**options.jit_flags)
        def apply(traversals_data, advector, advectee):
            null_vecfield, null_vecfield_bc = traversals_data.null_vector_field
            null_scalarfield, null_scalarfield_bc = traversals_data.null_scalar_field
            return apply_vector(
                *formulae_laplacian,
                *advector.field,
                *advectee.field,
                advectee.bc,
                *null_vecfield,
                null_vecfield_bc,
                *null_scalarfield,
                null_scalarfield_bc,
                traversals_data.buffer,
            )

    return apply


def make_heterogeneous_laplacian(
    non_unit_g_factor: bool, options: Options, traversals: Traversals
):
    """returns njit-ted function for heterogeneous diffusion with spatially varying diffusivity

    Note: heterogeneous diffusion is only supported when options.non_zero_mu_coeff is enabled
    """
    if not options.non_zero_mu_coeff:
        raise NotImplementedError(
            "Heterogeneous diffusion requires options.non_zero_mu_coeff to be enabled"
        )
    elif not options.heterogeneous_diffusion:
        raise NotImplementedError(
            "Heterogeneous diffusion requires options.heterogeneous_diffusion to be enabled"
        )

    else:
        idx = traversals.indexers[traversals.n_dims]
        apply_vector = traversals.apply_vector()

        formulae_heterogeneous = tuple(
            (
                __make_heterogeneous_laplacian(
                    options.jit_flags, idx.ats[i], options.epsilon, non_unit_g_factor
                )
                if idx.ats[i] is not None
                else None
            )
            for i in range(MAX_DIM_NUM)
        )

        @numba.njit(**options.jit_flags)
        def apply(traversals_data, advector, advectee, diffusivity_field):
            null_vecfield, null_vecfield_bc = traversals_data.null_vector_field
            return apply_vector(
                *formulae_heterogeneous,
                *advector.field,
                *advectee.field,
                advectee.bc,
                *null_vecfield,
                null_vecfield_bc,
                *diffusivity_field.field,
                diffusivity_field.bc,
                traversals_data.buffer,
            )

    return apply


def __make_laplacian(jit_flags, ats, epsilon, non_unit_g_factor):
    if non_unit_g_factor:
        raise NotImplementedError()

    @numba.njit(**jit_flags)
    def fun(advectee, _, __):
        return (
            -2
            * (ats(*advectee, 1) - ats(*advectee, 0))
            / (ats(*advectee, 1) + ats(*advectee, 0) + epsilon)
        )

    return fun


def __make_heterogeneous_laplacian(jit_flags, ats, epsilon, non_unit_g_factor):
    """Create heterogeneous Laplacian function that matches MPDATA's one-sided gradient pattern

    Note: Diffusivity values are expected to be non-negative. Negative values will cause
    an assertion error. Zero values are handled by setting a minimum threshold (epsilon).
    """
    if non_unit_g_factor:
        raise NotImplementedError()

    @numba.njit(**jit_flags)
    def fun(advectee, _, diffusivity):
        # Get concentration values (matching regular laplacian pattern)
        c_curr = ats(*advectee, 0)
        c_right = ats(*advectee, 1)

        # Get diffusivity values
        D_curr = ats(*diffusivity, 0)
        D_right = ats(*diffusivity, 1)

        # Input validation for diffusivity values
        assert D_curr >= 0.0, "Diffusivity values must be non-negative"
        assert D_right >= 0.0, "Diffusivity values must be non-negative"

        # Handle near-zero diffusivity by setting minimum threshold
        D_curr = max(D_curr, epsilon)
        D_right = max(D_right, epsilon)

        # Match the exact MPDATA pattern but with diffusivity weighting
        # Regular: -2 * (c[i+1] - c[i]) / (c[i+1] + c[i] + epsilon)
        # Heterogeneous: weight by diffusivity at face
        D_face = 0.5 * (D_curr + D_right)

        return -2 * D_face * (c_right - c_curr) / (c_right + c_curr + epsilon)

    return fun
