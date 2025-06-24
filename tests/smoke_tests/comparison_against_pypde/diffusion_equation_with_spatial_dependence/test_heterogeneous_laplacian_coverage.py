"""
Comprehensive tests for heterogeneous Laplacian functionality to improve code coverage.
Tests error conditions, edge cases, and the actual computation logic.
"""

import pytest
import numpy as np

from PyMPDATA import Options, ScalarField, Solver, Stepper, VectorField
from PyMPDATA.boundary_conditions import Constant
from PyMPDATA.impl.formulae_laplacian import (
    make_heterogeneous_laplacian,
    __make_heterogeneous_laplacian as _make_heterogeneous_laplacian,
)
from PyMPDATA.impl.traversals import Traversals
from PyMPDATA.impl.indexers import make_indexers
from PyMPDATA.impl.enumerations import META_AND_DATA_META, MAX_DIM_NUM


class TestHeterogeneousLaplacianErrorConditions:
    """Test error conditions that should raise NotImplementedError"""

    def test_make_heterogeneous_laplacian_without_non_zero_mu_coeff(self):
        """Test that make_heterogeneous_laplacian raises error when non_zero_mu_coeff is False"""
        options = Options(non_zero_mu_coeff=False, heterogeneous_diffusion=True)
        traversals = Traversals(
            grid=(10,),
            halo=options.n_halo,
            jit_flags=options.jit_flags,
            n_threads=1,
            left_first=tuple([True] * MAX_DIM_NUM),
            buffer_size=0,
        )

        with pytest.raises(
            NotImplementedError,
            match="requires options.non_zero_mu_coeff to be enabled",
        ):
            make_heterogeneous_laplacian(False, options, traversals)

    def test_make_heterogeneous_laplacian_without_heterogeneous_diffusion(self):
        """Test that make_heterogeneous_laplacian raises error when heterogeneous_diffusion is False"""
        options = Options(non_zero_mu_coeff=True, heterogeneous_diffusion=False)
        traversals = Traversals(
            grid=(10,),
            halo=options.n_halo,
            jit_flags=options.jit_flags,
            n_threads=1,
            left_first=tuple([True] * MAX_DIM_NUM),
            buffer_size=0,
        )

        with pytest.raises(
            NotImplementedError,
            match="requires options.heterogeneous_diffusion to be enabled",
        ):
            make_heterogeneous_laplacian(False, options, traversals)

    def test_make_heterogeneous_laplacian_with_non_unit_g_factor(self):
        """Test that __make_heterogeneous_laplacian raises error when non_unit_g_factor is True"""
        options = Options(non_zero_mu_coeff=True, heterogeneous_diffusion=True)
        traversals = Traversals(
            grid=(10,),
            halo=options.n_halo,
            jit_flags=options.jit_flags,
            n_threads=1,
            left_first=tuple([True] * MAX_DIM_NUM),
            buffer_size=0,
        )

        with pytest.raises(NotImplementedError):
            make_heterogeneous_laplacian(True, options, traversals)


class TestHeterogeneousLaplacianFunctionality:
    """Test the actual heterogeneous diffusion functionality"""

    def test_heterogeneous_diffusion_1d_basic(self):
        """Test basic 1D heterogeneous diffusion with varying diffusivity"""
        grid_size = 10
        dx = 1.0

        # Set up initial condition - Gaussian pulse
        x = np.linspace(0, (grid_size - 1) * dx, grid_size)
        x_center = x[grid_size // 2]
        sigma = 1.0
        c0 = np.exp(-0.5 * ((x - x_center) / sigma) ** 2)

        # Set up spatially varying diffusivity - higher in the center
        D_field = 0.1 + 0.9 * np.exp(-0.5 * ((x - x_center) / (2 * sigma)) ** 2)

        # Create options and fields
        options = Options(
            n_iters=3, non_zero_mu_coeff=True, heterogeneous_diffusion=True
        )

        advectee = ScalarField(
            data=c0, halo=options.n_halo, boundary_conditions=(Constant(0.0),)
        )
        advector = VectorField(
            data=(np.zeros(grid_size + 1),),
            halo=options.n_halo,
            boundary_conditions=(Constant(0.0),),
        )
        diffusivity_field = ScalarField(
            data=D_field, halo=options.n_halo, boundary_conditions=(Constant(0.1),)
        )

        # Create solver
        stepper = Stepper(options=options, grid=(grid_size,))
        solver = Solver(
            stepper=stepper,
            advectee=advectee,
            advector=advector,
            diffusivity_field=diffusivity_field,
        )

        # Store initial state
        initial_sum = solver.advectee.get().sum()
        initial_max = solver.advectee.get().max()

        # Advance one step
        solver.advance(n_steps=1, mu_coeff=(0.01,))

        # Check results
        final_state = solver.advectee.get()
        final_sum = final_state.sum()
        final_max = final_state.max()

        # Mass should be conserved
        np.testing.assert_almost_equal(final_sum, initial_sum, decimal=6)
        # Maximum should decrease due to diffusion
        assert final_max < initial_max
        # No negative values should appear
        assert np.all(final_state >= 0)

    def test_heterogeneous_diffusion_with_zero_diffusivity(self):
        """Test handling of zero diffusivity values"""
        grid_size = 5

        # Initial condition
        c0 = np.array([0.0, 0.0, 1.0, 0.0, 0.0])

        # Diffusivity with zeros
        D_field = np.array([0.0, 0.1, 0.0, 0.1, 0.0])

        options = Options(
            n_iters=2, non_zero_mu_coeff=True, heterogeneous_diffusion=True
        )

        advectee = ScalarField(
            data=c0, halo=options.n_halo, boundary_conditions=(Constant(0.0),)
        )
        advector = VectorField(
            data=(np.zeros(grid_size + 1),),
            halo=options.n_halo,
            boundary_conditions=(Constant(0.0),),
        )
        diffusivity_field = ScalarField(
            data=D_field, halo=options.n_halo, boundary_conditions=(Constant(0.0),)
        )

        stepper = Stepper(options=options, grid=(grid_size,))
        solver = Solver(
            stepper=stepper,
            advectee=advectee,
            advector=advector,
            diffusivity_field=diffusivity_field,
        )

        # Should not crash with zero diffusivity
        solver.advance(n_steps=1, mu_coeff=(0.01,))

        # Mass should still be conserved
        final_state = solver.advectee.get()
        np.testing.assert_almost_equal(final_state.sum(), c0.sum(), decimal=6)

    def test_heterogeneous_diffusion_2d_basic(self):
        """Test 2D heterogeneous diffusion"""
        grid_shape = (5, 5)

        # Initial condition - point source in center
        c0 = np.zeros(grid_shape)
        c0[2, 2] = 1.0

        # Spatially varying diffusivity
        D_field = np.ones(grid_shape) * 0.1
        D_field[2, 2] = 0.5  # Higher diffusivity at center

        options = Options(
            n_iters=3, non_zero_mu_coeff=True, heterogeneous_diffusion=True
        )

        boundary_conditions = tuple([Constant(0.0)] * 2)

        advectee = ScalarField(
            data=c0, halo=options.n_halo, boundary_conditions=boundary_conditions
        )
        advector = VectorField(
            data=(
                np.zeros((grid_shape[0] + 1, grid_shape[1])),
                np.zeros((grid_shape[0], grid_shape[1] + 1)),
            ),
            halo=options.n_halo,
            boundary_conditions=boundary_conditions,
        )
        diffusivity_field = ScalarField(
            data=D_field, halo=options.n_halo, boundary_conditions=boundary_conditions
        )

        stepper = Stepper(options=options, grid=grid_shape)
        solver = Solver(
            stepper=stepper,
            advectee=advectee,
            advector=advector,
            diffusivity_field=diffusivity_field,
        )

        initial_sum = solver.advectee.get().sum()

        # Advance multiple steps
        solver.advance(n_steps=2, mu_coeff=(0.01, 0.01))

        final_state = solver.advectee.get()
        final_sum = final_state.sum()

        # Mass conservation
        np.testing.assert_almost_equal(final_sum, initial_sum, decimal=6)
        # Diffusion should spread the mass
        assert final_state[2, 2] < 1.0
        assert np.sum(final_state > 0) > 1

    def test_heterogeneous_diffusion_high_contrast(self):
        """Test with high contrast in diffusivity values"""
        grid_size = 10

        # Initial condition
        c0 = np.zeros(grid_size)
        c0[5] = 1.0

        # High contrast diffusivity
        D_field = np.ones(grid_size) * 1e-6  # Very low diffusivity
        D_field[4:7] = 1.0  # High diffusivity region

        options = Options(
            n_iters=5, non_zero_mu_coeff=True, heterogeneous_diffusion=True
        )

        advectee = ScalarField(
            data=c0, halo=options.n_halo, boundary_conditions=(Constant(0.0),)
        )
        advector = VectorField(
            data=(np.zeros(grid_size + 1),),
            halo=options.n_halo,
            boundary_conditions=(Constant(0.0),),
        )
        diffusivity_field = ScalarField(
            data=D_field, halo=options.n_halo, boundary_conditions=(Constant(1e-6),)
        )

        stepper = Stepper(options=options, grid=(grid_size,))
        solver = Solver(
            stepper=stepper,
            advectee=advectee,
            advector=advector,
            diffusivity_field=diffusivity_field,
        )

        solver.advance(n_steps=1, mu_coeff=(0.1,))

        final_state = solver.advectee.get()

        # Should handle high contrast without numerical issues
        assert np.all(final_state >= 0)
        assert not np.any(np.isnan(final_state))
        assert not np.any(np.isinf(final_state))

    def test_heterogeneous_diffusion_mass_conservation_precision(self):
        """Test mass conservation with various diffusivity patterns"""
        grid_size = 20

        # Different diffusivity patterns to test
        x = np.linspace(0, 2 * np.pi, grid_size)
        patterns = [
            np.ones(grid_size) * 0.1,  # Uniform
            0.1 + 0.9 * np.sin(x) ** 2,  # Sinusoidal
            np.exp(-0.1 * (x - np.pi) ** 2),  # Gaussian
            np.where(x < np.pi, 0.01, 1.0),  # Step function
        ]

        for i, D_field in enumerate(patterns):
            # Initial condition - smooth profile
            c0 = np.exp(-0.5 * ((x - np.pi) / 0.5) ** 2)

            options = Options(
                n_iters=4, non_zero_mu_coeff=True, heterogeneous_diffusion=True
            )

            advectee = ScalarField(
                data=c0, halo=options.n_halo, boundary_conditions=(Constant(0.0),)
            )
            advector = VectorField(
                data=(np.zeros(grid_size + 1),),
                halo=options.n_halo,
                boundary_conditions=(Constant(0.0),),
            )
            diffusivity_field = ScalarField(
                data=D_field,
                halo=options.n_halo,
                boundary_conditions=(Constant(D_field[0]),),
            )

            stepper = Stepper(options=options, grid=(grid_size,))
            solver = Solver(
                stepper=stepper,
                advectee=advectee,
                advector=advector,
                diffusivity_field=diffusivity_field,
            )

            initial_mass = solver.advectee.get().sum()

            # Run for multiple steps
            for step in range(5):
                solver.advance(n_steps=1, mu_coeff=(0.02,))

                current_mass = solver.advectee.get().sum()

                # Mass should be conserved to high precision
                np.testing.assert_almost_equal(
                    current_mass,
                    initial_mass,
                    decimal=8,
                    err_msg=f"Mass not conserved for pattern {i} at step {step}",
                )


class TestHeterogeneousLaplacianDirectUnitTests:
    """Direct unit tests for __make_heterogeneous_laplacian function creation.

    Note: The internal computation lines (128-148) are covered through the integration
    tests in TestHeterogeneousLaplacianFunctionality, as direct testing requires
    complex Numba-compatible data structures that are difficult to mock properly.
    """

    def test_make_heterogeneous_laplacian_direct_with_non_unit_g_factor_error(self):
        """Test direct call to __make_heterogeneous_laplacian with non_unit_g_factor=True (line 123)"""
        options = Options()
        indexers = make_indexers(options.jit_flags)

        # This should raise NotImplementedError for non_unit_g_factor=True (line 123)
        with pytest.raises(NotImplementedError):
            _make_heterogeneous_laplacian(
                jit_flags=options.jit_flags,
                ats=indexers[1].ats[2],  # 1D, inner dimension
                epsilon=options.epsilon,
                non_unit_g_factor=True,
            )

    def test_make_heterogeneous_laplacian_function_creation_success(self):
        """Test that __make_heterogeneous_laplacian creates function successfully"""
        options = Options()
        indexers = make_indexers(options.jit_flags)

        # This should successfully create a function
        het_laplacian_func = _make_heterogeneous_laplacian(
            jit_flags=options.jit_flags,
            ats=indexers[1].ats[2],  # 1D, inner dimension
            epsilon=options.epsilon,
            non_unit_g_factor=False,
        )

        # Verify it's a callable function
        assert callable(het_laplacian_func)

        # Verify it has Numba compilation attributes
        assert hasattr(
            het_laplacian_func, "py_func"
        )  # Numba compiled function attribute
