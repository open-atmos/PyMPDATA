import numpy as np
from PyMPDATA import Solver, Stepper, ScalarField, VectorField
from PyMPDATA.boundary_conditions import Extrapolated, Constant
from PyMPDATA.impl.enumerations import INNER, OUTER
from .arakawa_c import arakawa_c


class MPDATA:
    # pylint: disable=too-few-public-methods
    def __init__(self, nz, dt, qv_of_zZ_at_t0, g_factor_of_zZ, nr, options, activation_bc):
        self.t = 0
        self.dt = dt
        self.fields = ('qv', 'ql')

        self.options = options

        self._solvers = {}
        for k in self.fields:
            grid = (nz, nr) if nr > 1 and k == 'ql' else (nz,)

            bcs_extrapol = tuple(
                Extrapolated(dim=d)
                for d in ((OUTER, INNER) if k == 'ql' and nr > 1 else (INNER,))
            )
            bcs_zero = tuple(
                Extrapolated(dim=d)
                for d in ((OUTER, INNER) if k == 'ql' and nr > 1 else (INNER,))
            )

            stepper = Stepper(options=self.options, n_dims=len(grid), non_unit_g_factor=True)

            data = g_factor_of_zZ(arakawa_c.z_scalar_coord(grid))
            if nr > 1 and k == 'ql':
                data = np.repeat(data.reshape(-1, 1), nr, axis=1).squeeze()
            g_factor = ScalarField(
                data=data,
                halo=self.options.n_halo,
                boundary_conditions=bcs_extrapol
            )

            if nr == 1 or k == 'qv':
                data = (np.zeros(nz + 1),)
            else:
                data = (np.zeros((nz + 1, nr)), np.zeros((nz, nr + 1)))
            advector = VectorField(
                data=data,
                halo=self.options.n_halo,
                boundary_conditions=bcs_zero
            )
            if k == 'qv':
                data = qv_of_zZ_at_t0(arakawa_c.z_scalar_coord(grid))
                bcs = (Constant(value=data[0]),)
            else:
                data = np.zeros(grid)
                if nr == 1:
                    bcs = (Constant(value=0),)
                else:
                    bcs = (
                        Constant(value=0),
                        activation_bc
                    )
            advectee = ScalarField(
                data=data,
                halo=self.options.n_halo,
                boundary_conditions=bcs
            )
            self._solvers[k] = Solver(
                stepper=stepper, advectee=advectee, advector=advector, g_factor=g_factor
            )

    def __getitem__(self, k):
        return self._solvers[k]
