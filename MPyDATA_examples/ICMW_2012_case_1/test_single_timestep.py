from MPyDATA.mpdata_factory import MPDATAFactory
from MPyDATA.options import Options
import numpy as np


def test_single_timestep():
    # Arrange
    grid = (75, 75)
    size = (1500, 1500)
    dt = 1
    w_max = .6

    def stream_function(xX, zZ):
        X = size[0]
        return - w_max * X / np.pi * np.sin(np.pi * zZ) * np.cos(2 * np.pi * xX)

    rhod_of = lambda z: 1 - z * 1e-4
    rhod = np.repeat(
        rhod_of(
            (np.arange(grid[1]) + 1 / 2) / grid[1]
        ).reshape((1, grid[1])),
        grid[0],
        axis=0
    )

    GC, eulerian_fields = MPDATAFactory.kinematic_2d(
        grid=grid, size=size, dt=dt,
        stream_function=stream_function,
        field_values={'th': 300, 'qv': .001},
        g_factor=rhod,
        opts = Options(n_iters=1)
    )

    # Plot

    # Act
    eulerian_fields.step()