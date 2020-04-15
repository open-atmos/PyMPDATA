from MPyDATA.mpdata_factory import MPDATAFactory
from MPyDATA.options import Options
import numpy as np
import pytest


# TODO: work in progress
@pytest.mark.parametrize(
    "options", [
        Options(n_iters=1),
        Options(n_iters=2),
        Options(n_iters=2, flux_corrected_transport=True),
        Options(n_iters=3, flux_corrected_transport=True),
        Options(n_iters=2, flux_corrected_transport=True, infinite_gauge=True),
        # TODO
        # Options(nug=True, fct=True, iga=True, tot=True),
        # Options(nug=True, fct=False, iga=True),
        # Options(nug=True, fct=False, tot=True),
        # Options(nug=True, fct=False, iga=True, tot=True)
    ]
)
def test_single_timestep(options):
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

    GC, mpdatas = MPDATAFactory.stream_function_2d(
        grid=grid, size=size, dt=dt,
        stream_function=stream_function,
        field_values={'th': np.full(grid, 300), 'qv': np.full(grid, .001)},
        g_factor=rhod,
        options=options
    )

    # Plot

    # Act
    for mpdata in mpdatas.values():
        mpdata.step(1)

    # Assert
    for k, v in mpdatas.items():
        assert np.isfinite(v.curr.get()).all()
