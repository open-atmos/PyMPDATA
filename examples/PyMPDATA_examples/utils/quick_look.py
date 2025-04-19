"""utility routines for plotting 2D fields"""

import numpy as np
from matplotlib import colors, pyplot

from PyMPDATA import ScalarField, VectorField
from PyMPDATA.impl.field import Field


def quick_look(field: Field, plot: bool = True):
    """plots either scalar or vector field together with halo region
    rendering arrows in a staggered-grid-aware manner"""
    halo = field.halo
    grid = field.grid
    pyplot.title(f"{grid=} {halo=} class={field.__class__.__name__}")
    if isinstance(field, ScalarField):
        norm = colors.Normalize(vmin=np.amin(field.get()), vmax=np.amax(field.get()))
        pyplot.imshow(
            X=field.data.T,
            origin="lower",
            extent=(-halo, grid[0] + halo, -halo, grid[1] + halo),
            cmap="gray",
            norm=norm,
        )
        pyplot.colorbar()
    elif isinstance(field, VectorField):
        arrow_colors = ("green", "blue")
        quiver_common_kwargs = {"pivot": "mid", "width": 0.005}
        abs_max_in_domain = max([np.amax(np.abs(field.get_component(i))) for i in (0, 1)])
        arrows_xy = (
            np.mgrid[
                -(halo - 1) : grid[0] + 1 + (halo - 1) : 1,
                1 / 2 - halo : grid[1] + halo : 1,
            ],
            np.mgrid[
                1 / 2 - halo : grid[0] + halo : 1,
                -(halo - 1) : grid[1] + 1 + (halo - 1) : 1,
            ],
        )
        pyplot.xlim(-halo, grid[0] + halo)
        pyplot.ylim(-halo, grid[1] + halo)
        for dim in (0, 1):
            pyplot.quiver(
                *arrows_xy[dim],
                field.data[dim].flatten() / abs_max_in_domain if dim == 0 else 0,
                field.data[dim].flatten() / abs_max_in_domain if dim == 1 else 0,
                color=arrow_colors[dim],
                **quiver_common_kwargs,
            )
            for i, val in enumerate(field.data[dim].flatten()):
                if np.isfinite(val):
                    continue
                pyplot.annotate(
                    text="NaN",
                    xy=(
                        arrows_xy[dim][0].flatten()[i],
                        arrows_xy[dim][1].flatten()[i],
                    ),
                    ha="center",
                    va="center",
                    color=arrow_colors[dim],
                )
    else:
        assert False
    pyplot.hlines(
        y=range(-halo, grid[1] + 1 + halo),
        xmin=-halo,
        xmax=grid[0] + halo,
        color="r",
        linewidth=0.5,
    )
    pyplot.vlines(
        x=range(-halo, grid[0] + 1 + halo),
        ymin=-halo,
        ymax=grid[1] + halo,
        color="r",
        linewidth=0.5,
    )
    pyplot.hlines(y=range(grid[1] + 1), xmin=0, xmax=grid[0], color="r", linewidth=3)
    pyplot.vlines(x=range(grid[0] + 1), ymin=0, ymax=grid[1], color="r", linewidth=3)
    for i, x_y in enumerate(("x", "y")):
        getattr(pyplot, f"{x_y}ticks")(
            np.linspace(-halo + 0.5, grid[i] + halo - 0.5, grid[i] + 2 * halo)
        )
    pyplot.xlabel("x/dx (outer dim)")
    pyplot.ylabel("y/dy (inner dim)")
    pyplot.grid(linestyle=":")
    if plot:
        pyplot.show()
    else:
        pyplot.clf()
