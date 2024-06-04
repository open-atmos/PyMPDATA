import matplotlib
import numpy as np
from matplotlib import pylab, pyplot

from .formulae import convert_to, si


def plot(
    var,
    mult,
    label,
    output,
    rng=None,
    threshold=None,
    cmap="copper",
    rasterized=False,
    figsize=None,
):
    lines = {3: ":", 6: "--", 9: "-", 12: "-."}
    colors = {3: "crimson", 6: "orange", 9: "navy", 12: "green"}
    fctr = 50  # rebin by fctr in time dimension (https://gist.github.com/zonca/1348792)

    dt = (output["t"][1] - output["t"][0]) * fctr
    dz = output["z"][1] - output["z"][0]
    tgrid = np.concatenate(((output["t"][0] - dt / 2,), output["t"][0::fctr] + dt / 2))
    zgrid = np.concatenate(((output["z"][0] - dz / 2,), output["z"] + dz / 2))
    convert_to(zgrid, si.km)

    fig = pyplot.figure(constrained_layout=True, figsize=figsize)
    gs = fig.add_gridspec(25, 5)
    ax1 = fig.add_subplot(gs[:-1, 0:3])

    data = output[var] * mult
    lngth = data.shape[1] - 1
    assert (lngth // fctr) * fctr == lngth

    tmp = np.empty((data.shape[0], (data.shape[1] - 1) // fctr + 1))
    tmp[:, 0] = data[:, 0]
    M, N = data[:, 1:].shape
    m, n = M, N // fctr
    tmp[:, 1:] = data[:, 1:].reshape((m, M // m, n, N // n)).mean(3).mean(1)
    data = tmp

    if threshold is not None:
        data[data < threshold] = np.nan
    mesh = ax1.pcolormesh(
        tgrid / si.minutes,
        zgrid,
        data,
        cmap=cmap,
        rasterized=rasterized,
        vmin=None if rng is None else rng[0],
        vmax=None if rng is None else rng[1],
    )

    ax1.set_xlabel("time [min]")
    ax1.set_xticks(list(lines.keys()))
    ax1.set_ylabel("z [km]")
    ax1.grid()

    cbar = fig.colorbar(mesh, fraction=0.05, location="top")
    cbar.set_label(label)

    ax2 = fig.add_subplot(gs[:-1, 3:], sharey=ax1)
    ax2.set_xlabel(label)
    ax2.grid()
    if rng is not None:
        ax2.set_xlim(rng)

    last_t = -1
    for i, t in enumerate(output["t"]):
        x, z = output[var][:, i] * mult, output["z"].copy()
        convert_to(z, si.km)
        params = {"color": "black"}
        for line_t, line_s in lines.items():
            if last_t < line_t * si.minutes <= t:
                params["ls"] = line_s
                params["color"] = colors[line_t]
                ax2.step(x, z - (dz / si.km) / 2, where="pre", **params)
                ax1.axvline(t / si.minutes, **params)
        last_t = t


def plot_3d(psi, settings, options, r_min, r_max, max_height):
    max_psi = np.amax(psi / (1 / si.mg / si.um))
    if max_psi > max_height:
        raise ValueError(f"max(psi)={max_psi} > {max_height}")
    pylab.figure(figsize=(10, 5))
    ax = pylab.subplot(projection="3d")
    ax.view_init(75, 85)

    cmap = matplotlib.colormaps["gray_r"]
    min_height = 0

    dz = np.rot90(psi, 2).flatten()
    rgba = [cmap((k - min_height) / (1 + max_height)) for k in dz]
    dz[dz < 0.05 * max_height] = np.nan
    factor = 0.9
    ax.bar3d(
        *[
            arr.flatten()
            for arr in np.meshgrid(
                (settings.bin_boundaries[-2::-1] + settings.dr * (1 - factor) / 2)
                / si.um,
                (np.arange(settings.nz - 1, -1, -1) + (1 - factor) / 2)
                * settings.dz
                / si.km,
            )
        ],
        z=0,
        dx=factor * settings.dr / si.um,
        dy=factor * settings.dz / si.km,
        dz=dz / (1 / si.mg / si.um),
        shade=True,
        color=rgba,
        lightsource=matplotlib.colors.LightSource(azdeg=-64, altdeg=15),
        zsort="max",
    )
    ax.set_title(f"MPDATA iterations: {options.n_iters}")
    ax.set_xlabel("droplet radius [μm]")
    ax.set_ylabel("z [km]")
    ax.set_zlabel("psi [1/μm 1/mg 1/m]")
    ax.set_zlim(0, max_height / (1 / si.mg / si.um))
    ax.set_zticks([0, max_height / (1 / si.mg / si.um)])
    ax.set_ylim(settings.z_max / si.km, 0)
    ax.set_xlim(r_max / si.um, r_min / si.um)
    nticks = 6
    ax.set_xticks((r_min + np.arange(nticks + 1) * (r_max - r_min) / nticks) / si.um)
