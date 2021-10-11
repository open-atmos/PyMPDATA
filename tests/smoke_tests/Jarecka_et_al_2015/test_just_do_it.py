import numpy as np
from matplotlib import pylab
from PyMPDATA_examples.Jarecka_et_al_2015 import Settings, Simulation, analytic_equations


def test_just_do_it(plot=True):
    # arrange
    settings = Settings()
    simulation = Simulation(settings)
    times = (1, 3, 7)
    lambdas_analytic = analytic_equations.d2_el_lamb_lamb_t_evol(
        times=[0] + list(times),
        lamb_x0=settings.lx0,
        lamb_y0=settings.ly0
    )[1:]

    # act
    output = simulation.run()

    # plot
    cuts = ('x', 'y')
    fig, axs = pylab.subplots(
        nrows=len(times),
        ncols=len(cuts),
        figsize=(9, 9),
        constrained_layout=True
    )
    momenta = {'x': 'uh', 'y': 'vh'}
    for i_cut, cut in enumerate(cuts):
        if cut == 'x':
            idx = (slice(None, None), slice(settings.nx // 2, settings.nx // 2 + 1))
            coord = settings.dx * (np.linspace(-settings.nx // 2, settings.nx // 2, settings.nx) + .5)
            x = coord
            y = 0
        else:
            idx = (slice(settings.ny // 2, settings.ny // 2 + 1), slice(None, None))
            coord = settings.dy * (np.linspace(-settings.ny // 2, settings.ny // 2, settings.ny) + .5)
            x = 0
            y = coord
        for i_t, t in enumerate(times):
            h_analytic = Simulation.amplitude(x, y, *lambdas_analytic[i_t, (0, 2)])
            axs[i_t, i_cut].set_xlim(-8.5, 8.5)
            axs[i_t, i_cut].set_xticks((-5, 0, 5))
            axs[i_t, i_cut].plot(coord, output[0]['h'][idx].squeeze(), color='black', label='h @ t=0')
            axs[i_t, i_cut].plot(coord, output[t]['h'][idx].squeeze(), color='red', label='MPDATA: h')
            axs[i_t, i_cut].plot(coord, h_analytic,
                                 color='blue', linestyle=':', label='analytic: h')
            twin = axs[i_t, i_cut].twinx()
            q = output[t][momenta[cut]][idx].squeeze()
            h = output[t]['h'][idx].squeeze()
            twin.plot(
                coord,
                np.where(h > settings.eps, np.divide(q, h, where=h > settings.eps), 0),
                color='red',
                linestyle='--',
                label='MPDATA: q'
            )
            twin.plot(
                coord,
                np.where(h_analytic > settings.eps,
                         coord / lambdas_analytic[i_t, 0 + 2 * i_cut] * lambdas_analytic[i_t, 1 + 2 * i_cut],
                         0),
                linestyle='-.',
                label='analytic: q',
                color='blue'
            )
            twin.set_ylabel('u' if cut == 'x' else 'v')
            twin.set_ylim(-1.2, 1.2)
            twin.set_yticks((-1, 0, 1))
            axs[i_t, i_cut].set_xlabel(cut)
            axs[i_t, i_cut].set_ylim(-.6, .6)
            axs[i_t, i_cut].set_yticks((-.5, 0, .5))
            axs[i_t, i_cut].set_ylabel('h')
            axs[i_t, i_cut].set_title(f"t={t}", y=1.0, pad=-14, x=0.075)
    legend_props = {'frameon': False, 'fontsize': 'small'}
    axs[i_t, i_cut].legend(loc='lower right', **legend_props)
    twin.legend(loc='lower center', **legend_props)
    if plot:
        pylab.show()

    # assert
    # TODO #273 + make the plotting code common with the example

