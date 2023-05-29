from matplotlib import pylab
import numpy as np
from PyMPDATA_examples.Jarecka_et_al_2015 import formulae


def plot_output(times, output, settings, return_data=True):
    lambdas_analytic = formulae.d2_el_lamb_lamb_t_evol(
        times=[0] + list(times),
        lamb_x0=settings.lx0,
        lamb_y0=settings.ly0
    )[1:]

    data = {}
    cuts = ('x', 'y')
    _, axs = pylab.subplots(
        nrows=len(times),
        ncols=len(cuts),
        figsize=(9, 9),
        constrained_layout=True
    )
    momenta = {'x': 'uh', 'y': 'vh'}
    for i_cut, cut in enumerate(cuts):
        if cut == 'x':
            n = settings.nx
            idx = (slice(None, None), slice(n // 2, n // 2 + 1))
            coord = settings.dx * (np.linspace(-n // 2, n // 2, n) + .5)
            x = coord
            y = 0
        else:
            n = settings.ny
            idx = (slice(n // 2, n // 2 + 1), slice(None, None))
            coord = settings.dy * (np.linspace(-n // 2, n // 2, n) + .5)
            x = 0
            y = coord
        for i_t, t in enumerate(times):
            key = f"cut={cut} t={t}"
            data[key] = {'coord': coord, 'x':x, 'y':y}
            datum = data[key]
            datum['h_analytic'] = formulae.amplitude(x, y, *lambdas_analytic[i_t, (0, 2)])
            where = 'mid'
            axs[i_t, i_cut].set_xlim(-8.5, 8.5)
            axs[i_t, i_cut].set_xticks((-5, 0, 5))
            axs[i_t, i_cut].step(datum['coord'], output[0]['h'][idx].squeeze(),
                                 color='black', label='h @ t=0', where=where)
            axs[i_t, i_cut].step(datum['coord'], output[t]['h'][idx].squeeze(),
                                 color='red', label='MPDATA: h', where=where)
            axs[i_t, i_cut].plot(datum['coord'], datum['h_analytic'],
                                 color='blue', linestyle=':', label='analytic: h')
            twin = axs[i_t, i_cut].twinx()
            datum['h_numeric'] = output[t]['h'][idx].squeeze()
            mask = datum['h_numeric'] > settings.eps
            datum['q_h_numeric'] = np.where(mask, np.nan, 0)
            np.divide(output[t][momenta[cut]][idx].squeeze(), datum['h_numeric'],
                      where=mask, out=datum['q_h_numeric'])
            twin.step(
                datum['coord'],
                datum['q_h_numeric'],
                color='red',
                linestyle='--',
                label='MPDATA: q/h',
                where=where
            )
            datum['q_h_analytic'] = np.where(
                datum['h_analytic'] > settings.eps,
                datum['coord'] / lambdas_analytic[i_t, 0 + 2 * i_cut] \
                    * lambdas_analytic[i_t, 1 + 2 * i_cut],
                0)
            twin.plot(
                datum['coord'],
                datum['q_h_analytic'],
                linestyle='-.',
                label='analytic: q/h',
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
            axs[i_t, i_cut].grid()

    legend_props = {'frameon': False, 'fontsize': 'small'}
    axs[-1, -1].legend(loc='lower right', **legend_props)
    twin.legend(loc='lower center', **legend_props)
    if return_data:
        return data
    return None
