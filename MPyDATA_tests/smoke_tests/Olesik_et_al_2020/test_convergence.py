import numpy as np
from MPyDATA_examples.Olesik_et_al_2020.setup import Setup
from MPyDATA_examples.Olesik_et_al_2020.coordinates import x_id, x_p2, x_p3, x_log_of_pn
from MPyDATA.options import Options
from MPyDATA_examples.Olesik_et_al_2020.simulation import Simulation
from joblib import Parallel, parallel_backend, delayed
from MPyDATA_examples.Olesik_et_al_2020.physics.equilibrium_drop_growth import PdfEvolver
from MPyDATA.arakawa_c.discretisation import discretised_analytical_solution
from MPyDATA_examples.utils.error_norms import GMD
from difflib import context_diff
import pathlib




GCs = np.linspace(.15,.85, 3)
nrs = np.linspace(64,256, 3, dtype = int)

grid_layout = x_p2()
psi_coord = x_p2()
#TODO: run over all opt_set
opt_set = (
    {'n_iters': 1},
    # {'n_iters': 2},
    # {'n_iters': 2,'infinite_gauge':True},
    # {'n_iters': 2, 'infinite_gauge': True, 'flux_corrected_transport': True},
    # {'n_iters': 2, 'third_order_terms':True},
    # {'n_iters': 3},
    # {'n_iters': 3, 'third_order_terms': True, 'infinite_gauge': True, 'flux_corrected_transport': True}
)




def test_convergence(generate = False, plot = False, opt_set=opt_set):
    for opt in opt_set:
        options = Options(**opt)
        with parallel_backend('threading', n_jobs=-2):
            results0 = Parallel(verbose=10)(
                delayed(analysis)(nr, GC, options)
                for nr in nrs
                for GC in GCs
            )
        results = [list(i) for i in zip(*results0)]
        values  = np.array(results[0:2])
        if plot:
            from MPyDATA_examples.Olesik_et_al_2020.demo_plot_convergence import plot
            from MPyDATA_examples.utils.show_plot import show_plot
            plot(values[0], values[1], values[2], n_levels=10)
            show_plot(filename=f'convergence.pdf')
        v_str = "\n".join(" ".join(map(str, x)) for x in values)
        with open(pathlib.Path(__file__).parent.joinpath("convergence_refdata.txt"), "w+" if generate else "r") as f:
            if generate:
                f.write(v_str)
            else:
                try:
                    assert ''.join(context_diff(f.read(), v_str)) == ''
                except: raise ValueError


def analysis(nr, GC,options):
    setup = Setup(nr = nr, mixing_ratios_g_kg = [4,])
    simulation = Simulation(setup, grid_layout,  psi_coord, options, GC)
    simulation.step(simulation.out_steps[-1])
    t = simulation.out_steps[-1] * simulation.dt
    rh = simulation.rh
    pdf_t = PdfEvolver(setup.pdf, setup.drdt, t)
    analytical = discretised_analytical_solution(
                simulation.rh.magnitude,
                lambda r: pdf_t(r * rh.units).magnitude
            ) * pdf_t(rh[0]).units
    numerical = simulation.n
    loc_of_maximum_num = simulation.r[np.argmax(numerical)]
    loc_of_maximum_anal = simulation.r[np.argmax(analytical)]
    maximum_num = np.max(numerical)
    maximum_anal = np.max(analytical)
    measure_location = (loc_of_maximum_num / loc_of_maximum_anal).magnitude
    measure_height = (maximum_num / maximum_anal).magnitude
    error_GMD = np.log(GMD(numerical.magnitude, analytical.magnitude, t.magnitude)) / np.log(2)
    return nr, GC, error_GMD, measure_location, measure_height
