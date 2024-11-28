from PyMPDATA_examples.Jarecka_et_al_2015 import Simulation

def vip_rhs_apply():
    pass
def calc_gc_extrapolate_in_time():
    pass
def calc_gc_interpolate_in_space(advector, solvers):

    idx = ((slice(1, -1), slice(None, None)), (slice(None, None), slice(1, -1)))

    for axis, psi in enumerate(('u', 'w')):
        advector.get_component(axis)[idx[axis]] = (
            Simulation.interpolate(solvers[psi].advectee.get(), axis)
        )    
def fill_stash(solvers, stash):
    for k in stash.keys():
        stash[k][:] = solvers[k].advectee.get()

def vip_rhs_expl_calc():
    pass
def apply_rhs(w, rhs_w, dt):
    w += rhs_w * dt/2

def hook_ante_loop_prs():
    pass
def hook_ante_loop_rhs_vip():
    pass
def ini_pressure():
    pass
def update_rhs(*, tht, rhs_w, tht_ref, g):
    rhs_w[:] += g * (tht - tht_ref) / tht_ref

def vip_rhs_impl_fnlz():
    pass