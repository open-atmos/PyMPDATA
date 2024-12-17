from PyMPDATA_examples.Jarecka_et_al_2015 import Simulation
import numpy as np

from PyMPDATA.vector_field import VectorField
def vip_rhs_apply(dt,vip_rhs, solvers):
     for k in ('u', 'w'):
         solvers[k].advectee.get()[:] += 0.5 * dt * vip_rhs[k][:]
         vip_rhs[k][:] = 0

def calc_gc_extrapolate_in_time(solvers,stash):
    for k in ('u', 'w'):
        stash[k] = -.5 * stash[k] + 1.5 * solvers[k].advectee.get()
#    

def calc_gc_interpolate_in_space(advector: VectorField,stash: dict,dt:float,dxy: tuple):

    idx = (
        (slice(1, -1), slice(None, None)),  # [1:-1, :]
        (slice(None, None), slice(1, -1))   # [:, 1:-1]
    )

    for axis, psi in enumerate(('u', 'w')):
        advector.get_component(axis)[idx[axis]] = dt/dxy[axis]*(
            Simulation.interpolate(stash[psi], axis)
        )    
def fill_stash(solvers, stash):
    for k in stash.keys():
        stash[k][:] = solvers[k].advectee.get()

def apply_rhs(w, rhs_w, dt):
    w += rhs_w * dt/2




def ini_pressure():
    """
    #Phi = 0
    #npoints = 1
    #for d in range(0,ndim):
    #    Phi -= real_t(0.5) * np.power(""vips()[d](this->ijk)"",2)
    #    npoints *= ""this->mem->distmem.grid_size[d]"" #Do I have this variable somewhere???
    #Phi_mean = prs_sum(Phi, ijk) / npoints #is prs_sum implemented? shold I go from procedural to object programming or start procedural and make object?
    #Phi -= Phi_mean
    #return Phi
    pass
    """
def update_rhs(*, tht, rhs_w, tht_ref, g):
    rhs_w[:] += g * (tht - tht_ref) / tht_ref

def pressure_solver_update(simple = False):
    pass
def pressure_solver_apply():
    pass

def normalize_vip(solvers):
    pass

def calc_grad(tmp_uvw, Phi, dxy):
    pass
def hook_ante_loop(Phi,vip_rhs,solvers,tmp_uvw):
       
     # save_edges does nothing set edges too(empty function + no change in chart)
     #xchng_pres seems to do nothing(no change in chart)
     # correct initial velocity
     Phi[:] = 0
     #xchng_pres(Phi)

     pressure_solver_update(True)

     #xchng_pres(Phi);
     calc_grad(tmp_uvw, Phi, dxy)
     pressure_solver_apply();

     # potential pressure
     ini_pressure();

     # allow pressure_solver_apply at the first time step
     #xchng_pres(Phi)
     calc_grad(tmp_uvw, Phi, this->ijk, this->dijk);
     for k in ('u', 'w'):
         vip_rhs[k][:]  -= tmp_uvw[k]
            

         
def vip_rhs_impl_fnlz(vip_rhs,dt,solvers):
    for k in ('u', 'w'):
        vip_rhs[k][:] = -solvers[k].advectee.get()
    pressure_solver_update()
    pressure_solver_apply()
    #normalize_vip(solvers) connected to xchng_pres only

    for k in ('u', 'w'):
        vip_rhs[k][:] += solvers[k].advectee.get()
        vip_rhs[k][:] /= 0.5 * dt