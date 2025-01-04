import numpy as np
from PyMPDATA.scalar_field import ScalarField
from PyMPDATA.vector_field import VectorField
from numba import njit
#@njit
def vip_rhs_apply(dt,vip_rhs, solvers):
     for k in ('u', 'w'):
         solvers[k].advectee.get()[:] += 0.5 * dt * vip_rhs[k]
         vip_rhs[k][:] = 0
#@njit
def calc_gc_extrapolate_in_time(solvers, stash):
    for k in ('u', 'w'):
        stash[k].get()[:] = -.5 * stash[k].get() + 1.5 * solvers[k].advectee.get()
        stash[k].get()[:] = -.5 * stash[k].get() + 1.5 * solvers[k].advectee.get()
        xchng_pres(stash[k])   
#@njit
def calc_gc_interpolate_in_space(advector: VectorField, stash: dict, dt:float, dxy: tuple):
    idx_diff = ( 
        (slice(None, -1), slice(None, None)),
        (slice(None, None), slice(None, -1)),
    )
    for axis, psi in enumerate(('u', 'w')):
        advector.data[axis][:] = dt / dxy[axis] * (
            np.diff(stash[psi].data, axis=axis) / 2 + stash[psi].data[idx_diff[axis]]
        )[:]
#@njit
def fill_stash(solvers, stash):
    for k in stash.keys():
        stash[k].get()[:] = solvers[k].advectee.get()
        xchng(stash[k].data, h=2)
#@njit
def apply_rhs(w, rhs_w : np.ndarray, dt : float):
    w += rhs_w * dt #same behaviour in step 1 observed in rhs_w as in libmpdata++
#@njit
def ini_pressure(Phi, solvers, N, M):
    npoints = N*M
    for k in ('u', 'w'):
        Phi.get()[:] -= 0.5 * np.power(solvers[k].advectee.get()[:],2)
    Phi_mean = np.sum(Phi.get()) / npoints
    Phi.get()[:] -= Phi_mean
#@njit
def xchng(data, h):
    data[0:h,:] = data[-2*h:-h,:]
    data[-h:,:] = data[h:2*h,:]
    data[:,0:h] = data[:,-2*h:-h]
    data[:,-h:] = data[:,h:2*h]
#@njit
def xchng_pres(Phi):
    xchng(h = Phi.halo, data=Phi.data)
#@njit
#TODO: checking...
def update_rhs(tht : np.ndarray, rhs_w : np.ndarray, tht_ref : int, g : float):

    rhs_w[:] = 0
    rhs_w[:] += g * (tht - tht_ref) / tht_ref
    """
    only iteration 0
    rhs.at(ix::tht)(ijk) +=  - this->tht_abs(ijk) * (tht(ijk) - this->tht_e(ijk));

    rhs_w[:] += g *  (this->state(ix::tht)(ijk) - this->tht_e(ijk)) / Tht_ref
    rest of iterations
     rhs.at(ix::tht)(ijk) +=  - this->tht_abs(ijk) * (
                            (tht(ijk) + real_t(0.5) * this->dt * this->tht_abs(ijk) * this->tht_e(ijk))
                            / (1 + real_t(0.5) * this->dt * this->tht_abs(ijk))- this->tht_e(ijk));
    rhs_w[:] += g * ((this->state(ix::tht)(ijk)
                                 + real_t(0.5) * this->dt * this->tht_abs(ijk) * this->tht_e(ijk))
                            / (1 + real_t(0.5) * this->dt * this->tht_abs(ijk))
                            - this->tht_e(ijk)
                            ) / this->Tht_ref
    """
#@njit    
def div(lap_tmp,dxy):
    h = lap_tmp['u'].halo
    return (
        np.gradient(lap_tmp['u'].data, dxy[0],axis=0) +
        np.gradient(lap_tmp['w'].data, dxy[1],axis=1)
    )[h:-h,h:-h]
#@njit
def lap(Phi, dxy, err_init, lap_tmp,tmp_uvw = None):
    xchng_pres(Phi)
    calc_grad(lap_tmp, Phi, dxy);
    if (err_init):
        for k in ('u', 'w'):
            lap_tmp[k].get()[:] -= tmp_uvw[k].get()

    for k in ('u', 'w'):
        xchng_pres(lap_tmp[k])
        
    return div(lap_tmp, dxy)
#@njit
def pressure_solver_loop_init(err,p_err,lap_p_err,dxy,lap_tmp,tmp_uvw):
    p_err[0].get()[:] = err.get()[:]
    lap_p_err[0][:] = lap(p_err[0], dxy, False, lap_tmp,tmp_uvw)
##@njit
def pressure_solver_loop_body(Phi,beta,converged,err,p_err,lap_p_err,dxy,k_iters,err_tol,lap_err,lap_tmp):
    tmp_den = [1.]*k_iters
    alpha = [1.]*k_iters
    for v in range(0,k_iters):

          tmp_den[v] = np.sum(lap_p_err[v]**2)
          if (tmp_den[v] != 0):
              beta = - np.dot(
                  err.get().ravel(),
                  lap_p_err[v].ravel()
              ) / tmp_den[v]
          Phi.get()[:] += beta * p_err[v].get()
          err.get()[:] += beta * lap_p_err[v]

          error = max(
             abs(np.amax(err.get())),
             abs(np.amin(err.get()))
          )

          if (error <= err_tol): 
              converged = True

          lap_err[:] = lap(err, dxy, False, lap_tmp);
          
          for l in range(v):
              if (tmp_den[l] != 0):
                    alpha[l] = - np.dot(lap_err.ravel(), lap_p_err[l].ravel()) / tmp_den[l];
          if (v < (k_iters - 1)):
              p_err[v + 1].get()[:] = err.get()
              lap_p_err[v + 1][:] = lap_err[:]
              for l in range(v):
                  p_err[v + 1].get()[:] += alpha[l] * p_err[l].get()
                  lap_p_err[v + 1][:] += alpha[l] * lap_p_err[l]
          else:
              p_err[0].get()[:] = err.get()[:] + alpha[0] * p_err[0].get()
              lap_p_err[0][:] = lap_err[:] + alpha[0] * lap_p_err[0]
              for l in range(1,v+1):
                    p_err[0].get()[:] += alpha[l] * p_err[l].get()
                    lap_p_err[0][:] += alpha[l] * lap_p_err[l]
    return converged


#@njit
def pressure_solver_update(solvers,Phi,beta,lap_tmp,tmp_uvw,err,p_err,lap_p_err,dxy,k_iters,err_tol,lap_err,simple = False):
    for k in ('u', 'w'):
         tmp_uvw[k].get()[:] = solvers[k].advectee.get()
    
    #initial error
    err.get()[:] = lap(Phi, dxy, True,lap_tmp,tmp_uvw)

    iters = 0
    converged = False

    pressure_solver_loop_init(err,p_err,lap_p_err,dxy,lap_tmp,tmp_uvw)
    #pseudo-time loop
    while not converged:        
         converged = pressure_solver_loop_body(Phi,beta,converged,err,p_err,lap_p_err,dxy,k_iters,err_tol,lap_err,lap_tmp)
         iters += 1

         if (iters > 10000): # going beyond 10000 iters means something is really wrong,
             # usually boundary conditions but not always !
             raise Exception("stuck in pressure solver")

    xchng_pres(Phi)

    calc_grad(tmp_uvw, Phi, dxy)

#@njit
def pressure_solver_apply(solvers,tmp_uvw):
    for k in ('u', 'w'):
        solvers[k].advectee.get()[:] -= tmp_uvw[k].get()

#@njit
def calc_grad(arg : ScalarField, Phi, dxy):
    h = Phi.halo
    idx = (slice(h,-h),slice(h,-h))
    arg['u'].get()[:] = np.gradient(Phi.data,dxy[0],axis = 0)[idx]
    arg['w'].get()[:] = np.gradient(Phi.data,dxy[1],axis = 1)[idx]


