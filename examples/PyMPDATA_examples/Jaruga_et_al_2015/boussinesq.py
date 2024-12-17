import os
os.environ['NUMBA_DISABLE_JIT']='1'

from PyMPDATA import Options
from PyMPDATA.boundary_conditions import Periodic
from PyMPDATA import ScalarField
from PyMPDATA import VectorField
from PyMPDATA import Solver
from PyMPDATA import Stepper
import numpy as np
import matplotlib.pyplot as plt
from PyMPDATA_examples.Jaruga_et_al_2015.temp import *

N, M  = 20, 20
dxy = 100, 100
Tht_ref = 300
g = 9.81
r0 = 250
nt = 10
dt = 7.5
beta  = 0.25
options = Options(n_iters=2)#,infinite_gauge= True,nonoscillatory = True)
assert options.n_halo == 1
mesh = np.full((N,M), fill_value=Tht_ref, dtype=float)

xi, yi = np.indices((N, M))
mask = np.sqrt((xi*dxy[0]-1000)**2+(yi*dxy[1]-260)**2) < r0
mesh += np.where(mask, 0.5, 0)

plt.imshow(mesh.T,origin='lower')
plt.colorbar()
plt.show()


halo = options.n_halo
bcond = (Periodic(), Periodic())

field_ctor_kwargs = {'halo': halo, 'boundary_conditions': bcond}

def new_sf(N,M):
    return ScalarField(data=np.zeros((N,M)), **field_ctor_kwargs)

advector = VectorField(data=(np.zeros((N+1, M)), np.zeros((N, M+1))), **field_ctor_kwargs)
stepper = Stepper(options=options, grid=(N, M))

solver_ctor_kwargs = {'stepper': stepper, 'advector': advector}
solvers = {
    'tht': Solver(advectee=ScalarField(data=mesh, **field_ctor_kwargs), **solver_ctor_kwargs),
    'u': Solver(advectee=new_sf(N,M), **solver_ctor_kwargs),
    'w': Solver(advectee=new_sf(N,M), **solver_ctor_kwargs),
}
state_0 = solvers["tht"].advectee.get().copy()

#actual going forward
outfreq = 1
prs_tol = 1e-7;
err_tol = prs_tol/dt
output = []
k_iters = 4
rhs_w = np.zeros((N,M))
stash = {k: new_sf(N,M) for k in ('u', 'w')}
vip_rhs = {'u': np.zeros((N,M)), 'w': np.zeros((N,M))}
Phi = new_sf(N,M)
tmp_uvw = {'u': new_sf(N,M), 'w': new_sf(N,M)}
lap_tmp = {'u': new_sf(N,M), 'w': new_sf(N,M)}
lap_err = np.zeros((N,M))
err = new_sf(N,M)
p_err = [new_sf(N,M) for _ in range(k_iters)]
lap_p_err = [np.empty((N,M)) for _ in range(k_iters)]

def debug(where):
    psi = vip_rhs['w']
    print(f"{where=} {np.amin(psi)=} {np.amax(psi)=}")

hook_ante_loop(Phi,beta,vip_rhs,solvers,tmp_uvw,lap_tmp,err,p_err,lap_p_err,k_iters,err_tol,lap_err,dxy,N,M)#here
for step in range(nt + 1):
    if step != 0:
        calc_gc_extrapolate_in_time(solvers, stash) # reads & writes to stash
        calc_gc_interpolate_in_space(advector, stash,dt,dxy) # reads from stash
        fill_stash(solvers, stash) # writes to stash
        apply_rhs(solvers["w"].advectee.get(), rhs_w, dt/2)
        debug("A")
        vip_rhs_apply(dt,vip_rhs, solvers)
        debug("B")
        for solver in solvers.values():
            solver.advance(n_steps=1)
        update_rhs(tht=solvers["tht"].advectee.get(), rhs_w=rhs_w, g=g, tht_ref=Tht_ref)
        apply_rhs(solvers["w"].advectee.get(), rhs_w, dt/2)
        debug("C")
        vip_rhs_impl_fnlz(vip_rhs,dt,solvers,err,Phi,beta,lap_tmp,tmp_uvw,p_err,lap_p_err,dxy,k_iters,err_tol,lap_err)
        debug("D")
    if step % outfreq == 0:
        output.append(solvers["tht"].advectee.get().copy())
    print(step)

    plt.clf()
    plt.imshow(solvers["tht"].advectee.get().T,origin='lower')
    plt.colorbar()
    plt.show()

