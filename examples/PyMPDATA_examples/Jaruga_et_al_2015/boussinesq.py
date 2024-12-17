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

N, M  = 200, 200
dx, dy = 10, 10
Tht_ref = 300
g = 9.81
r0 = 250
nt = 200
dt = 0.75

options = Options(n_iters=2,infinite_gauge= True,nonoscillatory = True)
mesh = np.full((N,M), fill_value=Tht_ref, dtype=float)

xi, yi = np.indices((N, M))
mask = np.sqrt((xi*dx-1000)**2+(yi*dy-260)**2) < r0
mesh += np.where(mask, 0.5, 0)

plt.imshow(mesh.T,origin='lower')
plt.colorbar()
plt.show()


halo = options.n_halo
bcond = (Periodic(), Periodic())

field_ctor_kwargs = {'halo': halo, 'boundary_conditions': bcond}
advector = VectorField(data=(np.zeros((N+1, M)), np.zeros((N, M+1))), **field_ctor_kwargs)
stepper = Stepper(options=options, grid=(N, M))

solver_ctor_kwargs = {'stepper': stepper, 'advector': advector}
solvers = {
    'tht': Solver(advectee=ScalarField(data=mesh, **field_ctor_kwargs), **solver_ctor_kwargs),
    'u': Solver(advectee=ScalarField(data=np.zeros((N,M)), **field_ctor_kwargs), **solver_ctor_kwargs),
    'w': Solver(advectee=ScalarField(data=np.zeros((N,M)), **field_ctor_kwargs), **solver_ctor_kwargs),
}
state_0 = solvers["tht"].advectee.get().copy()

#actual going forward
outfreq = 100
prs_tol = 1e-7;
output = []

rhs_w = np.zeros((N,M))
stash = {'u': np.zeros((N,M)), 'w': np.zeros((N,M))}
vip_rhs = {'u': np.zeros((N,M)), 'w': np.zeros((N,M))}
Phi = np.zeros((N,M))
tmp_uvw = {'u': np.zeros((N,M)), 'w': np.zeros((N,M))}


hook_ante_loop(Phi,vip_rhs,solvers,tmp_uvw)#here
for step in range(nt + 1):
    if step != 0:
        print(advector.data[0])
        calc_gc_extrapolate_in_time(solvers, stash) # reads & writes to stash
        calc_gc_interpolate_in_space(advector, stash,dt,(dx,dy)) # reads from stash
        fill_stash(solvers, stash) # writes to stash
        apply_rhs(solvers["w"].advectee.get(), rhs_w, dt/2)
        vip_rhs_apply(dt,vip_rhs, solvers)
        for solver in solvers.values():
            solver.advance(n_steps=1)
        update_rhs(tht=solvers["tht"].advectee.get(), rhs_w=rhs_w, g=g, tht_ref=Tht_ref)
        apply_rhs(solvers["w"].advectee.get(), rhs_w, dt/2)
        vip_rhs_impl_fnlz(vip_rhs,dt,solvers)
    if step % outfreq == 0:
        output.append(solvers["tht"].advectee.get().copy())
    print(step)

plt.clf()
plt.imshow(mesh.T,origin='lower')
plt.colorbar()
plt.show()

