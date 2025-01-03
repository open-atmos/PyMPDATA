from PyMPDATA import Options,ScalarField,VectorField,Solver,Stepper
from PyMPDATA.boundary_conditions import Periodic
from PyMPDATA_examples.Jaruga_et_al_2015.temp import *
import numpy as np
import matplotlib.pyplot as plt
from time import time
import os
#os.environ['NUMBA_DISABLE_JIT']='1'
np.set_printoptions(linewidth=300, precision=3)
start0 = time()
N, M  = 201, 201
dxy = 2000/(N-1), 2000/(M-1)
Tht_ref = 300.
g = 9.81
r0 = 250.
dt = 0.75
nt = int(600//dt)
assert dt * nt == 600
beta  = 0.25
options = Options(n_iters=2, infinite_gauge= True,nonoscillatory = True)
mesh = np.full((N,M), fill_value=Tht_ref, dtype=float)

xi, yi = np.indices((N, M))
mask = np.sqrt((xi*dxy[0]-1000)**2+(yi*dxy[1]-260)**2) < r0
mesh += np.where(mask, 0.5, 0)

def plot(step):
    data = solvers['tht'].advectee.get()
    plt.title("tht (t/dt="+str(step)+")")
    plt.xlabel("x/dx")
    plt.ylabel("y/dy")
    plt.imshow(data.T, origin='lower',extent=[-100,100,-100,100])#[minx,maxx,miny,maxy] can be made into variable
    plt.colorbar()
    plt.savefig(f"output_step={step}.svg")
    plt.close()


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
outfreq = 20
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
stop0 = time()
def debug(where):
    psi = vip_rhs['w']
    print(f"{where=} {np.amin(psi)=} {np.amax(psi)=}")
#action before loop
# correct initial velocity
start1 = time()
Phi.get()[:] = 0

pressure_solver_update(solvers,Phi,beta,lap_tmp,tmp_uvw,err,p_err,lap_p_err,dxy,k_iters,err_tol,lap_err,simple = True)

xchng_pres(Phi)
calc_grad(tmp_uvw, Phi, dxy)
pressure_solver_apply(solvers,tmp_uvw)
# potential pressure
ini_pressure(Phi,solvers,N,M)
# allow pressure_solver_apply at the first time step
xchng_pres(Phi)
calc_grad(tmp_uvw, Phi, dxy)
for k in ('u', 'w'):
     vip_rhs[k][:] -= tmp_uvw[k].get()
stop1 = time()
for step in range(nt + 1):
    if step != 0:

        calc_gc_extrapolate_in_time(solvers, stash) # reads & writes to stash
        calc_gc_interpolate_in_space(advector, stash,dt,dxy) # reads from stash
        fill_stash(solvers, stash) # writes to stash
        apply_rhs(solvers["w"].advectee.get(), rhs_w, dt/2)

        vip_rhs_apply(dt,vip_rhs, solvers)

        for solver in solvers.values():
            solver.advance(n_steps=1)

        update_rhs(tht=solvers["tht"].advectee.get(), rhs_w=rhs_w, g=g, tht_ref=Tht_ref)
        apply_rhs(solvers["w"].advectee.get(), rhs_w, dt/2)

        for k in ('u', 'w'):
            vip_rhs[k][:] = -solvers[k].advectee.get()
        pressure_solver_update(solvers,Phi,beta,lap_tmp,tmp_uvw,err,p_err,lap_p_err,dxy,k_iters,err_tol,lap_err)
        pressure_solver_apply(solvers,tmp_uvw)

        for k in ('u', 'w'):
            vip_rhs[k][:] += solvers[k].advectee.get()
            vip_rhs[k][:] /= 0.5 * dt
        
    if step % outfreq == 0:
        output.append(solvers["tht"].advectee.get().copy())
        plot(step)
    print(step)

