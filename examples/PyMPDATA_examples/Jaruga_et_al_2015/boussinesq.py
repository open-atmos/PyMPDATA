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
"""
struct ct_params_t //: ct_params_default_t
  {
    using real_t = double;
    enum { opts = opts::iga | opts::fct };
    enum { sptl_intrp = 0 }; // spatial interpolation of velocities
    enum { tmprl_extrp = 0 }; // temporal extrapolation of velocities
    enum { out_intrp_ord = 1 };  // order of temporal interpolation for output
    enum { hint_norhs = 0 }; // hint which equations feature zero on the right-hand side

    enum { n_dims = 2 };
    enum { n_eqns = 3 };
    enum { rhs_scheme = solvers::trapez };
    """
"""
    struct ix { enum {
      u, w, tht, 
      vip_i=u, vip_j=w, vip_den=-1
    }; };
  }; 

  using ix = typename ct_params_t::ix;
  using real_t = typename ct_params_t::real_t;
"""
  #options
options = Options(n_iters=2,infinite_gauge= True,nonoscillatory = True)
mesh = np.full((N,M), fill_value=Tht_ref, dtype=float)

xi, yi = np.indices((N, M))
mask = np.sqrt((xi*dx-1000)**2+(yi*dy-260)**2) < r0
mesh += np.where(mask, 0.5, 0)

plt.imshow(mesh.T,origin='lower')
plt.colorbar()
plt.show()


"""

  const int r0 = 250; 
  const int nx = 201, ny = 201, nt = 800;
  typename ct_params_t::real_t Tht_ref = 300; // reference state (constant throughout the domain)
"""
"""
  using slv_out_t = 
      output::gnuplot<
        solvers::boussinesq<ct_params_t>
    >;
"""
"""
  libmpdataxx::concurr::serial<
    slv_out_t, 
    bcond::cyclic, bcond::cyclic,
    bcond::cyclic, bcond::cyclic
  > slv(p);
"""
 #solver declaration 

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

"""
  {
    // initial condition
    blitz::firstIndex i;
    blitz::secondIndex j;
    slv.sclr_array("tht_e") = Tht_ref;
    slv.advectee(ix::tht) = Tht_ref + where(
      // if
      pow(i * p.di - 4    * r0 , 2) + 
      pow(j * p.dj - 1.04 * r0 , 2) <= pow(r0, 2), 
      // then
      .5, 
      // else
      0
    );
    slv.advectee(ix::u) = 0; 
    slv.advectee(ix::w) = 0; 
  }
"""
#init
"""
  // integration
  slv.advance(nt);  
"""
#actual going forward
outfreq = 100
prs_tol = 1e-7;
output = []

rhs_w = np.zeros((N,M))
stash = {'u': np.zeros((N,M)), 'w': np.zeros((N,M))}

hook_ante_loop_prs()
hook_ante_loop_rhs_vip()
ini_pressure()

for step in range(nt + 1):
    if step != 0:
        # TODO: extrapolation from stash and current state into stash
        calc_gc_extrapolate_in_time(solvers, stash) # reads & writes to stash
        # TODO: use stash instead of state for spatial interpolation
        calc_gc_interpolate_in_space(advector, stash) # reads from stash
        fill_stash(solvers, stash) # writes to stash
        vip_rhs_expl_calc()
        apply_rhs(solvers["w"].advectee.get(), rhs_w, dt/2)
        vip_rhs_apply()
        for solver in solvers.values():
            solver.advance(n_steps=1)
        update_rhs(tht=solvers["tht"].advectee.get(), rhs_w=rhs_w, g=g, tht_ref=Tht_ref)
        apply_rhs(solvers["w"].advectee.get(), rhs_w, dt/2)
        vip_rhs_impl_fnlz()
    if step % outfreq == 0:
        output.append(solvers["tht"].advectee.get().copy())
    print(step)
"""
  // run-time parameters
  slv_out_t::rt_params_t p;

  p.dt = .75;
  p.di = p.dj = 10.; 
  p.Tht_ref = Tht_ref; 

  p.outfreq = 100; //12;
  p.outvars = {
//    {ix::u,   {.name = "u",   .unit = "m/s"}}, dou
//    {ix::w,   {.name = "w",   .unit = "m/s"}}, 
    {ix::tht, {"tht", "K"  }}
  };
  p.gnuplot_view = "map";
  p.gnuplot_output = "figure_%s_%04d.svg";
  p.gnuplot_with = "lines";
  p.gnuplot_surface = false;
  p.gnuplot_contour = true;

  real_t eps = .01;

  p.gnuplot_cntrparam = "levels incremental 299.95, 0.1, 300.55";
  p.gnuplot_cbrange = "[299.95 : 300.55]";
  p.gnuplot_cbtics = "300.05, 0.1, 300.45";
  p.gnuplot_palette = "defined ("
    "299.95 '#ffffff', "
    "300.05 '#ffffff', 300.05 '#993399', "
    "300.15 '#993399', 300.15 '#00CCFF', "
    "300.25 '#00CCFF', 300.25 '#66CC00', "
    "300.35 '#66CC00', 300.35 '#FC8727', "
    "300.45 '#FC8727', 300.45 '#FFFF00', "
    "300.55 '#FFFF00'"
  ")";
  p.gnuplot_term = "svg";
  p.prs_tol = 1e-7;
  p.grid_size = {nx, ny};
  """
  # runtime
  #plotting
plt.clf()
plt.imshow(mesh.T,origin='lower')
plt.colorbar()
plt.show()
"""
            template <class ct_params_t>
            class boussinesq_common : public libmpdataxx::solvers::detail::mpdata_rhs_vip_prs_gcrk<ct_params_t,1,0>
            {
                using parent_t = libmpdataxx::solvers::detail::mpdata_rhs_vip_prs_gcrk<ct_params_t,1,0>;

            public:
                using real_t = typename ct_params_t::real_t;

            protected:
                real_t g, Tht_ref;
                typename parent_t::arr_t& tht_e, & tht_abs;

                virtual void calc_full_tht(typename parent_t::arr_t&) = 0;

            public:
                struct rt_params_t : parent_t::rt_params_t
                {
                    real_t g = 9.81, Tht_ref = 0;
                };

                // ctor
                boussinesq_common(
                    typename parent_t::ctor_args_t args,
                    const rt_params_t& p
                ) :
                    parent_t(args, p),
                    g(p.g),
                    Tht_ref(p.Tht_ref),
                    tht_e(args.mem->tmp[__FILE__][0][0]),
                    tht_abs(args.mem->tmp[__FILE__][1][0])
                {
                    assert(Tht_ref != 0);
                    tht_abs(this->ijk) = 0;
                }

                static void alloc(typename parent_t::mem_t* mem, const int& n_iters)
                {
                    parent_t::alloc(mem, n_iters);
                    parent_t::alloc_tmp_sclr(mem, __FILE__, 1, "tht_e");
                    parent_t::alloc_tmp_sclr(mem, __FILE__, 1, "tht_abs");

                }
            };
        
            template <class ct_params_t>
            class boussinesq_expl : public
                boussinesq_common<ct_params_t>

            {
                using parent_t = boussinesq_common<ct_params_t>;
            public:
                using real_t = typename ct_params_t::real_t;

            protected:
                // member fields
                using ix = typename ct_params_t::ix;

                typename parent_t::arr_t& tmp1, & tmp2;


                // helpers for buoyancy forces
                template<class ijk_t>
                inline auto buoy_at_0(const ijk_t& ijk)
                {
                    return return_helper<rng_t>(
                        this->g * (this->state(ix::tht)(ijk) - this->tht_e(ijk)) / this->Tht_ref
                    );
                }

                template<class ijk_t>
                inline auto buoy_at_1(const ijk_t& ijk)
                {
                    return return_helper<rng_t>(
                        this->g * (
                            (this->state(ix::tht)(ijk)
                                 + real_t(0.5) * this->dt * this->tht_abs(ijk) * this->tht_e(ijk))
                            / (1 + real_t(0.5) * this->dt * this->tht_abs(ijk))
                            - this->tht_e(ijk)
                            ) / this->Tht_ref
                    );
                }

                void calc_full_tht(typename parent_t::arr_t& full_tht) final
                {
                    full_tht(this->ijk) = this->state(ix::tht)(this->ijk);
                }

                // explicit forcings
                void update_rhs(
                    libmpdataxx::arrvec_t<
                    typename parent_t::arr_t
                    >& rhs,
                    const real_t& dt,
                    const int& at
                ) {
                    parent_t::update_rhs(rhs, dt, at);

                    const auto& tht = this->state(ix::tht);
                    const auto& ijk = this->ijk;

                    auto ix_w = this->vip_ixs[ct_params_t::n_dims - 1];

                    switch (at)
                    {
                    case (0):
                    {
                        rhs.at(ix::tht)(ijk) +=  - this->tht_abs(ijk) * (tht(ijk) - this->tht_e(ijk));

                        rhs.at(ix_w)(ijk) += buoy_at_0(ijk);


                        break;
                    }
                    case (1):
                    {
                        rhs.at(ix::tht)(ijk) +=  - this->tht_abs(ijk) * (
                            (tht(ijk) + real_t(0.5) * this->dt * this->tht_abs(ijk) * this->tht_e(ijk))
                            / (1 + real_t(0.5) * this->dt * this->tht_abs(ijk))
                            - this->tht_e(ijk)
                            );

                        rhs.at(ix_w)(ijk) += buoy_at_1(ijk);
                    }
                    }
                }
"""
"""
            public:

                // ctor
                boussinesq_expl(
                    typename parent_t::ctor_args_t args,
                    const typename parent_t::rt_params_t& p
                ) :
                    parent_t(args, p),
                    tmp1(args.mem->tmp[__FILE__][0][0]),
                    tmp2(args.mem->tmp[__FILE__][0][1])
                {}

                static void alloc(typename parent_t::mem_t* mem, const int& n_iters)
                {
                    parent_t::alloc(mem, n_iters);
                    parent_t::alloc_tmp_sclr(mem, __FILE__, 2); // tmp1, tmp2
                }
            };

"""