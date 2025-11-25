<img src="https://raw.githubusercontent.com/open-atmos/PyMPDATA/main/.github/pympdata_logo.svg" width=100 height=113 alt="pympdata logo">
# Introduction
PyMPDATA is a high-performance [Numba-accelerated](https://doi.org/10.1145/2833157.2833162) Pythonic implementation of the MPDATA
  algorithm of [Smolarkiewicz et al.](https://doi.org/10.1016%2F0021-9991%2884%2990121-9)
  used in geophysical fluid dynamics and beyond.
MPDATA numerically solves generalised transport equations -
  partial differential equations used to model conservation/balance laws, scalar-transport problems,
  convection-diffusion phenomena.
As of the current version, PyMPDATA supports homogeneous transport
  in 1D, 2D and 3D using structured meshes, optionally
  generalised by employment of a Jacobian of coordinate transformation or a fluid density profile.
PyMPDATA includes implementation of a set of MPDATA variants including
  the non-oscillatory option, infinite-gauge, divergent-flow, double-pass donor cell (DPDC) and
  third-order-terms options.
It also features support for integration of Fickian-terms in advection-diffusion
  problems using the pseudo-transport velocity approach.
In 2D and 3D simulations, domain-decomposition is used for multi-threaded parallelism.

PyMPDATA is engineered purely in Python targeting both performance and usability,
    the latter encompassing research users', developers' and maintainers' perspectives.
From researcher's perspective, PyMPDATA offers hassle-free installation on multitude
  of platforms including Linux, OSX and Windows, and eliminates compilation stage
  from the perspective of the user.
From developers' and maintainers' perspective, PyMPDATA offers a suite of unit tests,
  multi-platform continuous integration setup,
  seamless integration with Python development aids including debuggers and profilers.

PyMPDATA design features
  a custom-built multi-dimensional Arakawa-C grid layer allowing
  to concisely represent multi-dimensional stencil operations on both
  scalar and vector fields.
The grid layer is built on top of NumPy's ndarrays (using "C" ordering)
  using the Numba's ``@jit(nopython=True)`` functionality for high-performance array traversals.
It enables one to code once for multiple dimensions, and automatically
  handles (and hides from the user) any halo-filling logic related with boundary conditions.
Numba ``prange()`` functionality is used for implementing multi-threading
  (it offers analogous functionality to OpenMP parallel loop execution directives).
The Numba's deviation from Python semantics rendering [closure variables
  as compile-time constants](https://numba.readthedocs.io/en/stable/reference/pysemantics.html?highlight=deviations#global-and-closure-variables)
  is extensively exploited within ``PyMPDATA``
  code base enabling the just-in-time compilation to benefit from
  information on domain extents, algorithm variant used and problem
  characteristics (e.g., coordinate transformation used, or lack thereof).

# Tutorial (in Python, Julia, Rust and Matlab)
## Options class

The [``Options``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/options.html) class
groups both algorithm variant options as well as some implementation-related
flags that need to be set at the first place. All are set at the time
of instantiation using the following keyword arguments of the constructor
(all having default values indicated below):
- ``n_iters: int = 2``: number of iterations (2 means upwind + one corrective iteration)
- ``infinite_gauge: bool = False``: flag enabling the infinite-gauge option (does not maintain sign of the advected field, thus in practice implies switching flux corrected transport on)
- ``divergent_flow: bool = False``: flag enabling divergent-flow terms when calculating antidiffusive velocity
- ``nonoscillatory: bool = False``: flag enabling the non-oscillatory or monotone variant (a.k.a flux-corrected transport option, FCT)
- ``third_order_terms: bool = False``: flag enabling third-order terms
- ``epsilon: float = 1e-15``: value added to potentially zero-valued denominators
- ``non_zero_mu_coeff: bool = False``: flag indicating if code for handling the Fickian term is to be optimised out
- ``DPDC: bool = False``: flag enabling double-pass donor cell option (recursive pseudovelocities)
- ``dimensionally_split: bool = False``: flag disabling cross-dimensional terms in antidiffusive velocity
- ``dtype: np.floating = np.float64``: floating point precision

For a discussion of the above options, see e.g., [Smolarkiewicz & Margolin 1998](https://doi.org/10.1006/jcph.1998.5901),
[Jaruga, Arabas et al. 2015](https://doi.org/10.5194/gmd-8-1005-2015) and [Olesik et al. 2020](https://doi.org/10.5194/gmd-15-3879-2022)
(the last with examples using PyMPDATA).

In most use cases of PyMPDATA, the first thing to do is to instantiate the
[``Options``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/options.html) class
with arguments suiting the problem at hand, e.g.:

<details>
<summary>Julia code (click to expand)</summary>
```Julia
using Pkg
Pkg.add("PyCall")
using PyCall
Options = pyimport("PyMPDATA").Options
options = Options(n_iters=2)
```
</details>

<details>
<summary>Matlab code (click to expand)</summary>
```Matlab
Options = py.importlib.import_module('PyMPDATA').Options;
options = Options(pyargs('n_iters', 2));
```
</details>

<details>
<summary>Rust code (click to expand)</summary>
```Rust
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyTuple};

fn main() -> PyResult<()> {
  Python::with_gil(|py| {
    let options_args = [("n_iters", 2)].into_py_dict_bound(py);
    let options = py.import_bound("PyMPDATA")?.getattr("Options")?.call((), Some(&options_args))?;
```
</details>

<details open>
<summary>Python code (click to expand)</summary>
```Python
from PyMPDATA import Options
options = Options(n_iters=2)
```
</details>

## Arakawa-C grid layer and boundary conditions

In PyMPDATA, the solution domain is assumed to extend from the
first cell's boundary to the last cell's boundary (thus the
first scalar field value is at $\[\Delta x/2, \Delta y/2\]$.
The [``ScalarField``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/scalar_field.html)
and [``VectorField``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/vector_field.html) classes implement the
[Arakawa-C staggered grid](https://en.wikipedia.org/wiki/Arakawa_grids#Arakawa_C-grid) logic
in which:
- scalar fields are discretised onto cell centres (one value per cell),
- vector field components are discretised onto cell walls.

The schematic of the employed grid/domain layout in two dimensions is given below
(with the Python code snippet generating the figure as a part of CI workflow):
![plot](https://github.com/open-atmos/PyMPDATA/releases/download/tip/readme_grid.png)

<details>
<summary>Python code (click to expand)</summary>
```Python
import numpy as np
from matplotlib import pyplot

dx, dy = .2, .3
grid = (10, 5)

pyplot.scatter(*np.mgrid[
        dx / 2 : grid[0] * dx : dx,
        dy / 2 : grid[1] * dy : dy
    ], color='red',
    label='scalar-field values at cell centres'
)
pyplot.quiver(*np.mgrid[
        0 : (grid[0]+1) * dx : dx,
        dy / 2 : grid[1] * dy : dy
    ], 1, 0, pivot='mid', color='green', width=.005,
    label='vector-field x-component values at cell walls'
)
pyplot.quiver(*np.mgrid[
        dx / 2 : grid[0] * dx : dx,
        0: (grid[1] + 1) * dy : dy
    ], 0, 1, pivot='mid', color='blue', width=.005,
    label='vector-field y-component values at cell walls'
)
pyplot.xticks(np.linspace(0, grid[0]*dx, grid[0]+1))
pyplot.yticks(np.linspace(0, grid[1]*dy, grid[1]+1))
pyplot.title(f'staggered grid layout (grid={grid}, dx={dx}, dy={dy})')
pyplot.xlabel('x')
pyplot.ylabel('y')
pyplot.legend(bbox_to_anchor=(.1, -.1), loc='upper left', ncol=1)
pyplot.grid()
pyplot.savefig('readme_grid.png')
```
</details>

The ``__init__`` methods of
[``ScalarField``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/scalar_field.html)
and [``VectorField``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/vector_field.html)
have the following signatures:
- [``ScalarField(data: np.ndarray, halo: int, boundary_conditions)``](https://github.com/open-atmos/PyMPDATA/blob/main/PyMPDATA/scalar_field.py)
- [``VectorField(data: Tuple[np.ndarray, ...], halo: int, boundary_conditions)``](https://github.com/open-atmos/PyMPDATA/blob/main/PyMPDATA/vector_field.py)
The ``data`` parameters are expected to be Numpy arrays or tuples of Numpy arrays, respectively.
The ``halo`` parameter is the extent of ghost-cell region that will surround the
data and will be used to implement boundary conditions.
Its value (in practice 1 or 2) is
dependent on maximal stencil extent for the MPDATA variant used and
can be easily obtained using the ``Options.n_halo`` property.

As an example, the code below shows how to instantiate a scalar
and a vector field given a 2D constant-velocity problem,
using a grid of 24x24 points, Courant numbers of -0.5 and -0.25
in "x" and "y" directions, respectively, with periodic boundary
conditions and with an initial Gaussian signal in the scalar field
(settings as in Fig.&nbsp;5 in [Arabas et al. 2014](https://doi.org/10.3233/SPR-140379)):

<details>
<summary>Julia code (click to expand)</summary>
```Julia
ScalarField = pyimport("PyMPDATA").ScalarField
VectorField = pyimport("PyMPDATA").VectorField
Periodic = pyimport("PyMPDATA.boundary_conditions").Periodic

nx, ny = 24, 24
Cx, Cy = -.5, -.25
idx = CartesianIndices((nx, ny))
halo = options.n_halo
advectee = ScalarField(
    data=exp.(
        -(getindex.(idx, 1) .- .5 .- nx/2).^2 / (2*(nx/10)^2)
        -(getindex.(idx, 2) .- .5 .- ny/2).^2 / (2*(ny/10)^2)
    ),
    halo=halo,
    boundary_conditions=(Periodic(), Periodic())
)
advector = VectorField(
    data=(fill(Cx, (nx+1, ny)), fill(Cy, (nx, ny+1))),
    halo=halo,
    boundary_conditions=(Periodic(), Periodic())
)
```
</details>

<details>
<summary>Matlab code (click to expand)</summary>
```Matlab
ScalarField = py.importlib.import_module('PyMPDATA').ScalarField;
VectorField = py.importlib.import_module('PyMPDATA').VectorField;
Periodic = py.importlib.import_module('PyMPDATA.boundary_conditions').Periodic;

nx = int32(24);
ny = int32(24);

Cx = -.5;
Cy = -.25;

[xi, yi] = meshgrid(double(0:1:nx-1), double(0:1:ny-1));

halo = options.n_halo;
advectee = ScalarField(pyargs(...
    'data', py.numpy.array(exp( ...
        -(xi+.5-double(nx)/2).^2 / (2*(double(nx)/10)^2) ...
        -(yi+.5-double(ny)/2).^2 / (2*(double(ny)/10)^2) ...
    )), ...
    'halo', halo, ...
    'boundary_conditions', py.tuple({Periodic(), Periodic()}) ...
));
advector = VectorField(pyargs(...
    'data', py.tuple({ ...
        Cx * py.numpy.ones(int32([nx+1 ny])), ...
        Cy * py.numpy.ones(int32([nx ny+1])) ...
     }), ...
    'halo', halo, ...
    'boundary_conditions', py.tuple({Periodic(), Periodic()}) ...
));
```
</details>

<details>
<summary>Rust code (click to expand)</summary>
```Rust
    let vector_field = py.import_bound("PyMPDATA")?.getattr("VectorField")?;
    let scalar_field = py.import_bound("PyMPDATA")?.getattr("ScalarField")?;
    let periodic = py.import_bound("PyMPDATA.boundary_conditions")?.getattr("Periodic")?;

    let nx_ny = [24, 24];
    let cx_cy = [-0.5, -0.25];
    let boundary_con = PyTuple::new_bound(py, [periodic.call0()?, periodic.call0()?]).into_any();
    let halo = options.getattr("n_halo")?;

    let indices = PyDict::new_bound(py);
    Python::run_bound(py, &format!(r#"
import numpy as np
nx, ny = {}, {}
xi, yi = np.indices((nx, ny), dtype=float)
data=np.exp(
  -(xi+.5-nx/2)**2 / (2*(ny/10)**2)
  -(yi+.5-nx/2)**2 / (2*(ny/10)**2)
)
    "#, nx_ny[0], nx_ny[1]), None, Some(&indices)).unwrap();

    let advectee_arg = vec![("data", indices.get_item("data")?), ("halo", Some(halo.clone())), ("boundary_conditions", Some(boundary_con))].into_py_dict_bound(py);
    let advectee = scalar_field.call((), Some(&advectee_arg))?;
    let full = PyDict::new_bound(py);
    Python::run_bound(py, &format!(r#"
import numpy as np
nx, ny = {}, {}
Cx, Cy = {}, {}
data = (np.full((nx + 1, ny), Cx), np.full((nx, ny + 1), Cy))
    "#, nx_ny[0], nx_ny[1], cx_cy[0], cx_cy[1]), None, Some(&full)).unwrap();
    let boundary_con = PyTuple::new_bound(py, [periodic.call0()?, periodic.call0()?]).into_any();
    let advector_arg = vec![("data", full.get_item("data")?), ("halo", Some(halo.clone())), ("boundary_conditions", Some(boundary_con))].into_py_dict_bound(py);
    let advector = vector_field.call((), Some(&advector_arg))?;
```
</details>

<details open>
<summary>Python code (click to expand)</summary>
```Python
from PyMPDATA import ScalarField
from PyMPDATA import VectorField
from PyMPDATA.boundary_conditions import Periodic
import numpy as np

nx, ny = 24, 24
Cx, Cy = -.5, -.25
halo = options.n_halo

xi, yi = np.indices((nx, ny), dtype=float)
advectee = ScalarField(
  data=np.exp(
    -(xi+.5-nx/2)**2 / (2*(nx/10)**2)
    -(yi+.5-ny/2)**2 / (2*(ny/10)**2)
  ),
  halo=halo,
  boundary_conditions=(Periodic(), Periodic())
)
advector = VectorField(
  data=(np.full((nx + 1, ny), Cx), np.full((nx, ny + 1), Cy)),
  halo=halo,
  boundary_conditions=(Periodic(), Periodic())
)
```
</details>

Note that the shapes of arrays representing components
of the velocity field are different than the shape of
the scalar field array due to employment of the staggered grid.

Besides the exemplified [``Periodic``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/boundary_conditions/periodic.html) class representing
periodic boundary conditions, PyMPDATA supports
[``Extrapolated``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/boundary_conditions/extrapolated.html),
[``Constant``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/boundary_conditions/constant.html) and
[``Polar``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/boundary_conditions/polar.html)
boundary conditions.

## Stepper

The logic of the MPDATA iterative solver is represented
in PyMPDATA by the [``Stepper``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/stepper.html) class.

When instantiating the [``Stepper``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/stepper.html), the user has a choice
of either supplying just the  number of dimensions or specialising the stepper for a given grid:
<details>
<summary>Julia code (click to expand)</summary>
```Julia
Stepper = pyimport("PyMPDATA").Stepper

stepper = Stepper(options=options, n_dims=2)
```
</details>
<details>
<summary>Matlab code (click to expand)</summary>
```Matlab
Stepper = py.importlib.import_module('PyMPDATA').Stepper;

stepper = Stepper(pyargs(...
  'options', options, ...
  'n_dims', int32(2) ...
));
```
</details>
<details open>

<details>
<summary>Rust code (click to expand)</summary>
```Rust
let n_dims: i32 = 2;
let stepper_arg = PyDict::new_bound(py);
let _ = PyDictMethods::set_item(&stepper_arg, "options", &options);
let _ = PyDictMethods::set_item(&stepper_arg, "n_dims", &n_dims);
```
</details>
<details>
<summary>Python code (click to expand)</summary>
```Python
from PyMPDATA import Stepper

stepper = Stepper(options=options, n_dims=2)
```
</details>
or
<details>
<summary>Julia code (click to expand)</summary>
```Julia
stepper = Stepper(options=options, grid=(nx, ny))
```
</details>

<details>
<summary>Matlab code (click to expand)</summary>
```Matlab
stepper = Stepper(pyargs(...
  'options', options, ...
  'grid', py.tuple({nx, ny}) ...
));
```
</details>

<details>
<summary>Rust code (click to expand)</summary>
```Rust
 let _stepper_arg_alternative = vec![("options", &options), ("grid", &PyTuple::new_bound(py, nx_ny).into_any())].into_py_dict_bound(py);
 let stepper_ = py.import_bound("PyMPDATA")?.getattr("Stepper")?;
 let stepper = stepper_.call((), Some(&stepper_arg))?; //or use stepper args alternative
```
</details>

<details open>
<summary>Python code (click to expand)</summary>
```Python
stepper = Stepper(options=options, grid=(nx, ny))
```
</details>

In the latter case, noticeably
faster execution can be expected, however the resultant
stepper is less versatile as bound to the given grid size.
If number of dimensions is supplied only, the integration
might take longer, yet same instance of the
stepper can be used for different grids.

Since creating an instance of the [``Stepper``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/stepper.html) class
involves time-consuming compilation of the algorithm code,
the class is equipped with a cache logic - subsequent
calls with same arguments return references to previously
instantiated objects. Instances of [``Stepper``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/stepper.html) contain no
mutable data and are (thread-)safe to be reused.

The init method of [``Stepper``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/stepper.html) has an optional
``non_unit_g_factor`` argument which is a Boolean flag
enabling handling of the G factor term which can be used to
represent coordinate transformations and/or variable fluid density.

Optionally, the number of threads to use for domain decomposition
in the first (non-contiguous) dimension during 2D and 3D calculations
may be specified using the optional ``n_threads`` argument with a
default value of ``numba.get_num_threads()``. The multi-threaded
logic of PyMPDATA depends thus on settings of numba, namely on the
selected threading layer (either via ``NUMBA_THREADING_LAYER`` env
var or via ``numba.config.THREADING_LAYER``) and the selected size of the
thread pool (``NUMBA_NUM_THREADS`` env var or ``numba.config.NUMBA_NUM_THREADS``).


## Solver

Instances of the [``Solver``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/solver.html) class are used to control
the integration and access solution data. During instantiation,
additional memory required by the solver is
allocated according to the options provided.

The only method of the [``Solver``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/solver.html) class besides the
init is [``advance(n_steps, mu_coeff, ...)``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/solver.html#Solver.advance)
which advances the solution by ``n_steps`` timesteps, optionally
taking into account a given diffusion coefficient ``mu_coeff``.

Solution state is accessible through the ``Solver.advectee`` property.
Multiple solver[s] can share a single stepper, e.g., as exemplified in the shallow-water system solution in the examples package.

Continuing with the above code snippets, instantiating
a solver and making 75 integration steps looks as follows:
<details>
<summary>Julia code (click to expand)</summary>
```Julia
Solver = pyimport("PyMPDATA").Solver
solver = Solver(stepper=stepper, advectee=advectee, advector=advector)
solver.advance(n_steps=75)
state = solver.advectee.get()
```
</details>

<details>
<summary>Matlab code (click to expand)</summary>
```Matlab
Solver = py.importlib.import_module('PyMPDATA').Solver;
solver = Solver(pyargs('stepper', stepper, 'advectee', advectee, 'advector', advector));
solver.advance(pyargs('n_steps', 75));
state = solver.advectee.get();
```
</details>

<details>
<summary>Rust code (click to expand)</summary>
```Rust
    let solver_ = py.import_bound("PyMPDATA")?.getattr("Solver")?;
    let solver = solver_.call((), Some(&vec![("stepper", stepper), ("advectee", advectee), ("advector", advector)].into_py_dict_bound(py)))?;
    let _state_0 = solver.getattr("advectee")?.getattr("get")?.call0()?.getattr("copy")?.call0()?;
    solver.getattr("advance")?.call((), Some(&vec![("n_steps", 75)].into_py_dict_bound(py)))?;
    let _state = solver.getattr("advectee")?.getattr("get")?.call0()?;
    Ok(())
  })
}
```
</details>

<details open>
<summary>Python code (click to expand)</summary>
```Python
from PyMPDATA import Solver

solver = Solver(stepper=stepper, advectee=advectee, advector=advector)
state_0 = solver.advectee.get().copy()
solver.advance(n_steps=75)
state = solver.advectee.get()
```
</details>

Now let's plot the results using `matplotlib` roughly as in Fig.&nbsp;5 in [Arabas et al. 2014](https://doi.org/10.3233/SPR-140379):
<details>
<summary>Python code (click to expand)</summary>
```Python
def plot(psi, zlim, norm=None):
    xi, yi = np.indices(psi.shape)
    fig, ax = pyplot.subplots(subplot_kw={"projection": "3d"})
    pyplot.gca().plot_wireframe(
        xi+.5, yi+.5,
        psi, color='red', linewidth=.5
    )
    ax.set_zlim(zlim)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor('black')
        axis.pane.set_alpha(1)
    ax.grid(False)
    ax.set_zticks([])
    ax.set_xlabel('x/dx')
    ax.set_ylabel('y/dy')
    ax.set_proj_type('ortho')
    cnt = ax.contourf(xi+.5, yi+.5, psi, zdir='z', offset=-1, norm=norm)
    cbar = pyplot.colorbar(cnt, pad=.1, aspect=10, fraction=.04)
    return cbar.norm

zlim = (-1, 1)
norm = plot(state_0, zlim)
pyplot.savefig('readme_gauss_0.png')
plot(state, zlim, norm)
pyplot.savefig('readme_gauss.png')
```
</details>

![plot](https://github.com/open-atmos/PyMPDATA/releases/download/tip/readme_gauss_0.png)
![plot](https://github.com/open-atmos/PyMPDATA/releases/download/tip/readme_gauss.png)

# Debugging

PyMPDATA relies heavily on Numba to provide high-performance
number crunching operations. Arguably, one of the key advantage
of embracing Numba is that it can be easily switched off. This
brings multiple-order-of-magnitude drop in performance, yet
it also make the entire code of the library susceptible to
interactive debugging, one way of enabling it is by setting the
following environment variable before importing PyMPDATA:
<details>
<summary>Julia code (click to expand)</summary>
```Julia
ENV["NUMBA_DISABLE_JIT"] = "1"
```
</details>

<details>
<summary>Matlab code (click to expand)</summary>
```Matlab
setenv('NUMBA_DISABLE_JIT', '1');
```
</details>

<details open>
<summary>Python code (click to expand)</summary>
```Python
import os
os.environ["NUMBA_DISABLE_JIT"] = "1"
```
</details>

# Contributing, reporting issues, seeking support

See [README.md](https://github.com/open-atmos/PyMPDATA/tree/main/README.md).

# Other open-source MPDATA implementations:
- mpdat_2d in babyEULAG (FORTRAN)
  https://github.com/igfuw/bE_SDs/blob/master/babyEULAG.SDs.for#L741
- mpdata-oop (C++, Fortran, Python)
  https://github.com/igfuw/mpdata-oop
- apc-llc/mpdata (C++)
  https://github.com/apc-llc/mpdata
- libmpdata++ (C++):
  https://github.com/igfuw/libmpdataxx
- AtmosFOAM:
  https://github.com/AtmosFOAM/AtmosFOAM/tree/947b192f69d973ea4a7cfab077eb5c6c6fa8b0cf/applications/solvers/advection/MPDATAadvectionFoam
- advectionHPCtester (Fortran, CUDA, C preprocessor):
  https://zenodo.org/records/8178549
- PISM (C++, 2D only):
  https://github.com/pism/pism/blob/main/src/geometry/MPDATA2.cc
- ROMS (Fortran):
  https://github.com/myroms/roms/blob/main/ROMS/Nonlinear/mpdata_adiff.F
- sbPOM (Fortran):
  https://github.com/RinceWND/extPOM/blob/master/pom/solver.f
- TourOfJulia (Julia):
  https://github.com/themantra108/TourOfJulia.jl/blob/master/06b_smolarkiewicz.jl

# Other Python packages for solving hyperbolic transport equations

- PyPDE: https://pypi.org/project/PyPDE/
- FiPy: https://pypi.org/project/FiPy/
- ader: https://pypi.org/project/ader/
- centpy: https://pypi.org/project/centpy/
- mattflow: https://pypi.org/project/mattflow/
- FastFD: https://pypi.org/project/FastFD/
- Pyclaw: https://www.clawpack.org/pyclaw/
