# PyMPDATA

[![Python 3](https://img.shields.io/static/v1?label=Python&logo=Python&color=3776AB&message=3)](https://www.python.org/)
[![LLVM](https://img.shields.io/static/v1?label=LLVM&logo=LLVM&color=gold&message=Numba)](https://www.numba.org)
[![Linux OK](https://img.shields.io/static/v1?label=Linux&logo=Linux&color=yellow&message=%E2%9C%93)](https://en.wikipedia.org/wiki/Linux)
[![macOS OK](https://img.shields.io/static/v1?label=macOS&logo=Apple&color=silver&message=%E2%9C%93)](https://en.wikipedia.org/wiki/macOS)
[![Windows OK](https://img.shields.io/static/v1?label=Windows&logo=Windows&color=white&message=%E2%9C%93)](https://en.wikipedia.org/wiki/Windows)
[![Jupyter](https://img.shields.io/static/v1?label=Jupyter&logo=Jupyter&color=f37626&message=%E2%9C%93)](https://jupyter.org/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/atmos-cloud-sim-uj/PyMPDATA/graphs/commit-activity)
[![OpenHub](https://www.openhub.net/p/atmos-cloud-sim-uj-PyMPDATA/widgets/project_thin_badge?format=gif)](https://www.openhub.net/p/atmos-cloud-sim-uj-PyMPDATA)
[![status](https://joss.theoj.org/papers/10e7361e43785dbb1b3d659c5b01757a/status.svg)](https://joss.theoj.org/papers/10e7361e43785dbb1b3d659c5b01757a)
[![DOI](https://zenodo.org/badge/225610671.svg)](https://zenodo.org/badge/latestdoi/225610671)     
[![EU Funding](https://img.shields.io/static/v1?label=EU%20Funding%20by&color=103069&message=FNP&logoWidth=25&logo=image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC4AAAAeCAYAAABTwyyaAAAEzklEQVRYw9WYS2yUVRiGn3P5ZzozpZ3aUsrNgoKlKBINmkhpCCwwxIAhsDCpBBIWhmCMMYTEhSJ4i9EgBnSBEm81MRrFBhNXEuUSMCopiRWLQqEGLNgr085M5//POS46NNYFzHQ6qGc1i5nzP/P973m/9ztCrf7A8T9csiibCocUbvTzfxLcAcaM3cY3imXz25lT3Y34G7gQYAKV3+bFAHcATlBTPogJNADG92iY28FHW97kyPbnuW/W7xgzAhukQ9xe04PJeOT0HkQRwK0TlEeGWb/kOO9v3kdD3a8YK9GhDMfa6mg9fxunOm/lWPtcpDI4K7n/jnN8+uQbrFrUSiwU/DtSEUB/MsKKBT+zYslJqiYNgVE4JwhHkzy86wlWvrKVWDSZ/YFjZlU39yw4y/rGoyQGowWB67zl4QQue+jssMdXrQvZ/00jyeHwqCgDKwnsiJjSvkYAxsG5K9WsenYbJdqAtAjhCIxCSZt/4fK1w5A2WCvxrUAKCHwNVoA2aGmvq11jJQQapEXrgMBKqmJJugejKGWLIxXrBPFoigfv/omd675gRkU/xgqUDlAhH3UDaAAlLSqUQekAYyVTyhLs3tDMsvntlIYzOFcEcOcEGd9jx9oDbGs6QO0t/Tijxi9S4bhzxiWaVh5m94Zm0n7oui4ybo0raUlcncQnxx+g+WgDF/vLoYDmoqSl/dJUnt7XRCoTZjij0Z6Pc2LiNS4EBBkNvoeOJXN+yPWWSZeANOhwJq/98nKVwNdoL8B5AROxBKBL0gjh8DMhdCh3eJnrA0yqhLpplwmyup6IajvAOIGfKGVx3VmCRGnOMpe5QAdG0bT8CAeeep0d6z6nqjSJnQiZWEllLMWrmz6k+fE9rGk8MVqYgsGv5ZH2i1Opr+9kajzB5d74hKQ+KS3d/WVMLhtgdu1lriRiOR/4nDVunaR24x7qp3UV5Cb/fJvC83nv26W81LIK58SYNFmwq4hsGx/5BwKlzYRma2NUthgOJSew4i7ru9nJYCQF5tApb2yvjiDQKJV/IfJKh0o6qssSLKv/jcAoRKHQQzE2Lj2OMV5OkWFc4MZIpsev8uXWXRx6ZicbGk8QZLxxgwe+x/rlR3h3816+f2E7lbEU+ZDn3vKVpePCdFovzCISHqbl5EIoQOteKMPB1rto65zNyfOz+KOrGl06lHPQyi/WOohH0/T0l1MZH6A3GUEKl7Pmr2la6wBrBWWRDP2DUcqjKVKBGom9RZmABAykwnglafpSJSPQvsfiOR0EQ7ExVmazA8cY6N4K1iw6RdAXRwi4mgrheT5Dvs4LeuS81a15Ll/3dQisFVSVpnj7sf1sX/sZvhAc+6UOrQyBVUQ8gx/orFmDsZqtaw/y1qZ9zKjp5vDpenyjcNe+cLNmTiUdf/bEOddVQ0VpgsOn54ET+EYxvWKALSu+5tGG76it7MNaiZKGQ23zCIcMfUMxBnrjN3fmHHvCAlp+vJcXWx6itqoXpAEnUNLx8iMfo5Xh1i17R3PJYCpC2cZ3qK3sQ8WGEDDuXlAQuFKGHzpmopXhTNfk0bmxs7uC1w6uJul79AxFkMIiBJy5UoUWjrZLU5DCFdTARDHuDqVw+OkSwI0MCEW4gtNF2BPrBCo8fKNbtILWX9aUDqFqHnn7AAAAAElFTkSuQmCC)](https://www.fnp.org.pl/en/)
[![PL Funding](https://img.shields.io/static/v1?label=PL%20Funding%20by&color=d21132&message=NCN&logoWidth=25&logo=image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAANCAYAAACpUE5eAAAABmJLR0QA/wD/AP+gvaeTAAAAKUlEQVQ4jWP8////fwYqAiZqGjZqIHUAy4dJS6lqIOMdEZvRZDPcDQQAb3cIaY1Sbi4AAAAASUVORK5CYII=)](https://www.ncn.gov.pl/?language=en)
[![Copyright](https://img.shields.io/static/v1?label=Copyright&color=249fe2&message=Jagiellonian%20University&)](https://en.uj.edu.pl/)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.html)

[![Github Actions Build Status](https://github.com/atmos-cloud-sim-uj/PyMPDATA/actions/workflows/tests+pypi.yml/badge.svg?branch=main)](https://github.com/atmos-cloud-sim-uj/PyMPDATA/actions)
[![Appveyor Build status](http://ci.appveyor.com/api/projects/status/github/atmos-cloud-sim-uj/PyMPDATA?branch=main&svg=true)](https://ci.appveyor.com/project/slayoo/pympdata/branch/main)
[![Coverage Status](https://codecov.io/gh/atmos-cloud-sim-uj/PyMPDATA/branch/main/graph/badge.svg)](https://codecov.io/github/atmos-cloud-sim-uj/PyMPDATA?branch=main)
[![Github Actions Status](https://github.com/atmos-cloud-sim-uj/PyMPDATA/actions/workflows/pylint.yml/badge.svg?branch=main)](https://github.com/atmos-cloud-sim-uj/PyMPDATA/actions/workflows/pylint.yml)    
[![GitHub issues](https://img.shields.io/github/issues-pr/atmos-cloud-sim-uj/PyMPDATA.svg?logo=github&logoColor=white)](https://github.com/atmos-cloud-sim-uj/PyMPDATA/pulls?q=)
[![GitHub issues](https://img.shields.io/github/issues-pr-closed/atmos-cloud-sim-uj/PyMPDATA.svg?logo=github&logoColor=white)](https://github.com/atmos-cloud-sim-uj/PyMPDATA/pulls?q=is:closed)    
[![GitHub issues](https://img.shields.io/github/issues/atmos-cloud-sim-uj/PyMPDATA.svg?logo=github&logoColor=white)](https://github.com/atmos-cloud-sim-uj/PyMPDATA/issues?q=)
[![GitHub issues](https://img.shields.io/github/issues-closed/atmos-cloud-sim-uj/PyMPDATA.svg?logo=github&logoColor=white)](https://github.com/atmos-cloud-sim-uj/PyMPDATA/issues?q=)    
[![PyPI version](https://badge.fury.io/py/PyMPDATA.svg)](https://pypi.org/project/PyMPDATA)
[![API docs](https://img.shields.io/badge/API_docs-pdoc3-blue.svg)](https://atmos-cloud-sim-uj.github.io/PyMPDATA/)

PyMPDATA is a high-performance Numba-accelerated Pythonic implementation of the MPDATA 
  algorithm of Smolarkiewicz et al. used in geophysical fluid dynamics and beyond.
MPDATA numerically solves generalised transport equations -
  partial differential equations used to model conservation/balance laws, scalar-transport problems,
  convection-diffusion phenomena.
As of the current version, PyMPDATA supports homogeneous transport
  in 1D, 2D and 3D using structured meshes, optionally
  generalised by employment of a Jacobian of coordinate transformation. 
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
  using the Numba's ``@njit`` functionality for high-performance array traversals.
It enables one to code once for multiple dimensions, and automatically
  handles (and hides from the user) any halo-filling logic related with boundary conditions.
Numba ``prange()`` functionality is used for implementing multi-threading 
  (it offers analogous functionality to OpenMP parallel loop execution directives).
The Numba's deviation from Python semantics rendering [closure variables
  as compile-time constants](https://numba.pydata.org/numba-doc/dev/reference/pysemantics.html#global-and-closure-variables)
  is extensively exploited within ``PyMPDATA``
  code base enabling the just-in-time compilation to benefit from 
  information on domain extents, algorithm variant used and problem
  characteristics (e.g., coordinate transformation used, or lack thereof).
A separate project called [``numba-mpi``](https://pypi.org/project/numba-mpi) 
  has been developed with the intention to 
  set the stage for future MPI distributed memory parallelism in PyMPDATA.

The [``PyMPDATA-examples``](https://pypi.org/project/PyMPDATA-examples/) 
  package covers a set of examples presented in the form of Jupyer notebooks
  offering single-click deployment in the cloud using [mybinder.org](https://mybinder.org)
  or using [colab.research.google.com](https://colab.research.google.com/).
The examples reproduce results from several published
  works on MPDATA and its applications, and provide a validation of the implementation
  and its performance.
 
## Dependencies and installation

To install PyMPDATA, one may use: ``pip install PyMPDATA`` (or 
``pip install git+https://github.com/atmos-cloud-sim-uj/PyMPDATA.git`` to get updates beyond the latest release).
PyMPDATA depends on ``NumPy`` and ``Numba``.

Running the tests shipped with the package requires additional packages listed in the 
[test-time-requirements.txt](https://github.com/atmos-cloud-sim-uj/PyMPDATA/blob/main/test-time-requirements.txt) file
(which include ``PyMPDATA-examples``, see below).

## Examples (Jupyter notebooks reproducing results from literature):

PyMPDATA examples are hosted in a separate repository and constitute 
the [``PyMPDATA_examples``](https://github.com/atmos-cloud-sim-uj/PyMPDATA-examples) package.
The examples have additional dependencies listed in [``PyMPDATA_examples`` package ``setup.py``](https://github.com/atmos-cloud-sim-uj/PyMPDATA-examples/blob/main/setup.py) file.
Running the examples requires the ``PyMPDATA_examples`` package to be installed.
Since the examples package includes Jupyter notebooks (and their execution requires write access), the suggested install and launch steps are:
```
git clone https://github.com/atmos-cloud-sim-uj/PyMPDATA-examples.git
cd PyMPDATA-examples
pip install -e .
jupyter-notebook
```
Alternatively, one can also install the examples package from pypi.org by using ``pip install PyMPDATA-examples``.
  
## Package structure and API:

In short, PyMPDATA numerically solves the following equation:

![\partial_t (G \psi) + \nabla \cdot (Gu \psi) + \mu \Delta (G \psi) = 0](https://render.githubusercontent.com/render/math?math=%5Cpartial_t%20(G%20%5Cpsi)%20%2B%20%5Cnabla%20%5Ccdot%20(Gu%20%5Cpsi)%20%2B%20%5Cmu%20%5CDelta%20%28G%20%5Cpsi%29%20%3D%200)

where scalar field ![\psi](https://render.githubusercontent.com/render/math?math=%5Cpsi) is referred to as the advectee,
vector field u is referred to as advector, and the G factor corresponds to optional coordinate transformation.
The inclusion of the Fickian diffusion term is optional and is realised through modification of the
advective velocity field with MPDATA handling both the advection and diffusion (for discussion
see, e.g. [Smolarkiewicz and Margolin 1998](https://doi.org/10.1006/jcph.1998.5901), sec. 3.5, par. 4).

The key classes constituting the PyMPDATA interface are summarised below with code
snippets exemplifying usage of PyMPDATA from Python, Julia and Matlab.

A [pdoc-generated](https://pdoc3.github.io/pdoc) documentation of PyMPDATA public API is maintained at: [https://atmos-cloud-sim-uj.github.io/PyMPDATA](https://atmos-cloud-sim-uj.github.io/PyMPDATA) 

#### Options class

The [``Options``](https://atmos-cloud-sim-uj.github.io/PyMPDATA/options.html) class
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
[Jaruga, Arabas et al. 2015](https://doi.org/10.5194/gmd-8-1005-2015) and [Olesik, Arabas et al. 2020](https://arxiv.org/abs/2011.14726)
(the last with examples using PyMPDATA).

In most use cases of PyMPDATA, the first thing to do is to instantiate the 
[``Options``](https://atmos-cloud-sim-uj.github.io/PyMPDATA/options.html) class 
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
<details open>
<summary>Python code (click to expand)</summary>

```Python
from PyMPDATA import Options
options = Options(n_iters=2)
```
</details>

#### Arakawa-C grid layer and boundary conditions

In PyMPDATA, the solution domain is assumed to extend from the
first cell's boundary to the last cell's boundary (thus the
first scalar field value is at ![\[\Delta x/2, \Delta y/2\]](https://render.githubusercontent.com/render/math?math=%5B%5CDelta%20x%2F2%2C%20%5CDelta%20y%2F2%5D)).
The [``ScalarField``](https://atmos-cloud-sim-uj.github.io/PyMPDATA/scalar_field.html)
and [``VectorField``](https://atmos-cloud-sim-uj.github.io/PyMPDATA/vector_field.html) classes implement the
[Arakawa-C staggered grid](https://en.wikipedia.org/wiki/Arakawa_grids#Arakawa_C-grid) logic
in which:
- scalar fields are discretised onto cell centres (one value per cell),
- vector field components are discretised onto cell walls.

The schematic of the employed grid/domain layout in two dimensions is given below
(with the Python code snippet generating the figure):
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

![plot](https://github.com/atmos-cloud-sim-uj/PyMPDATA/releases/download/tip/readme_grid.png)

The ``__init__`` methods of
[``ScalarField``](https://atmos-cloud-sim-uj.github.io/PyMPDATA/scalar_field.html)
and [``VectorField``](https://atmos-cloud-sim-uj.github.io/PyMPDATA/vector_field.html)
have the following signatures:
- [``ScalarField(data: np.ndarray, halo: int, boundary_conditions)``](https://github.com/atmos-cloud-sim-uj/PyMPDATA/blob/main/PyMPDATA/scalar_field.py)
- [``VectorField(data: Tuple[np.ndarray, ...], halo: int, boundary_conditions)``](https://github.com/atmos-cloud-sim-uj/PyMPDATA/blob/main/PyMPDATA/vector_field.py)
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

Besides the exemplified [``Periodic``](https://atmos-cloud-sim-uj.github.io/PyMPDATA/boundary_conditions/periodic.html) class representing 
periodic boundary conditions, PyMPDATA supports 
[``Extrapolated``](https://atmos-cloud-sim-uj.github.io/PyMPDATA/boundary_conditions/extrapolated.html), 
[``Constant``](https://atmos-cloud-sim-uj.github.io/PyMPDATA/boundary_conditions/constant.html) and
[``Polar``](https://atmos-cloud-sim-uj.github.io/PyMPDATA/boundary_conditions/polar.html) 
boundary conditions.

#### Stepper

The logic of the MPDATA iterative solver is represented
in PyMPDATA by the [``Stepper``](https://atmos-cloud-sim-uj.github.io/PyMPDATA/stepper.html) class.

When instantiating the [``Stepper``](https://atmos-cloud-sim-uj.github.io/PyMPDATA/stepper.html), the user has a choice 
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

Since creating an instance of the [``Stepper``](https://atmos-cloud-sim-uj.github.io/PyMPDATA/stepper.html) class
involves time-consuming compilation of the algorithm code,
the class is equipped with a cache logic - subsequent
calls with same arguments return references to previously
instantiated objects. Instances of [``Stepper``](https://atmos-cloud-sim-uj.github.io/PyMPDATA/stepper.html) contain no
mutable data and are (thread-)safe to be reused.

The init method of [``Stepper``](https://atmos-cloud-sim-uj.github.io/PyMPDATA/stepper.html) has an optional
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


#### Solver

Instances of the [``Solver``](https://atmos-cloud-sim-uj.github.io/PyMPDATA/solver.html) class are used to control
the integration and access solution data. During instantiation, 
additional memory required by the solver is 
allocated according to the options provided. 

The only method of the [``Solver``](https://atmos-cloud-sim-uj.github.io/PyMPDATA/solver.html) class besides the
init is [``advance(n_steps, mu_coeff, ...)``](https://atmos-cloud-sim-uj.github.io/PyMPDATA/solver.html#PyMPDATA.solver.Solver.advance) 
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

![plot](https://github.com/atmos-cloud-sim-uj/PyMPDATA/releases/download/tip/readme_gauss_0.png)    
![plot](https://github.com/atmos-cloud-sim-uj/PyMPDATA/releases/download/tip/readme_gauss.png)

#### Debugging

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

## Contributing, reporting issues, seeking support 

Submitting new code to the project, please preferably use [GitHub pull requests](https://github.com/atmos-cloud-sim-uj/PyMPDATA/pulls) 
(or the [PyMPDATA-examples PR site](https://github.com/atmos-cloud-sim-uj/PyMPDATA-examples/pulls) if working on examples) - it helps to keep record of code authorship, 
track and archive the code review workflow and allows to benefit
from the continuous integration setup which automates execution of tests 
with the newly added code. 

As of now, the copyright to the entire PyMPDATA codebase is with the Jagiellonian
University, and code contributions are assumed to imply transfer of copyright.
Should there be a need to make an exception, please indicate it when creating
a pull request or contributing code in any other way. In any case, 
the license of the contributed code must be compatible with GPL v3.

Developing the code, we follow [The Way of Python](https://www.python.org/dev/peps/pep-0020/) and 
the [KISS principle](https://en.wikipedia.org/wiki/KISS_principle).
The codebase has greatly benefited from [PyCharm code inspections](https://www.jetbrains.com/help/pycharm/code-inspection.html)
and [Pylint](https://pylint.org) code analysis (Pylint checks are part of the
CI workflows).

Issues regarding any incorrect, unintuitive or undocumented bahaviour of
PyMPDATA are best to be reported on the [GitHub issue tracker](https://github.com/atmos-cloud-sim-uj/PyMPDATA/issues/new).
Feature requests are recorded in the "Ideas..." [PyMPDATA wiki page](https://github.com/atmos-cloud-sim-uj/PyMPDATA/wiki/Ideas-for-new-features-and-examples).

We encourage to use the [GitHub Discussions](https://github.com/atmos-cloud-sim-uj/PyMPDATA/discussions) feature
(rather than the issue tracker) for seeking support in understanding, using and extending PyMPDATA code.

Please use the PyMPDATA issue-tracking and dicsussion infrastructure for `PyMPDATA-examples` as well.
We look forward to your contributions and feedback.

## Credits:
Development of PyMPDATA was supported by the EU through a grant of the [Foundation for Polish Science](http://fnp.org.pl) (POIR.04.04.00-00-5E1C/18).

copyright: Jagiellonian University   
licence: GPL v3   

## Other open-source MPDATA implementations:
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

## Other Python packages for solving hyperbolic transport equations

- PyPDE: https://pypi.org/project/PyPDE/
- FiPy: https://pypi.org/project/FiPy/
- ader: https://pypi.org/project/ader/
- centpy: https://pypi.org/project/centpy/
- mattflow: https://pypi.org/project/mattflow/
- FastFD: https://pypi.org/project/FastFD/
- Pyclaw: https://www.clawpack.org/pyclaw/
