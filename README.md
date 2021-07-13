# PyMPDATA

[![Python 3](https://img.shields.io/static/v1?label=Python&logo=Python&color=3776AB&message=3)](https://www.python.org/)
[![LLVM](https://img.shields.io/static/v1?label=LLVM&logo=LLVM&color=gold&message=Numba)](https://www.numba.org)
[![Linux OK](https://img.shields.io/static/v1?label=Linux&logo=Linux&color=yellow&message=%E2%9C%93)](https://en.wikipedia.org/wiki/Linux)
[![macOS OK](https://img.shields.io/static/v1?label=macOS&logo=Apple&color=silver&message=%E2%9C%93)](https://en.wikipedia.org/wiki/macOS)
[![Windows OK](https://img.shields.io/static/v1?label=Windows&logo=Windows&color=white&message=%E2%9C%93)](https://en.wikipedia.org/wiki/Windows)
[![Jupyter](https://img.shields.io/static/v1?label=Jupyter&logo=Jupyter&color=f37626&message=%E2%9C%93)](https://jupyter.org/)
[![Dependabot](https://img.shields.io/static/v1?label=Dependabot&logo=dependabot&color=blue&message=on)](https://github.com/atmos-cloud-sim-uj/PyMPDATA/network/updates)
[![Coverage Status](https://codecov.io/gh/atmos-cloud-sim-uj/PyMPDATA/branch/master/graph/badge.svg)](https://codecov.io/github/atmos-cloud-sim-uj/PyMPDATA?branch=master)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/atmos-cloud-sim-uj/PyMPDATA/graphs/commit-activity)
[![OpenHub](https://www.openhub.net/p/atmos-cloud-sim-uj-PyMPDATA/widgets/project_thin_badge?format=gif)](https://www.openhub.net/p/atmos-cloud-sim-uj-PyMPDATA)
[![EU Funding](https://img.shields.io/static/v1?label=EU%20Funding%20by&color=103069&message=FNP&logoWidth=25&logo=image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC4AAAAeCAYAAABTwyyaAAAEzklEQVRYw9WYS2yUVRiGn3P5ZzozpZ3aUsrNgoKlKBINmkhpCCwwxIAhsDCpBBIWhmCMMYTEhSJ4i9EgBnSBEm81MRrFBhNXEuUSMCopiRWLQqEGLNgr085M5//POS46NNYFzHQ6qGc1i5nzP/P973m/9ztCrf7A8T9csiibCocUbvTzfxLcAcaM3cY3imXz25lT3Y34G7gQYAKV3+bFAHcATlBTPogJNADG92iY28FHW97kyPbnuW/W7xgzAhukQ9xe04PJeOT0HkQRwK0TlEeGWb/kOO9v3kdD3a8YK9GhDMfa6mg9fxunOm/lWPtcpDI4K7n/jnN8+uQbrFrUSiwU/DtSEUB/MsKKBT+zYslJqiYNgVE4JwhHkzy86wlWvrKVWDSZ/YFjZlU39yw4y/rGoyQGowWB67zl4QQue+jssMdXrQvZ/00jyeHwqCgDKwnsiJjSvkYAxsG5K9WsenYbJdqAtAjhCIxCSZt/4fK1w5A2WCvxrUAKCHwNVoA2aGmvq11jJQQapEXrgMBKqmJJugejKGWLIxXrBPFoigfv/omd675gRkU/xgqUDlAhH3UDaAAlLSqUQekAYyVTyhLs3tDMsvntlIYzOFcEcOcEGd9jx9oDbGs6QO0t/Tijxi9S4bhzxiWaVh5m94Zm0n7oui4ybo0raUlcncQnxx+g+WgDF/vLoYDmoqSl/dJUnt7XRCoTZjij0Z6Pc2LiNS4EBBkNvoeOJXN+yPWWSZeANOhwJq/98nKVwNdoL8B5AROxBKBL0gjh8DMhdCh3eJnrA0yqhLpplwmyup6IajvAOIGfKGVx3VmCRGnOMpe5QAdG0bT8CAeeep0d6z6nqjSJnQiZWEllLMWrmz6k+fE9rGk8MVqYgsGv5ZH2i1Opr+9kajzB5d74hKQ+KS3d/WVMLhtgdu1lriRiOR/4nDVunaR24x7qp3UV5Cb/fJvC83nv26W81LIK58SYNFmwq4hsGx/5BwKlzYRma2NUthgOJSew4i7ru9nJYCQF5tApb2yvjiDQKJV/IfJKh0o6qssSLKv/jcAoRKHQQzE2Lj2OMV5OkWFc4MZIpsev8uXWXRx6ZicbGk8QZLxxgwe+x/rlR3h3816+f2E7lbEU+ZDn3vKVpePCdFovzCISHqbl5EIoQOteKMPB1rto65zNyfOz+KOrGl06lHPQyi/WOohH0/T0l1MZH6A3GUEKl7Pmr2la6wBrBWWRDP2DUcqjKVKBGom9RZmABAykwnglafpSJSPQvsfiOR0EQ7ExVmazA8cY6N4K1iw6RdAXRwi4mgrheT5Dvs4LeuS81a15Ll/3dQisFVSVpnj7sf1sX/sZvhAc+6UOrQyBVUQ8gx/orFmDsZqtaw/y1qZ9zKjp5vDpenyjcNe+cLNmTiUdf/bEOddVQ0VpgsOn54ET+EYxvWKALSu+5tGG76it7MNaiZKGQ23zCIcMfUMxBnrjN3fmHHvCAlp+vJcXWx6itqoXpAEnUNLx8iMfo5Xh1i17R3PJYCpC2cZ3qK3sQ8WGEDDuXlAQuFKGHzpmopXhTNfk0bmxs7uC1w6uJul79AxFkMIiBJy5UoUWjrZLU5DCFdTARDHuDqVw+OkSwI0MCEW4gtNF2BPrBCo8fKNbtILWX9aUDqFqHnn7AAAAAElFTkSuQmCC)](https://www.fnp.org.pl/en/)
[![PL Funding](https://img.shields.io/static/v1?label=PL%20Funding%20by&color=d21132&message=NCN&logoWidth=25&logo=image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAANCAYAAACpUE5eAAAABmJLR0QA/wD/AP+gvaeTAAAAKUlEQVQ4jWP8////fwYqAiZqGjZqIHUAy4dJS6lqIOMdEZvRZDPcDQQAb3cIaY1Sbi4AAAAASUVORK5CYII=)](https://www.ncn.gov.pl/?language=en)
[![Copyright](https://img.shields.io/static/v1?label=Copyright&color=249fe2&message=Jagiellonian%20University&)](https://en.uj.edu.pl/)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.html)

#### [core package](https://github.com/atmos-cloud-sim-uj/PyMPDATA)

[![Github Actions Build Status](https://github.com/atmos-cloud-sim-uj/PyMPDATA/workflows/PyMPDATA/badge.svg?branch=master)](https://github.com/atmos-cloud-sim-uj/PyMPDATA/actions)
[![Travis Build Status](https://img.shields.io/travis/atmos-cloud-sim-uj/PyMPDATA/master.svg?logo=travis)](https://travis-ci.com/atmos-cloud-sim-uj/PyMPDATA)
[![Appveyor Build status](http://ci.appveyor.com/api/projects/status/github/atmos-cloud-sim-uj/PyMPDATA?branch=master&svg=true)](https://ci.appveyor.com/project/slayoo/pympdata/branch/master)    
[![GitHub issues](https://img.shields.io/github/issues-pr/atmos-cloud-sim-uj/PyMPDATA.svg?logo=github&logoColor=white)](https://github.com/atmos-cloud-sim-uj/PyMPDATA/pulls?q=)
[![GitHub issues](https://img.shields.io/github/issues-pr-closed/atmos-cloud-sim-uj/PyMPDATA.svg?logo=github&logoColor=white)](https://github.com/atmos-cloud-sim-uj/PyMPDATA/pulls?q=is:closed)    
[![GitHub issues](https://img.shields.io/github/issues/atmos-cloud-sim-uj/PyMPDATA.svg?logo=github&logoColor=white)](https://github.com/atmos-cloud-sim-uj/PyMPDATA/issues?q=)
[![GitHub issues](https://img.shields.io/github/issues-closed/atmos-cloud-sim-uj/PyMPDATA.svg?logo=github&logoColor=white)](https://github.com/atmos-cloud-sim-uj/PyMPDATA/issues?q=)    
[![PyPI version](https://badge.fury.io/py/PyMPDATA.svg)](https://pypi.org/project/PyMPDATA)

#### [examples package](https://github.com/atmos-cloud-sim-uj/PyMPDATA-examples)

[![Github Actions Build Status](https://github.com/atmos-cloud-sim-uj/PyMPDATA-examples/workflows/PyMPDATA-examples/badge.svg?branch=main)](https://github.com/atmos-cloud-sim-uj/PyMPDATA-examples/actions)    
[![GitHub issues](https://img.shields.io/github/issues-pr/atmos-cloud-sim-uj/PyMPDATA-examples.svg?logo=github&logoColor=white)](https://github.com/atmos-cloud-sim-uj/PyMPDATA-examples/pulls?q=)
[![GitHub issues](https://img.shields.io/github/issues-pr-closed/atmos-cloud-sim-uj/PyMPDATA-examples.svg?logo=github&logoColor=white)](https://github.com/atmos-cloud-sim-uj/PyMPDATA-examples/pulls?q=is:closed)    

PyMPDATA is a high-performance **Numba-accelerated Pythonic implementation of the MPDATA 
  algorithm of Smolarkiewicz et al.** for numerically solving generalised transport equations -
  partial differential equations used to model conservation/balance laws, scalar-transport problems,
  convection-diffusion phenomena (in geophysical fluid dynamics and beyond).
As of the current version, PyMPDATA supports homogeneous transport
  in 1D, 2D and 3D (work in progress) using structured meshes, optionally
  generalised by employment of a Jacobian of coordinate transformation. 
PyMPDATA includes implementation of a set of MPDATA **variants including
  the non-oscillatory option, infinite-gauge, divergent-flow, double-pass donor cell (DPDC) and 
  third-order-terms options**. 
It also features support for integration of Fickian-terms in advection-diffusion
  problems using the pseudo-transport velocity approach.
In 2D and 3D simulations, domain-decomposition is used for multi-threaded parallelism.

PyMPDATA is engineered purely in Python targeting both performance and usability,
    the latter encompassing research users', developers' and maintainers' perspectives.
From researcher's perspective, PyMPDATA offers **hassle-free installation on multitude
  of platforms including Linux, OSX and Windows**, and eliminates compilation stage
  from the perspective of the user.
From developers' and maintainers' perspective, PyMPDATA offers a suite of unit tests, 
  multi-platform continuous integration setup,
  seamless integration with Python development aids including debuggers and profilers.

PyMPDATA design features
  a **custom-built multi-dimensional Arakawa-C grid layer** allowing
  to concisely represent multi-dimensional stencil operations.
The grid layer is built on top of NumPy's ndarrays (using "C" ordering)
  using Numba's @njit functionality for high-performance array traversals.
It enables one to code once for multiple dimensions, and automatically
  handles (and hides from the user) any halo-filling logic related with boundary conditions.

PyMPDATA ships with a set of **examples/demos offered as github-hosted Jupyer notebooks
  offering single-click deployment in the cloud using [mybinder.org](https://mybinder.org)**
  or using [colab.research.google.com](https://colab.research.google.com/).
The examples/demos reproduce results from several published
  works on MPDATA and its applications, and provide a validation of the implementation
  and its performance.
 
## Dependencies and installation

PyMPDATA depends on NumPy, Numba and SciPy which are all listed as project dependencies 
within the project's [setup.py](https://github.com/atmos-cloud-sim-uj/PyMPDATA/blob/master/setup.py) file.  
 
To install PyMPDATA, one may use: ``pip install --pre git+https://github.com/atmos-cloud-sim-uj/PyMPDATA.git``

Running the tests shipped with the package requires additional packages listed in the 
[test-time-requirements.txt](https://github.com/atmos-cloud-sim-uj/PyMPDATA/blob/master/test-time-requirements.txt) file.

PyMPDATA examples listed below are hosted in a separate repository and constitute 
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

## Examples:

PyMPDATA ships with several demos that reproduce results from literature, including:
- [Smolarkiewicz 2006](http://doi.org/10.1002/fld.1071) Figs 3,4,10,11 & 12    
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/atmos-cloud-sim-uj/PyMPDATA-examples.git/main?urlpath=lab/tree/PyMPDATA_examples%2FSmolarkiewicz_2006_Figs_3_4_10_11_12/demo.ipynb) 
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atmos-cloud-sim-uj/PyMPDATA-examples/blob/main/PyMPDATA_examples/Smolarkiewicz_2006_Figs_3_4_10_11_12/demo.ipynb)    
  (1D homogeneous cases depicting infinite-gauge and flux-corrected transport cases)
- [Arabas & Farhat 2020](https://doi.org/10.1016/j.cam.2019.05.023) Figs 1-3 & Tab. 1     
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/atmos-cloud-sim-uj/PyMPDATA-examples.git/main?urlpath=lab/tree/PyMPDATA_examples%2FArabas_and_Farhat_2020/demo.ipynb) 
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atmos-cloud-sim-uj/PyMPDATA-examples/blob/main/PyMPDATA_examples/Arabas_and_Farhat_2020/demo.ipynb)    
  (1D advection-diffusion example based on Black-Scholes equation)
- [Olesik et al. 2020](https://arxiv.org/abs/2011.14726)    
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/atmos-cloud-simm-uj/PyMPDATA-examples.git/main?urlpath=lab/tree/PyMPDATA_examples%2FOlesik_et_al_2020/)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atmos-cloud-sim-uj/PyMPDATA-examples/blob/main/PyMPDATA_examples/Olesik_et_al_2020/demo_make_plots.ipynb)   
  (1D particle population condensational growth problem with coordinate transformations)
- Molenkamp 2D solid-body rotation test (as in [Jaruga et al. 2015](https://doi.org/10.5194/gmd-8-1005-2015), Fig. 12)    
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/atmos-cloud-sim-uj/PyMPDATA-examples.git/main?urlpath=lab/tree/PyMPDATA_examples%2FMolenkamp_test_as_in_Jaruga_et_al_2015_Fig_12/demo.ipynb)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atmos-cloud-sim-uj/PyMPDATA-examples/blob/main/PyMPDATA_examples/Molenkamp_test_as_in_Jaruga_et_al_2015_Fig_12/demo.ipynb)
- 1D advection-diffusion example with animation    
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/atmos-cloud-sim-uj/PyMPDATA-examples.git/main?urlpath=lab/tree/PyMPDATA_examples%2Fadvection_diffusion_1d/demo.ipynb) 
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atmos-cloud-sim-uj/PyMPDATA-examples/blob/main/PyMPDATA_examples/advection_diffusion_1d/demo.ipynb)    
- 2D advection on a sphere (**upwind-only** setup depicting coordinate transformation based on [Williamson and Rasch 1989](https://doi.org/10.1175/1520-0493(1989)117%3C0102:TDSLTW%3E2.0.CO;2))  
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/atmos-cloud-sim-uj/PyMPDATA-examples.git/main?urlpath=lab/tree/PyMPDATA_examples%2FWilliamson_and_Rasch_1989_as_in_Jaruga_et_al_2015_Fig_14/demo_over_the_pole.ipynb) 
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atmos-cloud-sim-uj/PyMPDATA-examples/blob/main/PyMPDATA_examples/Williamson_and_Rasch_1989_as_in_Jaruga_et_al_2015_Fig_14/demo_over_the_pole.ipynb)    
![animation](https://github.com/atmos-cloud-sim-uj/PyMPDATA/wiki/files/sphere_upwind.gif)  
  
## Package structure and API:

In short, PyMPDATA numerically solves the following equation:

![\partial_t (G \psi) + \nabla \cdot (Gu \psi) = 0](https://render.githubusercontent.com/render/math?math=%5Cpartial_t%20(G%20%5Cpsi)%20%2B%20%5Cnabla%20%5Ccdot%20(Gu%20%5Cpsi)%20%3D%200)

where scalar field ![\psi](https://render.githubusercontent.com/render/math?math=%5Cpsi) is referred to as the advectee,
vector field u is referred to as advector, and the G factor corresponds to optional coordinate transformation. 

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
- ``flux_corrected_transport: bool = False``: flag enabling flux-corrected transport (FCT) logic (a.k.a. non-oscillatory or monotone variant)
- ``third_order_terms: bool = False``: flag enabling third-order terms
- ``epsilon: float = 1e-15``: value added to potentially zero-valued denominators 
- ``non_zero_mu_coeff: bool = False``: flag indicating if code for handling the Fickian term is to be optimised out
- ``DPDC: bool = False``: flag enabling double-pass donor cell option (recursive pseudovelocities)
- ``dtype: np.floating = np.float64``: floating point precision

For a discussion of the above options, see e.g., [Smolarkiewicz & Margolin 1998](https://doi.org/10.1006/jcph.1998.5901),
[Jaruga, Arabas et al. 2015](https://doi.org/10.5194/gmd-8-1005-2015) and [Olesik, Arabas et al. 2020](https://arxiv.org/abs/2011.14726)
(the last with examples using PyMPDATA).

In most use cases of PyMPDATA, the first thing to do is to instantiate the 
[``Options``](https://atmos-cloud-sim-uj.github.io/PyMPDATA/options.html) class 
with arguments suiting the problem at hand, e.g.:
<details>
<summary>Julia (click to expand)</summary>

```Julia
using Pkg
Pkg.add("PyCall")
using PyCall
Options = pyimport("PyMPDATA").Options
options = Options(n_iters=3, infinite_gauge=true, flux_corrected_transport=true)
```
</details>
<details>
<summary>Matlab (click to expand)</summary>

```Matlab
Options = py.importlib.import_module('PyMPDATA').Options;
options = Options(pyargs(...
  'n_iters', 3, ...
  'infinite_gauge', true, ...
  'flux_corrected_transport', true ...
));
```
</details>
<details open>
<summary>Python</summary>

```python
from PyMPDATA import Options
options = Options(n_iters=3, infinite_gauge=True, flux_corrected_transport=True)
```
</details>

#### Arakawa-C grid layer and boundary conditions

The [``arakawa_c``](https://atmos-cloud-sim-uj.github.io/PyMPDATA/arakawa_c/index.html) subpackage contains modules implementing the
[Arakawa-C staggered grid](https://en.wikipedia.org/wiki/Arakawa_grids#Arakawa_C-grid) 
in which:
- scalar fields are discretised onto cell-center points,
- vector fields are discretised onto cell-boundary points.

In PyMPDATA, the solution domain is assumed to extend from the
first cell's boundary to the last cell's boundary (thus
first scalar field value is at ![\[\Delta x/2, \Delta y/2\]](https://render.githubusercontent.com/render/math?math=%5B%5CDelta%20x%2F2%2C%20%5CDelta%20y%2F2%5D)).

From the user perspective, the two key classes with their init methods are:
- [``ScalarField(data: np.ndarray, halo: int, boundary_conditions)``](https://github.com/atmos-cloud-sim-uj/PyMPDATA/blob/master/PyMPDATA/arakawa_c/scalar_field.py)
- [``VectorField(data: Tuple[np.ndarray, ...], halo: int, boundary_conditions)``](https://github.com/atmos-cloud-sim-uj/PyMPDATA/blob/master/PyMPDATA/arakawa_c/vector_field.py)

The ``data`` parameters are expected to be Numpy arrays or tuples of Numpy arrays, respectively.
The ``halo`` parameter is the extent of ghost-cell region that will surround the
data and will be used to implement boundary conditions. Its value (in practice 1 or 2) is
dependent on maximal stencil extent for the MPDATA variant used and
can be easily obtained using the ``Options.n_halo`` property.

As an example, the code below shows how to instantiate a scalar
and a vector field given a 2D constant-velocity problem,
using a grid of 100x100 points and cyclic boundary conditions (with all values set to zero):
<details>
<summary>Julia (click to expand)</summary>

```Julia
ScalarField = pyimport("PyMPDATA").ScalarField
VectorField = pyimport("PyMPDATA").VectorField
PeriodicBoundaryCondition = pyimport("PyMPDATA").PeriodicBoundaryCondition

nx, ny = 100, 100
halo = options.n_halo
advectee = ScalarField(
    data=zeros((nx, ny)), 
    halo=halo, 
    boundary_conditions=(PeriodicBoundaryCondition(), PeriodicBoundaryCondition())
)
advector = VectorField(
    data=(zeros((nx+1, ny)), zeros((nx, ny+1))),
    halo=halo,
    boundary_conditions=(PeriodicBoundaryCondition(), PeriodicBoundaryCondition())    
)
```
</details>
<details>
<summary>Matlab (click to expand)</summary>

```Matlab
ScalarField = py.importlib.import_module('PyMPDATA').ScalarField;
VectorField = py.importlib.import_module('PyMPDATA').VectorField;
PeriodicBoundaryCondition = py.importlib.import_module('PyMPDATA').PeriodicBoundaryCondition;

nx = int32(100);
ny = int32(100);
halo = options.n_halo;
advectee = ScalarField(pyargs(...
    'data', py.numpy.zeros(int32([nx ny])), ... 
    'halo', halo, ...
    'boundary_conditions', py.tuple({PeriodicBoundaryCondition(), PeriodicBoundaryCondition()}) ...
));
advector = VectorField(pyargs(...
    'data', py.tuple({py.numpy.zeros(int32([nx+1 ny])), py.numpy.zeros(int32([nx ny+1]))}), ...
    'halo', halo, ...
    'boundary_conditions', py.tuple({PeriodicBoundaryCondition(), PeriodicBoundaryCondition()}) ...
));
```
</details>
<details open>
<summary>Python</summary>

```python
from PyMPDATA import ScalarField
from PyMPDATA import VectorField
from PyMPDATA import PeriodicBoundaryCondition
import numpy as np

nx, ny = 100, 100
halo = options.n_halo
advectee = ScalarField(
    data=np.zeros((nx, ny)), 
    halo=halo, 
    boundary_conditions=(PeriodicBoundaryCondition(), PeriodicBoundaryCondition())
)
advector = VectorField(
    data=(np.zeros((nx+1, ny)), np.zeros((nx, ny+1))),
    halo=halo,
    boundary_conditions=(PeriodicBoundaryCondition(), PeriodicBoundaryCondition())    
)
```
</details>

Note that the shapes of arrays representing components 
of the velocity field are different than the shape of
the scalar field array due to employment of the staggered grid.

Besides the exemplified [``PeriodicBoundaryCondition``](https://atmos-cloud-sim-uj.github.io/PyMPDATA/arakawa_c/boundary_condition/periodic_boundary_condition.html) class representing 
periodic boundary conditions, PyMPDATA supports 
[``ExtrapolatedBoundaryCondition``](https://atmos-cloud-sim-uj.github.io/PyMPDATA/arakawa_c/boundary_condition/extrapolated_boundary_condition.html), 
[``ConstantBoundaryCondition``](https://atmos-cloud-sim-uj.github.io/PyMPDATA/arakawa_c/boundary_condition/constant_boundary_condition.html) and
[``PolarBoundaryCondition``](https://atmos-cloud-sim-uj.github.io/PyMPDATA/arakawa_c/boundary_condition/polar_boundary_condition.html).

#### Stepper

The logic of the MPDATA iterative solver is represented
in PyMPDATA by the [``Stepper``](https://atmos-cloud-sim-uj.github.io/PyMPDATA/stepper.html) class.

When instantiating the [``Stepper``](https://atmos-cloud-sim-uj.github.io/PyMPDATA/stepper.html), the user has a choice 
of either supplying just the  number of dimensions or specialising the stepper for a given grid:
<details>
<summary>Julia (click to expand)</summary>

```Julia
Stepper = pyimport("PyMPDATA").Stepper

stepper = Stepper(options=options, n_dims=2)
```
</details>
<details>
<summary>Matlab (click to expand)</summary>

```Matlab
Stepper = py.importlib.import_module('PyMPDATA').Stepper;

steper = Stepper(pyargs(...
  'options', options, ...
  'n_dims', int32(2) ...
));
```
</details>
<details open>
<summary>Python</summary>

```python
from PyMPDATA import Stepper

stepper = Stepper(options=options, n_dims=2)
```
</details>
or
<details>
<summary>Julia (click to expand)</summary>

```Julia
stepper = Stepper(options=options, grid=(nx, ny))
```
</details>
<details>
<summary>Matlab (click to expand)</summary>

```Matlab
stepper = Stepper(pyargs(...
  'options', options, ...
  'grid', py.tuple({nx, ny}) ...
))
```
</details>
<details open>
<summary>Python</summary>

```python
stepper = Stepper(options=options, grid=(nx, ny))
```
</details>

In the latter case, noticeably 
faster execution can be expected, however the resultant
stepper is less versatile as bound to the given grid size.
If number of dimensions is supplied only, the integration
will take longer, yet same instance of the
stepper can be used for different grids.  

Since creating an instance of the [``Stepper``](https://atmos-cloud-sim-uj.github.io/PyMPDATA/stepper.html) class
involves time consuming compilation of the algorithm code,
the class is equipped with a cache logic - subsequent
calls with same arguments return references to previously
instantiated objects. Instances of [``Stepper``](https://atmos-cloud-sim-uj.github.io/PyMPDATA/stepper.html) contain no
mutable data and are (thread-)safe to be reused.

The init method of [``Stepper``](https://atmos-cloud-sim-uj.github.io/PyMPDATA/stepper.html) has an optional
``non_unit_g_factor`` argument which is a Boolean flag 
enabling handling of the G factor term which can be used to 
represent coordinate transformations and/or variable fluid density. 

Optionally, the number of threads to use for domain decomposition
in first (non-contiguous) dimension during 2D and 3D calculations
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
init is ``advance(self, nt: int, mu_coeff: float = 0)`` 
which advances the solution by ``nt`` timesteps, optionally
taking into account a given value of diffusion coefficient.

Solution state is accessible through the ``Solver.advectee`` property.

Continuing with the above code snippets, instantiating
a solver and making one integration step looks as follows:
<details>
<summary>Julia (click to expand)</summary>

```Julia
Solver = pyimport("PyMPDATA").Solver
solver = Solver(stepper=stepper, advectee=advectee, advector=advector)
solver.advance(nt=1)
state = solver.advectee.get()
```
</details>
<details>
<summary>Matlab (click to expand)</summary>

```Matlab
Solver = py.importlib.import_module('PyMPDATA').Solver;
solver = Solver(pyargs('stepper', stepper, 'advectee', advectee, 'advector', advector));
solver.advance(pyargs('nt', 1));
state = solver.advectee.get();
```
</details>
<details open>
<summary>Python</summary>

```python
from PyMPDATA import Solver
solver = Solver(stepper=stepper, advectee=advectee, advector=advector)
solver.advance(nt=1)
state = solver.advectee.get()
```
</details>

#### Debugging

PyMPDATA relies heavily on Numba to provide high-performance 
number crunching operations. Arguably, one of the key advantage 
of embracing Numba is that it can be easily switched off. This
brings multiple-order-of-magnitude drop in performance, yet 
it also make the entire code of the library susceptible to
interactive debugging, one way of enabling it is by setting the 
following environment variable before importing PyMPDATA:
<details>
<summary>Julia (click to expand)</summary>

```Julia
ENV["NUMBA_DISABLE_JIT"] = "1"
```
</details>
<details>
<summary>Matlab (click to expand)</summary>

```Matlab
setenv('NUMBA_DISABLE_JIT', '1');
```
</details>
<details open>
<summary>Python</summary>

```python
import os
os.environ["NUMBA_DISABLE_JIT"] = "1"
```
</details>

## Credits:
Development of PyMPDATA is supported by the EU through a grant of the [Foundation for Polish Science](http://fnp.org.pl) (POIR.04.04.00-00-5E1C/18).

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
