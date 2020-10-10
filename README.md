[![Build Status](https://travis-ci.org/atmos-cloud-sim-uj/PyMPDATA.svg?branch=master)](https://travis-ci.org/atmos-cloud-sim-uj/PyMPDATA)
[![Coverage Status](https://img.shields.io/codecov/c/github/atmos-cloud-sim-uj/PyMPDATA/master.svg)](https://codecov.io/github/atmos-cloud-sim-uj/PyMPDATA?branch=master)

# PyMPDATA

PyMPDATA is a high-performance **Numba-accelerated Pythonic implementation of the MPDATA 
  algorithm of Smolarkiewicz et al.** for numerically solving generalised transport equations -
  partial differential equations used to model conservation/balance laws, scalar-transport problems,
  convection-diffusion phenomena (in geophysical fluid dynamics and beyond).
As of the current version, PyMPDATA supports homogeneous transport
  in 1D, 2D and 3D using structured meshes, optionally
  generalised by employment of a Jacobian of coordinate transformation. 
PyMPDATA includes implementation of a set of MPDATA **variants including
  flux-corrected transport (FCT), infinite-gauge, divergent-flow and 
  third-order-terms options**. 
It also features support for integration of Fickian-terms in advection-diffusion
  problems using the pseudo-transport velocity approach.
In 2D and 3D simulations, domain-decomposition is used for multi-threaded parallelism.

PyMPDATA is engineered purely in Python targeting both performance and usability,
    the latter encompassing research users', developers' and maintainers' perspectives.
From researcher's perspective, PyMPDATA offers **hassle-free installation on multitude
  of platforms including Linux, OSX and Windows**, and eliminates compilation stage
  from the perspective of the user.
From developers' and maintainers' perspective, PyMPDATA offers wide unit-test coverage, 
  multi-platform continuous integration setup,
  seamless integration with Python debugging and profiling tools, and robust susceptibility
  to static code analysis within integrated development environments.

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

PyMPDATA has Python-only dependencies, all available through [PyPi](https://pypi.org/) 
and listed in the project's [requirements.txt](https://github.com/atmos-cloud-sim-uj/PyMPDATA/blob/master/requirements.txt) file.  
 
To install PyMPDATA, one may use:
```bash
pip3 install --pre git+https://github.com/atmos-cloud-sim-uj/PyMPDATA.git
```
 
## Examples/Demos:

PyMPDATA ships with several demos that reproduce results from the literature, including:
- [Smolarkiewicz 2006](http://doi.org/10.1002/fld.1071) Figs 3,4,10,11 & 12
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/atmos-cloud-sim-uj/PyMPDATA.git/master?filepath=PyMPDATA_examples%2FSmolarkiewicz_2006_Figs_3_4_10_11_12/demo.ipynb) 
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atmos-cloud-sim-uj/PyMPDATA/blob/master/PyMPDATA_examples/Smolarkiewicz_2006_Figs_3_4_10_11_12/demo.ipynb)    
  (1D homogeneous cases depicting infinite-gauge and flux-corrected transport cases)
- [Arabas & Farhat 2020](https://doi.org/10.1016/j.cam.2019.05.023) Figs 1-3 & Tab. 1 
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/atmos-cloud-sim-uj/PyMPDATA.git/master?filepath=PyMPDATA_examples%2FArabas_and_Farhat_2020/demo.ipynb) 
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atmos-cloud-sim-uj/PyMPDATA/blob/master/PyMPDATA_examples/Arabas_and_Farhat_2020/demo.ipynb)    
  (1D advection-diffusion example based on Black-Scholes equation)
- Olesik et al. 2020 (in preparation) 
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/atmos-cloud-simm-uj/PyMPDATA.git/master?filepath=PyMPDATA_examples%2FOlesik_et_al_2020/)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atmos-cloud-sim-uj/PyMPDATA/blob/master/PyMPDATA_examples/Olesik_et_al_2020/demo_make_plots.ipynb)   
  (1D particle population condensational growth problem with coordinate transformations)
- Molenkamp test (as in [Jaruga et al. 2015](https://doi.org/10.5194/gmd-8-1005-2015), Fig. 12)
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/atmos-cloud-sim-uj/PyMPDATA.git/master?filepath=PyMPDATA_examples%2FMolenkamp_test_as_in_Jaruga_et_al_2015_Fig_12/)      
  (2D solid-body rotation test)
- 1D advection-diffusion example with animation
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/atmos-cloud-sim-uj/PyMPDATA.git/master?filepath=PyMPDATA_examples%2Fadvection_diffusion_1d/demo.ipynb) 
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atmos-cloud-sim-uj/PyMPDATA/blob/master/PyMPDATA_examples/advection_diffusion_1d/demo.ipynb)    
  
  
  
## Package structure and API:

PyMPDATA is designed as a simple library in the spirit of "Do One Thing And Do It Well", namely to numerically solve the following equation:

![\partial_t (G \psi) + \nabla \cdot (Gu \psi) = 0](https://render.githubusercontent.com/render/math?math=%5Cpartial_t%20(G%20%5Cpsi)%20%2B%20%5Cnabla%20%5Ccdot%20(Gu%20%5Cpsi)%20%3D%200)

where scalar field ![\psi](https://render.githubusercontent.com/render/math?math=%5Cpsi) is referred to as the advectee,
vector field u is referred to as advector, and the G factor corresponds to optional coordinate transformation. 

The key classes constituting the PyMPDATA interface are summarised below:

#### Options class

The [``Options``](https://github.com/atmos-cloud-sim-uj/PyMPDATA/blob/master/PyMPDATA/options.py) class
groups both algorithm variant options as well as some implementation-related
flags that need to be set at the first place. All are set at the time
of instantiation using the following keyword arguments of the constructor 
(all having default values indicated below):
- ``n_iters:int = 2``: number of iterations (2 means upwind + one corrective iteration)
- ``infinite_gauge: bool = False``: flag enabling the infinite-gauge option (does not maintain sign of the advected field, thus in practice implies switching flux corrected transport on)
- ``divergent_flow: bool = False``: flag enabling divergent-flow terms when calculating antidiffusive velocity
- ``flux_corrected_transport: bool = False``: flag enabling flux-corrected transport (FCT) logic (a.k.a. non-oscillatory or monotone variant)
- ``third_order_terms: bool = False``: flag enabling third-order terms
- ``epsilon: float = 1e-15``: value added to potentially zero-valued denominators 
- ``non_zero_mu_coeff: bool = False``: flag indicating if code for handling the Fickian term is to be optimised out

For a discussion of the above options, see e.g., [Smolarkiewicz & Margolin 1998](https://doi.org/10.1006/jcph.1998.5901).

In most use cases of PyMPDATA, the first thing to do is to instantiate the Options class 
with arguments suiting the problem at hand, e.g.:
```python
from PyMPDATA import Options
options = Options(n_iters=3, infinite_gauge=True, flux_corrected_transport=True)
```

#### Arakawa-C grid layer and boundary conditions

The ```arakawa_c``` subpackage contains modules implementing the
[Arakawa-C staggered grid](https://en.wikipedia.org/wiki/Arakawa_grids#Arakawa_C-grid) 
in which:
- scalar fields are discretised onto cell-center points,
- vector fields are discretised onto cell-boundary points.

In PyMPDATA, the solution domain is assumed to extend from the
first cell's boundary to the last cell's boundary (thus
first scalar field value is at ![\[\Delta x/2, \Delta y/2\]](https://render.githubusercontent.com/render/math?math=%5B%5CDelta%20x%2F2%2C%20%5CDelta%20y%2F2%5D)).

From the user perspective, the two key classes with their init methods are:
- [``ScalarField(data: np.ndarray, halo: int, boundary_conditions)``](https://github.com/atmos-cloud-sim-uj/PyMPDATA/blob/master/PyMPDATA/arakawa_c/scalar_field.py)
- [``VectorField(data, halo: int, boundary_conditions)``](https://github.com/atmos-cloud-sim-uj/PyMPDATA/blob/master/PyMPDATA/arakawa_c/vector_field.py)

The ``data`` parameters are expected to be Numpy arrays or tuples of Numpy arrays, respectively.
The ``halo`` parameter is the extent of ghost-cell region that will surround the
data and will be used to implement boundary conditions. Its value (in practice 1 or 2) is
dependent on maximal stencil extent for the MPDATA variant used and
can be easily obtained using the ``Options.n_halo`` property.

As an example, the code below shows how to instantiate a scalar
and a vector field given a 2D constant-velocity problem,
using a grid of 100x100 points and cyclic boundary conditions (with all values set to zero):
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

Note that the shapes of arrays representing components 
of the velocity field are different than the shape of
the scalar field array due to employment of the staggered grid.

Besides the exemplified ``Cyclic`` class representing 
periodic boundary conditions, PyMPDATA supports 
``Extrapolated`` and ``Constant`` boundary conditions.

#### Stepper

The logic of the MPDATA iterative solver is represented
in PyMPDATA by the ``Stepper`` class.

When instantiating the ``Stepper``, the user has a choice 
of either supplying just the
number of dimensions or specialising the stepper for
a given grid:
```python
from PyMPDATA import Stepper

stepper = Stepper(options=options, n_dims=2)
```
or
```python
stepper = Stepper(options=options, grid=(nx, ny))
```

In the latter case, noticeably 
faster execution can be expected, however the resultant
stepper is less versatile as bound to the given grid size.
If number of dimensions is supplied only, the integration
will take longer, yet same instance of the
stepper can be used for different grids.  

Since creating an instance of the ``Stepper`` class
involves time consuming compilation of the algorithm code,
the class is equipped with a cache logic - subsequent
calls with same arguments return references to previously
instantiated objects. Instances of ``Stepper`` contain no
mutable data and are (thread-)safe to be reused.

The init method of ``Stepper`` has an optional
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

Instances of the ``Solver`` class are used to control
the integration and access solution data. During instantiation, 
additional memory required by the solver is 
allocated according to the options provided. 

The only method of the ``Solver`` class besides the
init is ``advance(self, nt: int, mu_coeff: float = 0)`` 
which advances the solution by ``nt`` timesteps, optionally
taking into account a given value of diffusion coefficient.

Solution state is accessible through the ``Solver.advectee`` property.

Continuing with the above code snippets, instantiating
a solver and making one integration step looks as follows:
```python
from PyMPDATA import Solver
solver = Solver(stepper=stepper, advectee=advectee, advector=advector)
solver.advance(nt=1)
state = solver.advectee.get()
```

#### Debugging

PyMPDATA relies heavily on Numba to provide high-performance 
number crunching operations. Arguably, one of the key advantage 
of embracing Numba is that it can be easily switched off. This
brings multiple-order-of-magnitude drop in performance, yet 
it also make the entire code of the library susceptible to
interactive debugging, one way of enabling it is by setting the 
following environment variable before importing PyMPDATA:

```python
import os
os.environ["NUMBA_DISABLE_JIT"] = "1"
```

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
  https://github.com/AtmosFOAM/AtmosFOAM/blob/MPDATA/applications/solvers/advection/MPDATAadvectionFoam/MPDATAadvectionFoam.C
