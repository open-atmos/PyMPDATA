<img src="https://raw.githubusercontent.com/open-atmos/PyMPDATA/main/.github/pympdata_logo.svg" width=100 height=113 alt="pympdata logo">

# PyMPDATA

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

A separate project called [``PyMPDATA-MPI``](https://github.com/open-atmos/PyMPDATA-MPI) 
  depicts how [``numba-mpi``](https://pypi.org/project/numba-mpi) can be used
  to enable distributed memory parallelism in PyMPDATA.

The [``PyMPDATA-examples``](https://pypi.org/project/PyMPDATA-examples/) 
  package covers a set of examples presented in the form of Jupyer notebooks
  offering single-click deployment in the cloud using [mybinder.org](https://mybinder.org)
  or using [colab.research.google.com](https://colab.research.google.com/).
The examples reproduce results from several published
  works on MPDATA and its applications, and provide a validation of the implementation
  and its performance.
