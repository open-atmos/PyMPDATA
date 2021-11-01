---
title: 'PyMPDATA v1: Numba-accelerated implementation of&nbsp;MPDATA with examples in Python, Julia and Matlab'
tags:
  - Python
  - pde-solver 
  - numerical-integration 
  - numba 
  - advection-diffusion
authors:
  - name: Piotr Bartman
    orcid: 0000-0003-0265-6428
    affiliation: "1"
  - name: Jakub Banaśkiewicz
    affiliation: "1"
  - name: Szymon Drenda
    affiliation: "1"
  - name: Maciej&nbsp;Manna
    affiliation: "1"
  - name: Michael Olesik
    orcid: 0000-0002-6319-9358
    affiliation: "1"
  - name: Paweł Rozwoda
    affiliation: "1"
  - name: Michał Sadowski
    orcid: 0000-0003-3482-9733
    affiliation: "1"
  - name: Sylwester Arabas
    orcid: 0000-0003-2361-0082
    affiliation: "2,1"
affiliations:
 - name: Jagiellonian University, Kraków, Poland 
   index: 1
 - name: University of Illinois at Urbana-Champaign, IL, USA
   index: 2
date: October 2021
bibliography: paper.bib

---

# Statement of need 

Convection-diffusion problems arise across a wide range of pure and applied research,
  in particular in geosciences, aerospace engineering, and financial modelling
  (for an overview of applications, see, e.g., section 1.1 in @Morton_1996).
One of the key challenges in numerical solutions of problems involving advective transport is
  sign preservation of the advected field (for an overview of this and other
  aspects of numerical solutions to advection problems, see, e.g., @Roed_2019).
The Multidimensional Positive Definite Advection Transport Algorithm (``MPDATA``) is a robust, 
  explicit-in-time, and sign-preserving solver introduced in @Smolarkiewicz_1983 and @Smolarkiewicz_1984.
``MPDATA`` has been subsequently developed into a family of numerical schemes with numerous variants 
  and solution procedures addressing a diverse set of problems in geophysical fluid dynamics and beyond.
For reviews of ``MPDATA`` applications and variants, see, e.g., @Smolarkiewicz_and_Margolin_1998_JCP and 
  @Smolarkiewicz_2006.
  
The ``PyMPDATA`` project introduced herein constitutes a high-performance and multi-threaded implementation of
  structured-mesh ``MPDATA`` in ``Python``.
``PyMPDATA`` is aimed to address several aspects which steepen the learning curve and limit collaborative 
  usage and development of existing ``C++`` [e.g., @Jaruga_et_al_2015] 
  and ``Fortran`` [e.g., @Kuehnlein_et_al_2019] implementations of ``MPDATA``.
Performance on par with compiled-language implementations is targeted by employing just-in-time (JIT) compilation
  using ``Numba`` [@Lam_et_al_2015], which translates ``Python`` code into fast machine code using the Low Level Virtual Machine (``LLVM``, https://llvm.org/)
  compiler infrastructure [for a discussion of another JIT implementation of ``MPDATA`` using ``PyPy``, see @Arabas_et_al_2014].
``PyMPDATA`` is engineered aiming at both performance and usability, the latter encompassing 
  research user's, developer's and maintainer's perspectives.
From researcher's perspective, ``PyMPDATA`` offers hassle-free installation on 
  ``Linux``, ``macOS``, and ``Windows``.
It also eliminates the compilation stage from the perspective of the user.
From developer's and maintainer's perspectives, ``PyMPDATA`` offers a suite of unit tests, multi-platform 
  continuous integration setup, seamless integration with ``Python`` development tools including debuggers, profilers,
  and code analysers.

# Summary

``PyMPDATA`` interface uses ``NumPy`` for array-oriented input and output. 
Usage of ``PyMPDATA`` from ``Julia`` (https://julialang.org) and ``Matlab`` (https://mathworks.com) 
  through ``PyCall`` and the built-in ``Python`` interoperability tools, respectively,
  is depicted in the PyMPDATA README file.
The appendices of the present paper include ``Python``, ``Julia`` and ``Matlab`` minimal 
  code snippets covering steps needed to complete a basic ``PyMPDATA`` simulation
  depicted in Fig. (\autoref{fig}) and based on Fig. 5 from @Arabas_et_al_2014.

\begin{figure}[t]
    \centering
    \includegraphics[width=0.5\textwidth]{readme_gauss_0-crop}
    \includegraphics[width=0.5\textwidth]{readme_gauss-crop}
    \caption{Visualisation of the initial condition (left) and simulation state after 75 timesteps (right)
      from the 2D simulation for which sample codes are given in the appendices, based on Fig. 5 from @Arabas_et_al_2014.}
    \label{fig}
\end{figure}

As of the current version, ``PyMPDATA`` supports homogeneous transport in one (1D), two (2D), and three dimensions (3D) 
  using structured meshes, optionally generalised by coordinate transformation 
  [@Smolarkiewicz_and_Clark_1986;@Smolarkiewicz_and_Margolin_1993]. 
``PyMPDATA`` includes implementation of a subset of ``MPDATA`` variants, such as 
  the non-oscillatory option [@Smolarkiewicz_and_Grabowski_1990], 
  the infinite-gauge variant [@Smolarkiewicz_and_Clark_1986;@Margolin_and_Shashkov_2006], 
  the divergent-flow option [@Smolarkiewicz_1984;@Smolarkiewicz_and_Margolin_1998_SIAM],
  the double-pass donor cell (DPDC) flavour [@Beason_Margolin_1988;@Smolarkiewicz_and_Margolin_1998_SIAM;@Margolin_and_Shashkov_2006], and 
  the third-order-terms options [@Smolarkiewicz_and_Margolin_1998_SIAM]. 
It also features support for integration of Fickian-terms in advection-diffusion problems using 
  the pseudo-transport velocity approach [@Smolarkiewicz_and_Clark_1986;@Smolarkiewicz_and_Szmelter_2005]. 

A companion package named ``PyMPDATA-examples`` contains a set of Jupyter notebooks, which reproduce
  results from literature using ``PyMPDATA``.
These examples are also executed within continuous integration runs.
Several of the examples feature comparisons against analytical solution, and these are
  also included in the test suite of ``PyMPDATA``.
The ``PyMPDATA-examples`` README file includes links (badges) offering single-click deployment 
  in the cloud using either the Binder (``https://mybinder.org``) or the Colab (``https://colab.research.google.com``) platforms.

A separate project named ``numba-mpi`` has been developed to set the stage for future Message Passing Interface (``MPI``)
  distributed memory parallelism in ``PyMPDATA``.
The ``PyMPDATA``, the ``PyMPDATA-examples`` and the ``numba-mpi`` packages are available in the 
  ``PyPI`` package repository, and installation of these packages reduces to
  typing ``pip install package_name``.
Development of all three packages is hosted on ``GitHub`` at: https://github.com/atmos-cloud-sim-uj/
  and continuous integration runs on ``Linux``, ``macOS`` and ``Windows`` using
  ``GitHub Actions`` and ``Appveyor`` platforms (the latter used for 32-bit runs on Windows).
Auto-generated documentation sites built with ``pdoc3``  
are hosted at
  https://atmos-cloud-sim-uj.github.io/PyMPDATA/, 
  https://atmos-cloud-sim-uj.github.io/PyMPDATA-examples/, and 
  https://atmos-cloud-sim-uj.github.io/numba-mpi/.
 
``PyMPDATA`` is a free and open-source software released under the terms of the GNU General Public License 3.0 (https://www.gnu.org/licenses/gpl-3.0). 
 
# Usage examples

Simulations included in the ``PyMPDATA-examples`` package 
  are listed below, labelled with the paper reference on which the example setup is based.
Each example is annotated with the dimensionality, 
  number of equations constituting the system, and an outline of setup.
  
   - 1D:
      - @Smolarkiewicz_2006: single-equation advection-only homogeneous problem with different algorithm options depicted with constant velocity field
      - @Arabas_and_Farhat_2020: single-equation advection-diffusion problem resulting from a transformation of the Black-Scholes equation into either homogeneous  or heterogeneous problem for European or American option valuation, respectively
      - @Olesik_et_al_2021: single-equation advection-only homogeneous problem with coordinate transformation depicting application of ``MPDATA`` for condensational growth of a population of particles
   - 2D:
     - @Molenkamp_1968: single-equation homogeneous transport with different algorithm options
     - @Jarecka_et_al_2015: shallow-water system with three equations representing conservation of mass and two components of momentum (with the momentum equations featuring source terms) modelling spreading under gravity of a three-dimensional elliptic drop on a two-dimensional plane
     - @Williamson_and_Rasch_1989: advection on a spherical plane depicting transformation to spherical coordinates
     - @Shipway_and_Hill_2012: coupled system of water vapour mass (single spatial dimension) and water droplet number conservation (spatial and spectral dimensions) with the latter featuring source term modelling activation of water droplets on aerosol particles, coordinate transformation used for representation of air density profile
   - 3D:
     - @Smolarkiewicz_1984: homogeneous single-equation example depicting revolution of a spherical signal in a constant angular velocity rotational velocity field 

In addition, ``PyMPDATA`` is used in a two-dimensional setup in of the examples in the sister PySDM package [@Bartman_et_al_2021].

# Implementation highlights

In 2D and 3D simulations, domain-decomposition is used for multi-threaded parallelism. 
Domain decomposition is performed along the outer dimension only and is realised using
  the ``numba.prange()`` functionality.

``PyMPDATA`` design features a custom-built multi-dimensional Arakawa-C staggered grid layer, allowing to concisely 
  represent multi-dimensional stencil operations on both scalar and vector fields. 
The grid layer is built on top of ``NumPy``'s ``ndarray``s (using "C" ordering) using the ``Numba``'s ``@njit`` 
  functionality for high-performance multi-threaded array traversals. 
The array-traversal layer enables to code once for multiple dimensions (i.e. one set of ``MPDATA`` formulae for 1D, 2D, and 3D), 
  and automatically handles (if needed) any halo-filling logic related to boundary conditions. 

The ``Numba``'s deviation from ``Python`` semantics rendering closure variables as compile-time constants 
  is extensively exploited within ``PyMPDATA`` code base enabling the just-in-time compilation to benefit
  from information on domain extents, algorithm variant used and problem characteristics (e.g., coordinate 
  transformation used, or lack thereof). 

In general, the numerical and concurrency aspects of ``PyMPDATA`` implementation follow the ``libmpdata++`` 
  open-source ``C++`` implementation of ``MPDATA`` [@Jaruga_et_al_2015].

# Appendix P: Python sample code 

```Python
import numpy as np
from PyMPDATA import Options, ScalarField, VectorField, Stepper, Solver
from PyMPDATA.boundary_conditions import Periodic

options = Options(n_iters=2)
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
stepper = Stepper(options=options, grid=(nx, ny))
solver = Solver(stepper=stepper, advectee=advectee, advector=advector)
solver.advance(n_steps=75)
state = solver.advectee.get()
```

# Appendix J: Julia sample code

```Julia
using PyCall

Options = pyimport("PyMPDATA").Options
ScalarField = pyimport("PyMPDATA").ScalarField
VectorField = pyimport("PyMPDATA").VectorField
Stepper = pyimport("PyMPDATA").Stepper
Solver = pyimport("PyMPDATA").Solver
Periodic = pyimport("PyMPDATA.boundary_conditions").Periodic

options = Options(n_iters=2)
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

stepper = Stepper(options=options, grid=(nx, ny))
solver = Solver(stepper=stepper, advectee=advectee, advector=advector)
solver.advance(n_steps=75)
state = solver.advectee.get()
```

# Appendix M: Matlab sample code

```Matlab
Options = py.importlib.import_module('PyMPDATA').Options;
ScalarField = py.importlib.import_module('PyMPDATA').ScalarField;
VectorField = py.importlib.import_module('PyMPDATA').VectorField;
Stepper = py.importlib.import_module('PyMPDATA').Stepper;
Solver = py.importlib.import_module('PyMPDATA').Solver;
Periodic = py.importlib.import_module('PyMPDATA.boundary_conditions').Periodic;

options = Options(pyargs('n_iters', 2));
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

stepper = Stepper(pyargs(...
  'options', options, ... 
  'grid', py.tuple({nx, ny}) ... 
));
solver = Solver(...
    pyargs('stepper', ...
    stepper, 'advectee', ...
    advectee, 'advector', ...
    advector ...
));
solver.advance(pyargs('n_steps', 75));
state = solver.advectee.get();
```

# Author contributions

PB had been the architect of ``PyMPDATA`` with SA taking the role of main developer and maintainer over the time.
MO participated in the package core development and led the development of the condensational-growth example, which was the basis of his MSc thesis.
JB contributed the DPDC algorithm variant handling.
SD contributed the advection-diffusion example.
MM contributed to the ``numba-mpi`` package.
PR contributed the shallow-water example.
MS contributed the advection-on-a-sphere example.
The paper was composed by SA and is based on the contents of the README files of the ``PyMPDATA``, ``PyMPDATA-examples``, and ``numba-mpi`` packages.

# Acknowledgements

Development of ``PyMPDATA`` has been carried out within the POWROTY/REINTEGRATION programme of the Foundation for Polish Science, co-financed by the European Union under the European Regional Development Fund (POIR.04.04.00-00-5E1C/18).

# References
