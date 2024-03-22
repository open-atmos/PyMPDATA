# PyMPDATA-MPI

[![Python 3](https://img.shields.io/static/v1?label=Python&logo=Python&color=3776AB&message=3)](https://www.python.org/)
[![LLVM](https://img.shields.io/static/v1?label=LLVM&logo=LLVM&color=gold&message=Numba)](https://numba.pydata.org)
[![Linux OK](https://img.shields.io/static/v1?label=Linux&logo=Linux&color=yellow&message=%E2%9C%93)](https://en.wikipedia.org/wiki/Linux)
[![macOS OK](https://img.shields.io/static/v1?label=macOS&logo=Apple&color=silver&message=%E2%9C%93)](https://en.wikipedia.org/wiki/macOS)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/open-atmos/PyMPDATA-MPI/graphs/commit-activity)

[![PL Funding](https://img.shields.io/static/v1?label=PL%20Funding%20by&color=d21132&message=NCN&logoWidth=25&logo=image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAANCAYAAACpUE5eAAAABmJLR0QA/wD/AP+gvaeTAAAAKUlEQVQ4jWP8////fwYqAiZqGjZqIHUAy4dJS6lqIOMdEZvRZDPcDQQAb3cIaY1Sbi4AAAAASUVORK5CYII=)](https://www.ncn.gov.pl/?language=en)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.html)
[![Copyright](https://img.shields.io/static/v1?label=Copyright&color=249fe2&message=Jagiellonian%20University&)](https://en.uj.edu.pl/)

[![GitHub issues](https://img.shields.io/github/issues-pr/open-atmos/PyMPDATA-MPI.svg?logo=github&logoColor=white)](https://github.com/open-atmos/PyMPDATA-MPI/pulls?q=)
[![GitHub issues](https://img.shields.io/github/issues-pr-closed/open-atmos/PyMPDATA-MPI.svg?logo=github&logoColor=white)](https://github.com/open-atmos/PyMPDATA-MPI/pulls?q=is:closed)    
[![GitHub issues](https://img.shields.io/github/issues/open-atmos/PyMPDATA-MPI.svg?logo=github&logoColor=white)](https://github.com/open-atmos/PyMPDATA-MPI/issues?q=)
[![GitHub issues](https://img.shields.io/github/issues-closed/open-atmos/PyMPDATA-MPI.svg?logo=github&logoColor=white)](https://github.com/open-atmos/PyMPDATA-MPI/issues?q=is:closed)   
[![Github Actions Build Status](https://github.com/open-atmos/PyMPDATA-MPI/workflows/tests+pypi/badge.svg?branch=main)](https://github.com/open-atmos/PyMPDATA-MPI/actions)
[![PyPI version](https://badge.fury.io/py/PyMPDATA-MPI.svg)](https://pypi.org/project/PyMPDATA-MPI)
[![API docs](https://img.shields.io/badge/API_docs-pdoc3-blue.svg)](https://open-atmos.github.io/PyMPDATA-MPI/)

PyMPDATA-MPI constitutes a [PyMPDATA](https://github.com/open-atmos/PyMPDATA) +
[numba-mpi](https://github.com/numba-mpi/numba-mpi) coupler enabling numerical solutions
of transport equations with the MPDATA numerical scheme in a
hybrid parallelisation model with both multi-threading and MPI distributed memory communication.
PyMPDATA-MPI adapts to API of PyMPDATA offering domain decomposition logic.

## Hello world examples

In a minimal setup, PyMPDATA-MPI can be used to solve the following transport equation: 
$$\partial_t (G \psi) + \nabla \cdot (Gu \psi)= 0$$
in an environment with multiple nodes.
Every node (process) is responsible for computing its part of the decomposed domain.

### Spherical scenario (2D)

In spherical geometry, the $G$ factor represents the Jacobian of coordinate transformation.
In this example (based on a test case from [Williamson & Rasch 1989](https://doi.org/10.1175/1520-0493(1989)117<0102:TDSLTW>2.0.CO;2)),
  domain decomposition is done cutting the sphere along meridians.
The inner dimension uses the [`MPIPolar`](https://open-atmos.github.io/PyMPDATA-MPI/mpi_polar.html) 
  boundary condition class, while the outer dimension uses
  [`MPIPeriodic`](https://open-atmos.github.io/PyMPDATA-MPI/mpi_periodic.html).
Note that the spherical animations below depict simulations without MPDATA corrective iterations,
  i.e. only plain first-order upwind scheme is used (FIX ME).

### 1 worker
<p align="middle">
  <img src="https://github.com/open-atmos/PyMPDATA-MPI/releases/download/latest-generated-plots/n_iters.1_rank_0_size_1_c_field_.0.5.0.25.-SphericalScenario-anim.gif" width="49%" /> 
</p>

### 2 workers
<p align="middle">
  <img src="https://github.com/open-atmos/PyMPDATA-MPI/releases/download/latest-generated-plots/n_iters.1_rank_1_size_2_c_field_.0.5.0.25.-SphericalScenario-anim.gif" width="49%" /> 
  <img src="https://github.com/open-atmos/PyMPDATA-MPI/releases/download/latest-generated-plots/n_iters.1_rank_0_size_2_c_field_.0.5.0.25.-SphericalScenario-anim.gif" width="49%" />
</p>

### Cartesian scenario (2D)

In the cartesian example below (based on a test case from [Arabas et al. 2014](https://doi.org/10.3233/SPR-140379)),
  a constant advector field $u$ is used (and $G=1$).
MPI (Message Passing Interface) is used 
  for handling data transfers and synchronisation with the domain decomposition
  across MPI workers done in either inner or in the outer dimension (user setting).
Multi-threading (using, e.g., OpenMP via Numba) is used for shared-memory parallelisation 
  within subdomains with further subdomain split across the inner dimension (PyMPDATA logic).
In this example, two corrective MPDATA iterations are employed.

### 1 worker
<p align="middle">
  <img src="https://github.com/open-atmos/PyMPDATA-MPI/releases/download/latest-generated-plots/n_iters.3_rank_0_size_1_c_field_.0.5.0.25.-CartesianScenario-anim.gif" width="49%" /> 
</p>

### 2 workers
<p align="middle">
  <img src="https://github.com/open-atmos/PyMPDATA-MPI/releases/download/latest-generated-plots/n_iters.3_rank_0_size_2_c_field_.0.5.0.25.-CartesianScenario-anim.gif" width="49%" />
  <img src="https://github.com/open-atmos/PyMPDATA-MPI/releases/download/latest-generated-plots/n_iters.3_rank_1_size_2_c_field_.0.5.0.25.-CartesianScenario-anim.gif" width="49%" /> 
</p>

### 3 workers
<p align="middle">
  <img src="https://github.com/open-atmos/PyMPDATA-MPI/releases/download/latest-generated-plots/n_iters.3_rank_0_size_3_c_field_.0.5.0.25.-CartesianScenario-anim.gif" width="32%" />
  <img src="https://github.com/open-atmos/PyMPDATA-MPI/releases/download/latest-generated-plots/n_iters.3_rank_1_size_3_c_field_.0.5.0.25.-CartesianScenario-anim.gif" width="32%" />
  <img src="https://github.com/open-atmos/PyMPDATA-MPI/releases/download/latest-generated-plots/n_iters.3_rank_2_size_3_c_field_.0.5.0.25.-CartesianScenario-anim.gif" width="32%" />
</p>

## Package architecture

```mermaid
    flowchart BT

    H5PY ---> HDF{{HDF5}}
    subgraph pythonic-dependencies [Python]
      TESTS --> H[pytest-mpi]
      subgraph PyMPDATA-MPI ["PyMPDATA-MPI"]
        TESTS["PyMPDATA-MPI[tests]"] --> CASES(simulation scenarios)
        A1["PyMPDATA-MPI[examples]"] --> CASES
        CASES --> D[PyMPDATA-MPI]
      end
      A1 ---> C[py-modelrunner]
      CASES ---> H5PY[h5py]
      D --> E[numba-mpi]
      H --> X[pytest]
      E --> N
      F --> N[Numba]
      D --> F[PyMPDATA]
    end
    H ---> MPI
    C ---> slurm{{slurm}}
    N --> OMPI{{OpenMP}}
    N --> L{{LLVM}}
    E ---> MPI{{MPI}}
    HDF --> MPI
    slurm --> MPI

style D fill:#7ae7ff,stroke-width:2px,color:#2B2B2B

click H "https://pypi.org/p/pytest-mpi"
click X "https://pypi.org/p/pytest"
click F "https://pypi.org/p/PyMPDATA"
click N "https://pypi.org/p/numba"
click C "https://pypi.org/p/py-modelrunner"
click H5PY "https://pypi.org/p/h5py"
click E "https://pypi.org/p/numba-mpi"
click A1 "https://pypi.org/p/PyMPDATA-MPI"
click D "https://pypi.org/p/PyMPDATA-MPI"
click TESTS "https://pypi.org/p/PyMPDATA-MPI"
```
Rectangular boxes indicate pip-installable Python packages (click to go to pypi.org package site).
## Credits:

Development of PyMPDATA-MPI has been supported by the [Poland's National Science Centre](https://www.ncn.gov.pl/?language=en)  
(grant no. 2020/39/D/ST10/01220).

copyright: [Jagiellonian University](https://en.uj.edu.pl/en) & [AGH University of Krakow](https://agh.edu.pl/en)   
licence: [GPL v3](https://www.gnu.org/licenses/gpl-3.0.html)

## Design goals

- MPI support for PyMPDATA implemented externally (i.e., not incurring any overhead or additional dependencies for PyMPDATA users)
- MPI calls within Numba njitted code (hence not using `mpi4py`, but leveraging `numba-mpi`)
- hybrid domain decomposition parallelisation: threading (internal in PyMPDATA, in the inner dimension) + MPI (either inner or outer dimension)
- portability across major OSes (currently Linux & macOS; no Windows support due [challenges in getting HDF5/MPI-IO to work there](https://docs.h5py.org/en/stable/build.html#source-installation-on-windows))
- full test coverage including CI builds asserting on same results with multi-node vs. single-node computations
- Continuous Integration with different OSes and different MPI implementation

## Related resources

### open-source Large-Eddy-Simulation and related software

#### Julia
- https://github.com/CliMA/ClimateMachine.jl/
#### C++
- https://github.com/microhh/microhh
- https://github.com/igfuw/UWLCM
#### C/CUDA
- https://github.com/NCAR/FastEddy-model
#### FORTRAN
- https://github.com/dalesteam/dales
- https://github.com/uclales/uclales
- https://github.com/UCLALES-SALSA/UCLALES-SALSA
- https://github.com/igfuw/bE_SDs
- https://github.com/pencil-code/pencil-code
- https://github.com/AtmosFOAM/AtmosFOAM
- https://github.com/scale-met/scale
#### Python (incl. Cython) 
- https://github.com/CliMA/pycles
- https://github.com/pnnl/pinacles
