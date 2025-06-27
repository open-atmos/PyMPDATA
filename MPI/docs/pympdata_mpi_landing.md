# PyMPDATA-MPI
PyMPDATA-MPI constitutes a [PyMPDATA](https://github.com/open-atmos/PyMPDATA) +
[numba-mpi](https://github.com/numba-mpi/numba-mpi) coupler enabling numerical solutions
of transport equations with the MPDATA numerical scheme in a
hybrid parallelisation model with both multi-threading and MPI distributed memory communication.
PyMPDATA-MPI adapts to API of PyMPDATA offering domain decomposition logic.

# Example gallery
In a minimal setup, PyMPDATA-MPI can be used to solve the following transport equation:
$$\partial_t (G \psi) + \nabla \cdot (Gu \psi)= 0$$
in an environment with multiple nodes.
Every node (process) is responsible for computing its part of the decomposed domain.

## Spherical scenario (2D)
In spherical geometry, the $G$ factor represents the Jacobian of coordinate transformation.
In this example (based on a test case from [Williamson & Rasch 1989](https://doi.org/10.1175/1520-0493(1989)117%3C0102:TDSLTW%3E2.0.CO;2)),
  domain decomposition is done cutting the sphere along meridians.
The inner dimension uses the [`MPIPolar`](https://open-atmos.github.io/PyMPDATA-MPI/mpi_polar.html)
  boundary condition class, while the outer dimension uses
  [`MPIPeriodic`](https://open-atmos.github.io/PyMPDATA-MPI/mpi_periodic.html).
Note that the spherical animations below depict simulations without MPDATA corrective iterations,
  i.e. only plain first-order upwind scheme is used (FIX ME).

### 1 worker (n_threads = 1)
<p align="middle">
  <img src="https://github.com/open-atmos/PyMPDATA/releases/download/latest-generated-plots/n_iters.1_rank_0_size_1_c_field_.0.5.0.25._mpi_dim_0_n_threads_1-SphericalScenario-anim.gif" width="49%" />
</p>

### 2 workers (MPI_DIM = 0, n_threads = 1)
<p align="middle">
  <img src="https://github.com/open-atmos/PyMPDATA/releases/tip/latest-generated-plots/n_iters.1_rank_1_size_2_c_field_.0.5.0.25._mpi_dim_0_n_threads_1-SphericalScenario-anim.gif" width="49%" />
  <img src="https://github.com/open-atmos/PyMPDATA/releases/tip/latest-generated-plots/n_iters.1_rank_0_size_2_c_field_.0.5.0.25._mpi_dim_0_n_threads_1-SphericalScenario-anim.gif" width="49%" />
</p>

## Cartesian scenario (2D)
In the cartesian example below (based on a test case from [Arabas et al. 2014](https://doi.org/10.3233/SPR-140379)),
  a constant advector field $u$ is used (and $G=1$).
MPI (Message Passing Interface) is used
  for handling data transfers and synchronisation with the domain decomposition
  across MPI workers done in either inner or in the outer dimension (user setting).
Multi-threading (using, e.g., OpenMP via Numba) is used for shared-memory parallelisation
  within subdomains (indicated by dotted lines in the animations below) with threading subdomain
  split done across the inner dimension (internal PyMPDATA logic).
In this example, two corrective MPDATA iterations are employed.

### 1 worker (n_threads=3)
<p align="middle">
  <img src="https://github.com/open-atmos/PyMPDATA/releases/tip/latest-generated-plots/n_iters.3_rank_0_size_1_c_field_.0.5.0.25._mpi_dim_0_n_threads_3-CartesianScenario-anim.gif" width="49%" />
</p>

### 2 workers (MPI_DIM = OUTER, n_threads = 3)
<p align="middle">
  <img src="https://github.com/open-atmos/PyMPDATA/releases/tip/latest-generated-plots/n_iters.3_rank_0_size_2_c_field_.0.5.0.25._mpi_dim_0_n_threads_3-CartesianScenario-anim.gif" width="49%" />
  <img src="https://github.com/open-atmos/PyMPDATA/releases/tip/latest-generated-plots/n_iters.3_rank_1_size_2_c_field_.0.5.0.25._mpi_dim_0_n_threads_3-CartesianScenario-anim.gif" width="49%" />
</p>

### 2 workers (MPI_DIM = INNER, n_threads = 3)
<p align="middle">
  <img src="https://github.com/open-atmos/PyMPDATA/releases/tip/latest-generated-plots/n_iters.3_rank_0_size_2_c_field_.0.5.0.25._mpi_dim_-1_n_threads_3-CartesianScenario-anim.gif" width="49%" />
  <img src="https://github.com/open-atmos/PyMPDATA/releases/tip/latest-generated-plots/n_iters.3_rank_1_size_2_c_field_.0.5.0.25._mpi_dim_-1_n_threads_3-CartesianScenario-anim.gif" width="49%" />
</p>

### 3 workers (MPI_DIM = OUTER, n_threads = 3)
<p align="middle">
  <img src="https://github.com/open-atmos/PyMPDATA/releases/tip/latest-generated-plots/n_iters.3_rank_0_size_3_c_field_.0.5.0.25._mpi_dim_0_n_threads_3-CartesianScenario-anim.gif" width="32%" />
  <img src="https://github.com/open-atmos/PyMPDATA/releases/tip/latest-generated-plots/n_iters.3_rank_1_size_3_c_field_.0.5.0.25._mpi_dim_0_n_threads_3-CartesianScenario-anim.gif" width="32%" />
  <img src="https://github.com/open-atmos/PyMPDATA/releases/tip/latest-generated-plots/n_iters.3_rank_2_size_3_c_field_.0.5.0.25._mpi_dim_0_n_threads_3-CartesianScenario-anim.gif" width="32%" />
</p>

### 3 workers (MPI_DIM = INNER, n_threads = 3)
<p align="middle">
  <img src="https://github.com/open-atmos/PyMPDATA/releases/tip/latest-generated-plots/n_iters.3_rank_0_size_3_c_field_.0.5.0.25._mpi_dim_-1_n_threads_3-CartesianScenario-anim.gif" width="32%" />
  <img src="https://github.com/open-atmos/PyMPDATA/releases/tip/latest-generated-plots/n_iters.3_rank_1_size_3_c_field_.0.5.0.25._mpi_dim_-1_n_threads_3-CartesianScenario-anim.gif" width="32%" />
  <img src="https://github.com/open-atmos/PyMPDATA/releases/tip/latest-generated-plots/n_iters.3_rank_2_size_3_c_field_.0.5.0.25._mpi_dim_-1_n_threads_3-CartesianScenario-anim.gif" width="32%" />
</p>

## Shallow Water Scenario
The Shallow Water Scenario is based on a numerical test case from [Jarecka et al. 2015](https://doi.org/10.1016/j.jcp.2015.02.003). The scenario follows Cartesian Scenario, but implements its own solving algorithm to account for changes of the advector field induced by the advectee.

### MPDATA with "nonoscillatory" and "infinite-gauge" options, n_threads = 1, MPI_DIM=OUTER
<p align="middle">
  <img src="https://github.com/open-atmos/PyMPDATA/releases/tip/latest-generated-plots/n_iters.2_nonoscillatory.True_infinite_gauge.True_rank_0_size_2_c_field_None_mpi_dim_0_n_threads_1-ShallowWaterScenario-anim.gif", width="49%" />
  <img src="https://github.com/open-atmos/PyMPDATA/releases/tip/latest-generated-plots/n_iters.2_nonoscillatory.True_infinite_gauge.True_rank_1_size_2_c_field_None_mpi_dim_0_n_threads_1-ShallowWaterScenario-anim.gif", width="49%" />
</p>

### MPDATA with "nonoscillatory" and "infinite-gauge" options, n_threads = 1, MPI_DIM=INNER
<p align="middle">
  <img src="https://github.com/open-atmos/PyMPDATA/releases/tip/latest-generated-plots/n_iters.2_nonoscillatory.True_infinite_gauge.True_rank_0_size_2_c_field_None_mpi_dim_-1_n_threads_1-ShallowWaterScenario-anim.gif", width="49%" />
  <img src="https://github.com/open-atmos/PyMPDATA/releases/tip/latest-generated-plots/n_iters.2_nonoscillatory.True_infinite_gauge.True_rank_1_size_2_c_field_None_mpi_dim_-1_n_threads_1-ShallowWaterScenario-anim.gif", width="49%" />
</p>

# Package architecture
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

# Credits & acknowledgments:
PyMPDATA-MPI started as an [MSc project of Kacper Derlatka](https://www.ap.uj.edu.pl/diplomas/166883) ([@Delcior](https://github.com/Delcior)) mentored by [@slayoo](https://github.com/slayoo).

Development of PyMPDATA-MPI has been supported by the [Poland's National Science Centre](https://www.ncn.gov.pl/?language=en)
(grant no. 2020/39/D/ST10/01220).

We acknowledge Poland’s high-performance computing infrastructure [PLGrid](https://plgrid.pl/) (HPC Centers: [ACK Cyfronet AGH](https://www.cyfronet.pl/en/))
for providing computer facilities and support within computational grant no. PLG/2023/016369

copyright: [Jagiellonian University](https://en.uj.edu.pl/en) & [AGH University of Krakow](https://agh.edu.pl/en)
licence: [GPL v3](https://www.gnu.org/licenses/gpl-3.0.html)

# Design goals
- MPI support for [PyMPDATA](https://pypi.org/project/PyMPDATA/) implemented externally (i.e., not incurring any overhead or additional dependencies for PyMPDATA users)
- MPI calls within [Numba njitted code](https://numba.pydata.org/numba-doc/dev/reference/jit-compilation.html) (hence not using [`mpi4py`](https://mpi4py.readthedocs.io/), but rather [`numba-mpi`](https://pypi.org/p/numba-mpi/))
- hybrid domain-decomposition parallelism: threading (internal in PyMPDATA, in the inner dimension) + MPI (either inner or outer dimension)
- example simulation scenarios featuring HDF5/MPI-IO output storage (using [h5py](https://www.h5py.org/))
- [py-modelrunner](https://github.com/zwicker-group/py-modelrunner) simulation orchestration
- portability across Linux & macOS (no Windows support as of now due to [challenges in getting HDF5/MPI-IO to work there](https://docs.h5py.org/en/stable/build.html#source-installation-on-windows))
- Continuous Integration (CI) with different OSes and different MPI implementations (leveraging to mpi4py's [setup-mpi Github Action](https://github.com/mpi4py/setup-mpi/))
- full test coverage including CI builds asserting on same results with multi-node vs. single-node computations (with help of [pytest-mpi](https://pypi.org/p/pytest-mpi/))
- ships as a [pip-installable package](https://pypi.org/project/PyMPDATA-MPI) - aimed to be a dependency of domain-specific packages

# Related resources

### open-source Large-Eddy-Simulation and related software

#### Julia
- https://github.com/CliMA/ClimateMachine.jl/
#### C++
- https://github.com/microhh/microhh
- https://github.com/igfuw/UWLCM
- https://github.com/mrnorman/portUrb
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
- https://github.com/google-research/swirl-jatmos
- https://github.com/MetLab-HKUST/LEX/
