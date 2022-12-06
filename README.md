# PySuperDropletLES (pre-alpha)

[![Python 3](https://img.shields.io/static/v1?label=Python&logo=Python&color=3776AB&message=3)](https://www.python.org/)
[![LLVM](https://img.shields.io/static/v1?label=LLVM&logo=LLVM&color=gold&message=Numba)](https://numba.pydata.org)
[![Linux OK](https://img.shields.io/static/v1?label=Linux&logo=Linux&color=yellow&message=%E2%9C%93)](https://en.wikipedia.org/wiki/Linux)
[![macOS OK](https://img.shields.io/static/v1?label=macOS&logo=Apple&color=silver&message=%E2%9C%93)](https://en.wikipedia.org/wiki/macOS)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/atmos-cloud-sim-uj/PySuperDropletLES/graphs/commit-activity)

[![PL Funding](https://img.shields.io/static/v1?label=PL%20Funding%20by&color=d21132&message=NCN&logoWidth=25&logo=image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAANCAYAAACpUE5eAAAABmJLR0QA/wD/AP+gvaeTAAAAKUlEQVQ4jWP8////fwYqAiZqGjZqIHUAy4dJS6lqIOMdEZvRZDPcDQQAb3cIaY1Sbi4AAAAASUVORK5CYII=)](https://www.ncn.gov.pl/?language=en)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.html)
[![Copyright](https://img.shields.io/static/v1?label=Copyright&color=249fe2&message=Jagiellonian%20University&)](https://en.uj.edu.pl/)

[![GitHub issues](https://img.shields.io/github/issues-pr/atmos-cloud-sim-uj/PySuperDropletLES.svg?logo=github&logoColor=white)](https://github.com/atmos-cloud-sim-uj/PySuperDropletLES/pulls?q=)
[![GitHub issues](https://img.shields.io/github/issues-pr-closed/atmos-cloud-sim-uj/PySuperDropletLES.svg?logo=github&logoColor=white)](https://github.com/atmos-cloud-sim-uj/PySuperDropletLES/pulls?q=is:closed)    
[![GitHub issues](https://img.shields.io/github/issues/atmos-cloud-sim-uj/PySuperDropletLES.svg?logo=github&logoColor=white)](https://github.com/atmos-cloud-sim-uj/PySuperDropletLES/issues?q=)
[![GitHub issues](https://img.shields.io/github/issues-closed/atmos-cloud-sim-uj/PySuperDropletLES.svg?logo=github&logoColor=white)](https://github.com/atmos-cloud-sim-uj/PySuperDropletLES/issues?q=is:closed)   
[![Github Actions Build Status](https://github.com/atmos-cloud-sim-uj/PySuperDropletLES/workflows/main/badge.svg?branch=main)](https://github.com/atmos-cloud-sim-uj/PySuperDropletLES/actions)
[![PyPI version](https://badge.fury.io/py/PySuperDropletLES.svg)](https://pypi.org/project/PySuperDropletLES)
[![API docs](https://img.shields.io/badge/API_docs-pdoc3-blue.svg)](https://atmos-cloud-sim-uj.github.io/PySuperDropletLES/)

[PySDM](https://github.com/atmos-cloud-sim-uj/PySDM) + 
[PyMPDATA](https://github.com/atmos-cloud-sim-uj/PyMPDATA) +
[numba-mpi](https://github.com/atmos-cloud-sim-uj/numba-mpi) coupler sandbox (with a long-term goal of developing a pure-Python LES system)

## Credits:

Development of PySuperDropletLES has been supported by the [Poland's National Science Centre](https://www.ncn.gov.pl/?language=en)  
(grant no. 2020/39/D/ST10/01220 realised at the [Jagiellonian University](https://en.uj.edu.pl/en).

copyright: [Jagiellonian University](https://en.uj.edu.pl/en)    
licence: [GPL v3](https://www.gnu.org/licenses/gpl-3.0.html)

## Design goals

- MPI support for PyMPDATA implemented externally (i.e., within PySuperDropletLES)
- portability across major OSes (currently Linux & macOS; no Windows support due [challenges in getting HDF5/MPI-IO to work there](https://docs.h5py.org/en/stable/build.html#source-installation-on-windows))

## Related resources

### open-source Large-Eddy-Simulation and related software

#### Julia
- https://github.com/CliMA/ClimateMachine.jl/
#### C++
- https://github.com/microhh/microhh
- https://github.com/igfuw/UWLCM
#### FORTRAN
- https://github.com/dalesteam/dales
- https://github.com/uclales/uclales
- https://github.com/UCLALES-SALSA/UCLALES-SALSA
- https://github.com/igfuw/bE_SDs
- https://github.com/pencil-code/pencil-code
- https://github.com/AtmosFOAM/AtmosFOAM
- https://github.com/scale-met/scale
#### Python/Cython/C 
- https://github.com/CliMA/pycles
