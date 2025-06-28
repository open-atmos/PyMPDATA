# PyMPDATA-MPI
PyMPDATA-MPI constitutes a [PyMPDATA](https://github.com/open-atmos/PyMPDATA) +
[numba-mpi](https://github.com/numba-mpi/numba-mpi) coupler enabling numerical solutions
of transport equations with the MPDATA numerical scheme in a
hybrid parallelisation model with both multi-threading and MPI distributed memory communication.
PyMPDATA-MPI adapts to API of PyMPDATA offering domain decomposition logic.

## Dependencies and installation

Along with ``PyMPDATA`` dependencies, this package also depends on ``numba-mpi`` and ``h5py``.

To install PyMPDATA-MPI, first we recommend installing MPI-compatible ``h5py`` and then downloading the package via pip:
```
HDF5_MPI="ON" pip install --no-binary=h5py h5py
pip install PyMPDATA-MPI
```

## Examples gallery

We strongly encourage to take a look at out examples gallery availible on [Github](https://open-atmos.github.io/PyMPDATA/PyMPDATA_MPI.html).

## Contributing, reporting issues, seeking support 

Submitting new code to the project, please preferably use [GitHub pull requests](https://github.com/open-atmos/PyMPDATA/pulls) it helps to keep record of code authorship, 
track and archive the code review workflow and allows to benefit
from the continuous integration setup which automates execution of tests 
with the newly added code. 

## Design goals

- MPI support for [PyMPDATA](https://pypi.org/project/PyMPDATA/) implemented externally (i.e., not incurring any overhead or additional dependencies for PyMPDATA users)
- MPI calls within [Numba njitted code](https://numba.pydata.org/numba-doc/dev/reference/jit-compilation.html) (hence not using [`mpi4py`](https://mpi4py.readthedocs.io/), but rather [`numba-mpi`](https://pypi.org/p/numba-mpi/))
- hybrid domain-decomposition parallelism: threading (internal in PyMPDATA, in the inner dimension) + MPI (either inner or outer dimension)
- example simulation scenarios featuring HDF5/MPI-IO output storage (using [h5py](https://www.h5py.org/))
- [py-modelrunner](https://github.com/zwicker-group/py-modelrunner) simulation orchestration
- portability across Linux & macOS (no Windows support as of now due to [challenges in getting HDF5/MPI-IO to work there](https://docs.h5py.org/en/stable/build.html#source-installation-on-windows))
- Continuous Integration (CI) with different OSes and different MPI implementations (leveraging to mpi4py's [setup-mpi Github Action](https://github.com/mpi4py/setup-mpi/))
- full test coverage including CI builds asserting on same results with multi-node vs. single-node computations (with help of [pytest-mpi](https://pypi.org/p/pytest-mpi/))
- ships as a [pip-installable package](https://pypi.org/project/PyMPDATA-MPI) - aimed to be a dependency of domain-specific packages

## Credits & acknowledgments:

PyMPDATA-MPI started as a separate project for the [MSc thesis of Kacper Derlatka](https://www.ap.uj.edu.pl/diplomas/166883) ([@Delcior](https://github.com/Delcior)).
Integration of PyMPDATA-MPI into PyMPDATA repo was carried out as a part of BEng project of [Michał Wroński](https://github.com/Sfonxu/).

Development of PyMPDATA-MPI has been supported by the [Poland's National Science Centre](https://www.ncn.gov.pl/?language=en)
(grant no. 2020/39/D/ST10/01220).

We acknowledge Poland’s high-performance computing infrastructure [PLGrid](https://plgrid.pl/) (HPC Centers: [ACK Cyfronet AGH](https://www.cyfronet.pl/en/))
for providing computer facilities and support within computational grant no. PLG/2023/016369

copyright: [Jagiellonian University](https://en.uj.edu.pl/en) & [AGH University of Krakow](https://agh.edu.pl/en)
licence: [GPL v3](https://www.gnu.org/licenses/gpl-3.0.html)