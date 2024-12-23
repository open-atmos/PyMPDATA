# <img src="https://raw.githubusercontent.com/open-atmos/PyMPDATA/main/.github/pympdata_logo.svg" width=100 height=113 alt="pympdata logo">
# PyMPDATA

[![Python 3](https://img.shields.io/static/v1?label=Python&logo=Python&color=3776AB&message=3)](https://www.python.org/)
[![LLVM](https://img.shields.io/static/v1?label=LLVM&logo=LLVM&color=gold&message=Numba)](https://www.numba.org)
[![Linux OK](https://img.shields.io/static/v1?label=Linux&logo=Linux&color=yellow&message=%E2%9C%93)](https://en.wikipedia.org/wiki/Linux)
[![macOS OK](https://img.shields.io/static/v1?label=macOS&logo=Apple&color=silver&message=%E2%9C%93)](https://en.wikipedia.org/wiki/macOS)
[![Windows OK](https://img.shields.io/static/v1?label=Windows&logo=Windows&color=white&message=%E2%9C%93)](https://en.wikipedia.org/wiki/Windows)
[![Jupyter](https://img.shields.io/static/v1?label=Jupyter&logo=Jupyter&color=f37626&message=%E2%9C%93)](https://jupyter.org/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/open-atmos/PyMPDATA/graphs/commit-activity)
[![OpenHub](https://www.openhub.net/p/atmos-cloud-sim-uj-PyMPDATA/widgets/project_thin_badge?format=gif)](https://www.openhub.net/p/atmos-cloud-sim-uj-PyMPDATA)   
[![status](https://joss.theoj.org/papers/10e7361e43785dbb1b3d659c5b01757a/status.svg)](https://joss.theoj.org/papers/10e7361e43785dbb1b3d659c5b01757a)
[![DOI](https://zenodo.org/badge/225610671.svg)](https://zenodo.org/badge/latestdoi/225610671)     
[![EU Funding](https://img.shields.io/static/v1?label=EU%20Funding%20by&color=103069&message=FNP&logoWidth=25&logo=image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC4AAAAeCAYAAABTwyyaAAAEzklEQVRYw9WYS2yUVRiGn3P5ZzozpZ3aUsrNgoKlKBINmkhpCCwwxIAhsDCpBBIWhmCMMYTEhSJ4i9EgBnSBEm81MRrFBhNXEuUSMCopiRWLQqEGLNgr085M5//POS46NNYFzHQ6qGc1i5nzP/P973m/9ztCrf7A8T9csiibCocUbvTzfxLcAcaM3cY3imXz25lT3Y34G7gQYAKV3+bFAHcATlBTPogJNADG92iY28FHW97kyPbnuW/W7xgzAhukQ9xe04PJeOT0HkQRwK0TlEeGWb/kOO9v3kdD3a8YK9GhDMfa6mg9fxunOm/lWPtcpDI4K7n/jnN8+uQbrFrUSiwU/DtSEUB/MsKKBT+zYslJqiYNgVE4JwhHkzy86wlWvrKVWDSZ/YFjZlU39yw4y/rGoyQGowWB67zl4QQue+jssMdXrQvZ/00jyeHwqCgDKwnsiJjSvkYAxsG5K9WsenYbJdqAtAjhCIxCSZt/4fK1w5A2WCvxrUAKCHwNVoA2aGmvq11jJQQapEXrgMBKqmJJugejKGWLIxXrBPFoigfv/omd675gRkU/xgqUDlAhH3UDaAAlLSqUQekAYyVTyhLs3tDMsvntlIYzOFcEcOcEGd9jx9oDbGs6QO0t/Tijxi9S4bhzxiWaVh5m94Zm0n7oui4ybo0raUlcncQnxx+g+WgDF/vLoYDmoqSl/dJUnt7XRCoTZjij0Z6Pc2LiNS4EBBkNvoeOJXN+yPWWSZeANOhwJq/98nKVwNdoL8B5AROxBKBL0gjh8DMhdCh3eJnrA0yqhLpplwmyup6IajvAOIGfKGVx3VmCRGnOMpe5QAdG0bT8CAeeep0d6z6nqjSJnQiZWEllLMWrmz6k+fE9rGk8MVqYgsGv5ZH2i1Opr+9kajzB5d74hKQ+KS3d/WVMLhtgdu1lriRiOR/4nDVunaR24x7qp3UV5Cb/fJvC83nv26W81LIK58SYNFmwq4hsGx/5BwKlzYRma2NUthgOJSew4i7ru9nJYCQF5tApb2yvjiDQKJV/IfJKh0o6qssSLKv/jcAoRKHQQzE2Lj2OMV5OkWFc4MZIpsev8uXWXRx6ZicbGk8QZLxxgwe+x/rlR3h3816+f2E7lbEU+ZDn3vKVpePCdFovzCISHqbl5EIoQOteKMPB1rto65zNyfOz+KOrGl06lHPQyi/WOohH0/T0l1MZH6A3GUEKl7Pmr2la6wBrBWWRDP2DUcqjKVKBGom9RZmABAykwnglafpSJSPQvsfiOR0EQ7ExVmazA8cY6N4K1iw6RdAXRwi4mgrheT5Dvs4LeuS81a15Ll/3dQisFVSVpnj7sf1sX/sZvhAc+6UOrQyBVUQ8gx/orFmDsZqtaw/y1qZ9zKjp5vDpenyjcNe+cLNmTiUdf/bEOddVQ0VpgsOn54ET+EYxvWKALSu+5tGG76it7MNaiZKGQ23zCIcMfUMxBnrjN3fmHHvCAlp+vJcXWx6itqoXpAEnUNLx8iMfo5Xh1i17R3PJYCpC2cZ3qK3sQ8WGEDDuXlAQuFKGHzpmopXhTNfk0bmxs7uC1w6uJul79AxFkMIiBJy5UoUWjrZLU5DCFdTARDHuDqVw+OkSwI0MCEW4gtNF2BPrBCo8fKNbtILWX9aUDqFqHnn7AAAAAElFTkSuQmCC)](https://www.fnp.org.pl/en/)
[![PL Funding](https://img.shields.io/static/v1?label=PL%20Funding%20by&color=d21132&message=NCN&logoWidth=25&logo=image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAANCAYAAACpUE5eAAAABmJLR0QA/wD/AP+gvaeTAAAAKUlEQVQ4jWP8////fwYqAiZqGjZqIHUAy4dJS6lqIOMdEZvRZDPcDQQAb3cIaY1Sbi4AAAAASUVORK5CYII=)](https://www.ncn.gov.pl/?language=en)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.html)

[![Github Actions Build Status](https://github.com/open-atmos/PyMPDATA/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/open-atmos/PyMPDATA/actions)
[![Appveyor Build status](http://ci.appveyor.com/api/projects/status/github/open-atmos/PyMPDATA?branch=main&svg=true)](https://ci.appveyor.com/project/slayoo/pympdata/branch/main)
[![Coverage Status](https://codecov.io/gh/open-atmos/PyMPDATA/branch/main/graph/badge.svg)](https://app.codecov.io/gh/open-atmos/PyMPDATA)

[![PyPI version](https://badge.fury.io/py/PyMPDATA.svg)](https://pypi.org/project/PyMPDATA)
[![API docs](https://shields.mitmproxy.org/badge/docs-pdoc.dev-brightgreen.svg)](https://open-atmos.github.io/PyMPDATA/index.html)


PyMPDATA is a high-performance Numba-accelerated Pythonic implementation of the MPDATA 
algorithm of Smolarkiewicz et al. used in geophysical fluid dynamics and beyond for 
numerically solving generalised convection-diffusion PDEs in 1D, 2D and 3D structured meshes 
with coordinate transformations.

In short, PyMPDATA numerically solves the following equation:

$$ \partial_t (G \psi) + \nabla \cdot (Gu \psi) + \mu \Delta (G \psi) = 0 $$

where scalar field $\psi$ is referred to as the advectee,
vector field u is referred to as advector, and the G factor corresponds to optional coordinate transformation.
The inclusion of the Fickian diffusion term is optional and is realised through modification of the
advective velocity field with MPDATA handling both the advection and diffusion (for discussion
see, e.g. [Smolarkiewicz and Margolin 1998](https://doi.org/10.1006/jcph.1998.5901), sec. 3.5, par. 4).

PyMPDATA [documentation](https://open-atmos.github.io/PyMPDATA/index.html) is generated via [``pdoc``](https://pdoc.dev/).

A separate project called [``PyMPDATA-MPI``](https://github.com/open-atmos/PyMPDATA-MPI) 
  depicts how [``numba-mpi``](https://pypi.org/project/numba-mpi) can be used
  to enable distributed memory parallelism in PyMPDATA.applications, and provide a validation of the implementation
  and its performance.

## Dependencies and installation

To install PyMPDATA, one may use: ``pip install PyMPDATA`` (or 
``pip install git+https://github.com/open-atmos/PyMPDATA.git`` to get updates beyond the latest release).
PyMPDATA depends on ``NumPy`` and ``Numba``.

Running the tests shipped with the package requires additional packages that are installed
if pip is invoked with: ``pip install PyMPDATA[tests]``.

## Examples (Jupyter notebooks reproducing results from literature):

PyMPDATA examples are bundled with PyMPDATA and located in the `examples` subfolder.
They constitute a separate [``PyMPDATA_examples``](https://pypi.org/p/PyMPDATA-examples) Python package which is also available at PyPI.
The examples have additional dependencies listed in [``PyMPDATA_examples`` package ``setup.py``](https://github.com/open-atmos/PyMPDATA/blob/main/examples/setup.py) file.
Running the examples requires the ``PyMPDATA_examples`` package to be installed.
Since the examples package includes Jupyter notebooks (and their execution requires write access), the suggested install and launch steps are:
```
git clone https://github.com/open-atmos/PyMPDATA-examples.git
cd PyMPDATA-examples
pip install -e .
jupyter-notebook
```
Alternatively, one can also install the examples package from pypi.org by using ``pip install PyMPDATA-examples``.
  
## Package structure and API:

The key classes constituting the PyMPDATA interface are summarised below.

#### Options class

The [``Options``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/options.html) class
groups both algorithm variant options as well as some implementation-related
flags.

#### Arakawa-C grid layer

In PyMPDATA, the solution domain is assumed to extend from the
first cell's boundary to the last cell's boundary (thus the
first scalar field value is at $\[\Delta x/2, \Delta y/2\]$.
The [``ScalarField``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/scalar_field.html)
and [``VectorField``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/vector_field.html) classes implement the
[Arakawa-C staggered grid](https://en.wikipedia.org/wiki/Arakawa_grids#Arakawa_C-grid) logic.

#### Boundary conditions

Boundary conditions are implemented as classes defined in 
[``BoundaryCondition``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/boundary_conditions.html).

#### Stepper

The logic of the MPDATA iterative solver is represented
in PyMPDATA by the [``Stepper``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/stepper.html) class.

#### Solver

Instances of the [``Solver``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/solver.html) class are used to control
the integration and access solution data. During instantiation, 
additional memory required by the solver is 
allocated according to the options provided.

## Contributing, reporting issues, seeking support 

Submitting new code to the project, please preferably use [GitHub pull requests](https://github.com/open-atmos/PyMPDATA/pulls) 
(or the [PyMPDATA-examples PR site](https://github.com/open-atmos/PyMPDATA-examples/pulls) if working on examples) - it helps to keep record of code authorship, 
track and archive the code review workflow and allows to benefit
from the continuous integration setup which automates execution of tests 
with the newly added code. 

As of now, the copyright to the entire PyMPDATA codebase is with the Jagiellonian
University (2019-2023) and AGH University of Krakow (2023 onwards) - work places of the main maintainer.
Code contributions are assumed to imply transfer of copyright.
Should there be a need to make an exception, please indicate it when creating
a pull request or contributing code in any other way. In any case, 
the license of the contributed code must be compatible with GPL v3.

Developing the code, we follow [The Way of Python](https://www.python.org/dev/peps/pep-0020/) and 
the [KISS principle](https://en.wikipedia.org/wiki/KISS_principle).
The codebase has greatly benefited from [PyCharm code inspections](https://www.jetbrains.com/help/pycharm/code-inspection.html)
and [Pylint](https://pylint.org) code analysis (Pylint checks are part of the
CI workflows).

Issues regarding any incorrect, unintuitive or undocumented bahaviour of
PyMPDATA are best to be reported on the [GitHub issue tracker](https://github.com/open-atmos/PyMPDATA/issues/new).
Feature requests are recorded in the "Ideas..." [PyMPDATA wiki page](https://github.com/open-atmos/PyMPDATA/wiki/Ideas-for-new-features-and-examples).

We encourage to use the [GitHub Discussions](https://github.com/open-atmos/PyMPDATA/discussions) feature
(rather than the issue tracker) for seeking support in understanding, using and extending PyMPDATA code.

Please use the PyMPDATA issue-tracking and dicsussion infrastructure for `PyMPDATA-examples` as well.
We look forward to your contributions and feedback.

## Credits:
Development of PyMPDATA was supported by the EU through a grant of the [Foundation for Polish Science](http://fnp.org.pl) (POIR.04.04.00-00-5E1C/18) 
and by the [Polish National Science Centre](https://ncn.gov.pl/en) (grant no. 2020/39/D/ST10/01220)

copyright: Jagiellonian University (2019-2023) & AGH University of Krakow (2023 onwards)   
licence: GPL v3   

