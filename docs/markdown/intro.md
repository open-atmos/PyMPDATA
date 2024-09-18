<img src="https://raw.githubusercontent.com/open-atmos/PyMPDATA/main/.github/pympdata_logo.svg" width=100 height=113 alt="pympdata logo">

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

[![Github Actions Build Status](https://github.com/open-atmos/PyMPDATA/actions/workflows/tests+pypi.yml/badge.svg?branch=main)](https://github.com/open-atmos/PyMPDATA/actions)
[![Appveyor Build status](http://ci.appveyor.com/api/projects/status/github/open-atmos/PyMPDATA?branch=main&svg=true)](https://ci.appveyor.com/project/slayoo/pympdata/branch/main)
[![Coverage Status](https://codecov.io/gh/open-atmos/PyMPDATA/branch/main/graph/badge.svg)](https://app.codecov.io/gh/open-atmos/PyMPDATA)

[![PyPI version](https://badge.fury.io/py/PyMPDATA.svg)](https://pypi.org/project/PyMPDATA)
[![API docs](https://shields.mitmproxy.org/badge/docs-pdoc.dev-brightgreen.svg)](https://open-atmos.github.io/PyMPDATA/index.html)

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