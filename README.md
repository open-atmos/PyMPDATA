[![Build Status](https://travis-ci.org/atmos-cloud-sim-uj/MPyDATA.svg?branch=master)](https://travis-ci.org/atmos-cloud-sim-uj/MPyDATA)
[![Coverage Status](https://img.shields.io/codecov/c/github/atmos-cloud-sim-uj/MPyDATA/master.svg)](https://codecov.io/github/atmos-cloud-sim-uj/MPyDATA?branch=master)

# MPyDATA

MPyDATA is a high-performance Numba-accelerated Pythonic implementation of the MPDATA 
  algorithm of Smolarkiewicz et al. for numerically solving generalised transport equations.
As of the current version, it supports 1D and 2D integration on structured meshes optionally
  abstracted through Jacobian of coordinate transformation. 
MPyDATA includes implementation of a set of MPDATA algorithm variants including
  flux-corrected transport (FCT), infinite-gauge, divergent-flow and 
  third-order-terms options. 
It also features support for integration of Fickian-terms in advection-diffusion
  problems using the pseudo-transport velocity approach.
No domain-decomposition parallelism supported yet.

MPyDATA is engineered purely in Python targeting both performance and usability,
    the latter encompassing research users', developers' and maintainers' perspectives.
From researcher's perspective, MPyDATA offers hassle-free installation on multitude
  of platforms including Linux, OSX and Windows, and eliminates compilation stage
  from the perspective of the user.
From developers' and maintainers' perspective, MPyDATA offers wide unit-test coverage, 
  multi-platform continuous integration setup,
  seamless integration with Python debugging and profiling tools, and robust susceptibility
  to static code analysis within integrated development environments.

MPyDATA design features
  a custom-built multi-dimensional Arakawa-C grid layer allowing
  to concisely represent multi-dimensional stencil operations.
The grid layer is built on top of NumPy's ndarrays using Numba's @njit
  functionality and has been carefully profiled for performance.
It enables one to code once for multiple dimensions, and automatically
  handles (and hides from the user) any halo-filling logic related with boundary conditions.

MPyDATA ships with a set of examples/demos offered as github-hosted Jupyer notebooks
  offering single-click deployment in the cloud using such platforms as
  mybinder.org.
The examples/demos reproduce results from several published
  works on MPDATA and its applications, and provide a validation of the implementation
  and its performance.
 
## Examples/Demos:
- [Smolarkiewicz 2006](http://doi.org/10.1002/fld.1071) Figs 3,4,10,11 & 12
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/atmos-cloud-sim-uj/MPyDATA.git/master?filepath=MPyDATA_examples%2FSmolarkiewicz_2006_Figs_3_4_10_11_12/demo.ipynb)
- [Arabas & Farhat 2020](https://doi.org/10.1016/j.cam.2019.05.023) Figs 1-3 & Tab. 1 
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/atmos-cloud-sim-uj/MPyDATA.git/master?filepath=MPyDATA_examples%2FArabas_and_Farhat_2020/)

## Credits:
Development of MPyDATA is supported by the EU through a grant of the Foundation for Polish Science (POIR.04.04.00-00-5E1C/18).

copyright: Jagiellonian University   
code licence: GPL v3   
tutorials licence: CC-BY

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
