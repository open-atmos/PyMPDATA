[![Build Status](https://travis-ci.org/atmos-cloud-sim-uj/MPyDATA.svg?branch=master)](https://travis-ci.org/atmos-cloud-sim-uj/MPyDATA)
[![Coverage Status](https://img.shields.io/codecov/c/github/atmos-cloud-sim-uj/MPyDATA/master.svg)](https://codecov.io/github/atmos-cloud-sim-uj/MPyDATA?branch=master)

# MPyDATA

MPyDATA is a Numba-accelerated Pythonic implementation of the MPDATA algorithm of Smolarkiewicz et al.
As of the current version, it supports 1D and 2D integration without any parallelism.

## Demos:
- [Smolarkiewicz 2006](http://doi.org/10.1002/fld.1071) Figs 3,4,10,11 & 12
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/atmos-cloud-sim-uj/MPyDATA.git/master?filepath=MPyDATA_examples%2FSmolarkiewicz_2006_Figs_3_4_10_11_12/demo.ipynb)
- [Arabas & Farhat 2019](https://doi.org/10.1016/j.cam.2019.05.023) Figs 1-3 & Tab. 1   
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/atmos-cloud-sim-uj/MPyDATA.git/master?filepath=MPyDATA_examples%2FArabas_and_Farhat_2019/)

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