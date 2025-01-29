"""
This module showcases a PyMPDATA implementation of an MPDATA-based Boussinesq system solver
with Poisson equation for the pressure-term solved using a bespoke (libmpdata++-based) implementation
of the generalised conjugate-residual scheme. Simulation setup based on
[Smolarkiewicz & Pudykiewicz 1992](https://doi.org/10.1175/1520-0469(1992)049%3C2082:ACOSLA%3E2.0.CO;2)
(following Fig 19 in [Jaruga_et_al_2015](https://doi.org/10.5194/gmd-8-1005-2015)).

fig19.ipynb:
.. include:: ./fig19.ipynb.badges.md
"""
