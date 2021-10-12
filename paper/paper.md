---
title: 'PyMPDATA v1: Numba-accelerated implementation of MPDATA with examples in Python, Julia and Matlab'
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
    orcid: 0000-0003-0361-0082
    affiliation: "2,1"
affiliations:
 - name: Jagiellonian University, Kraków, Poland 
   index: 1
 - name: University of Illinois at Urbana-Champaign, IL, USA
   index: 2
date: October 2021
bibliography: paper.bib

---

# Summary

Convection-diffusion problems arise across a wide range of pure and applied research,
  in particular in geosciences, aerospace engineering and financial modelling
  (for an overview of applications, see, e.g., section 1.1 in [@Morton_1996]).
One of the key challenges in numerical solutions of problems involving advective transport is
  the preservation of sign of the advected field (for an overview of this and other
  aspects of numerical solutions, see, e.g., [@Roed_2019]).
The Multidimensional Positive Definite Advection Transport Algorithm (MPDATA) is a robust 
  explicit-in-time sign-preserving solver introduced in [@Smolarkiewicz_1983] and [@Smolarkiewicz_1984].
MPDATA has been subsequently developed into a family of schemes with numerous variants 
  and solution procedures addressing diverse set of problems in geophysical fluid dynamics and beyond
  (for reviews of MPDATA applications and variants, see, e.g.,: [@Smolarkiewicz_and_Margolin_1998_JCP] and 
  [@Smolarkiewicz_2006]).
The PyMPDATA project introduced herein constitutes a high-performance multi-threaded implementation of
  structured-mesh MPDATA in Python.
PyMPDATA is built on top of Numba [@Lam_et_al_2015] which is a just-in-time compiler 
  that translates Python code into fast machine code using the Low Level Virtual Machine (LLVM)
  compiler infrastructure.
Thanks to extensive interoperability of Python, PyMPDATA is readily usable not only from within Python
  but also from such environments as Julia and Matlab, and the package comes with examples depicting it.
PyMPDATA is an open source software released under the terms of the GNU General Public License v3,
  and is available in the PyPI package repository.

# ...

[@Arabas_et_al_2014]
[@Jaruga_et_al_2015]
  
# variants
  
[@Beason_Margolin_1988]
[@Smolarkiewicz_and_Grabowski_1990]
[@Hill_2010]
[@Smolarkiewicz_and_Clark_1986]
[@Smolarkiewicz_and_Margolin_1993]
[@Smolarkiewicz_and_Margolin_1998_SIAM]
[@Margolin_and_Shashkov_2006]

# Usage examples

[@Jaruga_et_al_2015]
[@Jarecka_et_al_2015]
[@Arabas_and_Farhat_2020]
[@Olesik_et_al_2021]
[@Shipway_and_Hill_2012]
[@Williamson and Rasch 1989]
[@Molenkamp_1968]
[@Smolarkiewicz_1984]

# Author contributions

PB had been the architect of PyMPDATA v1 with SA taking the role of main developer and maintainer over the time.
MO participated in the package core development and led the development of the condensational-growth example which was the basis of his MSc thesis.
JB contributed the DPDC algorithm variant handling.
SD contributed the advection-diffusion example.
MM contributed to the numba-mpi package.
PR contributed the shallow-water example.
MS contributed the advection-on-a-sphere example
The paper was composed by SA and is based on the content of the README files of the PyMPDATA, PyMPDATA-examples and numba-mpi packages.

# Acknowledgements

Development of PyMPDATA has been carried out within the POWROTY/REINTEGRATION programme of the Foundation for Polish Science co-financed by the European Union under the European Regional Development Fund (POIR.04.04.00-00-5E1C/18).

# References
