#### Debugging

PyMPDATA relies heavily on Numba to provide high-performance 
number crunching operations. Arguably, one of the key advantage 
of embracing Numba is that it can be easily switched off. This
brings multiple-order-of-magnitude drop in performance, yet 
it also make the entire code of the library susceptible to
interactive debugging, one way of enabling it is by setting the 
following environment variable before importing PyMPDATA:
<details>
<summary>Julia code (click to expand)</summary>

```Julia
ENV["NUMBA_DISABLE_JIT"] = "1"
```
</details>
<details>
<summary>Matlab code (click to expand)</summary>

```Matlab
setenv('NUMBA_DISABLE_JIT', '1');
```
</details>
<details open>
<summary>Python code (click to expand)</summary>

```Python
import os
os.environ["NUMBA_DISABLE_JIT"] = "1"
```
</details>

## Contributing, reporting issues, seeking support 

See [https://github.com/open-atmos/PyMPDATA/tree/main/README.md](README.md).

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
  https://github.com/AtmosFOAM/AtmosFOAM/tree/947b192f69d973ea4a7cfab077eb5c6c6fa8b0cf/applications/solvers/advection/MPDATAadvectionFoam

## Other Python packages for solving hyperbolic transport equations

- PyPDE: https://pypi.org/project/PyPDE/
- FiPy: https://pypi.org/project/FiPy/
- ader: https://pypi.org/project/ader/
- centpy: https://pypi.org/project/centpy/
- mattflow: https://pypi.org/project/mattflow/
- FastFD: https://pypi.org/project/FastFD/
- Pyclaw: https://www.clawpack.org/pyclaw/
