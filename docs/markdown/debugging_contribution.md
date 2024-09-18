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

Submitting new code to the project, please preferably use [GitHub pull requests](https://github.com/open-atmos/PyMPDATA/pulls) 
(or the [PyMPDATA-examples PR site](https://github.com/open-atmos/PyMPDATA-examples/pulls) if working on examples) - it helps to keep record of code authorship, 
track and archive the code review workflow and allows to benefit
from the continuous integration setup which automates execution of tests 
with the newly added code. 

As of now, the copyright to the entire PyMPDATA codebase is with the Jagiellonian
University, and code contributions are assumed to imply transfer of copyright.
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
Development of PyMPDATA was supported by the EU through a grant of the [Foundation for Polish Science](http://fnp.org.pl) (POIR.04.04.00-00-5E1C/18).

copyright: Jagiellonian University   
licence: GPL v3   

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