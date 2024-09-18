The ``__init__`` methods of
[``ScalarField``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/scalar_field.html)
and [``VectorField``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/vector_field.html)
have the following signatures:
- [``ScalarField(data: np.ndarray, halo: int, boundary_conditions)``](https://github.com/open-atmos/PyMPDATA/blob/main/PyMPDATA/scalar_field.py)
- [``VectorField(data: Tuple[np.ndarray, ...], halo: int, boundary_conditions)``](https://github.com/open-atmos/PyMPDATA/blob/main/PyMPDATA/vector_field.py)
The ``data`` parameters are expected to be Numpy arrays or tuples of Numpy arrays, respectively.
The ``halo`` parameter is the extent of ghost-cell region that will surround the
data and will be used to implement boundary conditions. 
Its value (in practice 1 or 2) is
dependent on maximal stencil extent for the MPDATA variant used and
can be easily obtained using the ``Options.n_halo`` property.

As an example, the code below shows how to instantiate a scalar
and a vector field given a 2D constant-velocity problem,
using a grid of 24x24 points, Courant numbers of -0.5 and -0.25
in "x" and "y" directions, respectively, with periodic boundary 
conditions and with an initial Gaussian signal in the scalar field
(settings as in Fig.&nbsp;5 in [Arabas et al. 2014](https://doi.org/10.3233/SPR-140379)):