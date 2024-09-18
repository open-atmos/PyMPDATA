#### Arakawa-C grid layer and boundary conditions

In PyMPDATA, the solution domain is assumed to extend from the
first cell's boundary to the last cell's boundary (thus the
first scalar field value is at $\[\Delta x/2, \Delta y/2\]$.
The [``ScalarField``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/scalar_field.html)
and [``VectorField``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/vector_field.html) classes implement the
[Arakawa-C staggered grid](https://en.wikipedia.org/wiki/Arakawa_grids#Arakawa_C-grid) logic
in which:
- scalar fields are discretised onto cell centres (one value per cell),
- vector field components are discretised onto cell walls.

The schematic of the employed grid/domain layout in two dimensions is given below
(with the Python code snippet generating the figure as a part of CI workflow):