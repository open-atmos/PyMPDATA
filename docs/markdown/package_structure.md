## Package structure and API:

In short, PyMPDATA numerically solves the following equation:

$$ \partial_t (G \psi) + \nabla \cdot (Gu \psi) + \mu \Delta (G \psi) = 0 $$

where scalar field $\psi$ is referred to as the advectee,
vector field u is referred to as advector, and the G factor corresponds to optional coordinate transformation.
The inclusion of the Fickian diffusion term is optional and is realised through modification of the
advective velocity field with MPDATA handling both the advection and diffusion (for discussion
see, e.g. [Smolarkiewicz and Margolin 1998](https://doi.org/10.1006/jcph.1998.5901), sec. 3.5, par. 4).

The key classes constituting the PyMPDATA interface are summarised below with code
snippets exemplifying usage of PyMPDATA from Python, Julia and Matlab.

A [pdoc-generated](https://pdoc.dev/) documentation of PyMPDATA public API is maintained at: [https://open-atmos.github.io/PyMPDATA](  https://open-atmos.github.io/PyMPDATA/index.html) 
