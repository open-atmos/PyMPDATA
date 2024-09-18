## Options class

The [``Options``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/options.html) class
groups both algorithm variant options as well as some implementation-related
flags that need to be set at the first place. All are set at the time
of instantiation using the following keyword arguments of the constructor 
(all having default values indicated below):
- ``n_iters: int = 2``: number of iterations (2 means upwind + one corrective iteration)
- ``infinite_gauge: bool = False``: flag enabling the infinite-gauge option (does not maintain sign of the advected field, thus in practice implies switching flux corrected transport on)
- ``divergent_flow: bool = False``: flag enabling divergent-flow terms when calculating antidiffusive velocity
- ``nonoscillatory: bool = False``: flag enabling the non-oscillatory or monotone variant (a.k.a flux-corrected transport option, FCT)
- ``third_order_terms: bool = False``: flag enabling third-order terms
- ``epsilon: float = 1e-15``: value added to potentially zero-valued denominators 
- ``non_zero_mu_coeff: bool = False``: flag indicating if code for handling the Fickian term is to be optimised out
- ``DPDC: bool = False``: flag enabling double-pass donor cell option (recursive pseudovelocities)
- ``dimensionally_split: bool = False``: flag disabling cross-dimensional terms in antidiffusive velocity
- ``dtype: np.floating = np.float64``: floating point precision

For a discussion of the above options, see e.g., [Smolarkiewicz & Margolin 1998](https://doi.org/10.1006/jcph.1998.5901),
[Jaruga, Arabas et al. 2015](https://doi.org/10.5194/gmd-8-1005-2015) and [Olesik, Arabas et al. 2020](https://arxiv.org/abs/2011.14726)
(the last with examples using PyMPDATA).

In most use cases of PyMPDATA, the first thing to do is to instantiate the 
[``Options``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/options.html) class 
with arguments suiting the problem at hand, e.g.:

<details>
<summary>Julia code (click to expand)</summary>

```Julia
using Pkg
Pkg.add("PyCall")
using PyCall
Options = pyimport("PyMPDATA").Options
options = Options(n_iters=2)
```
</details>

<details>
<summary>Matlab code (click to expand)</summary>

```Matlab
Options = py.importlib.import_module('PyMPDATA').Options;
options = Options(pyargs('n_iters', 2));
```
</details>

<details open>
<summary>Python code (click to expand)</summary>

```Python
from PyMPDATA import Options
options = Options(n_iters=2)
```
</details>