#### Stepper

The logic of the MPDATA iterative solver is represented
in PyMPDATA by the [``Stepper``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/stepper.html) class.

When instantiating the [``Stepper``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/stepper.html), the user has a choice 
of either supplying just the  number of dimensions or specialising the stepper for a given grid:
<details>
<summary>Julia code (click to expand)</summary>

```Julia
Stepper = pyimport("PyMPDATA").Stepper

stepper = Stepper(options=options, n_dims=2)
```
</details>
<details>
<summary>Matlab code (click to expand)</summary>

```Matlab
Stepper = py.importlib.import_module('PyMPDATA').Stepper;

stepper = Stepper(pyargs(...
  'options', options, ...
  'n_dims', int32(2) ...
));
```
</details>
<details open>
<summary>Python code (click to expand)</summary>

```Python
from PyMPDATA import Stepper

stepper = Stepper(options=options, n_dims=2)
```
</details>
or
<details>
<summary>Julia code (click to expand)</summary>

```Julia
stepper = Stepper(options=options, grid=(nx, ny))
```
</details>
<details>
<summary>Matlab code (click to expand)</summary>

```Matlab
stepper = Stepper(pyargs(...
  'options', options, ...
  'grid', py.tuple({nx, ny}) ...
));
```
</details>
<details open>
<summary>Python code (click to expand)</summary>

```Python
stepper = Stepper(options=options, grid=(nx, ny))
```
</details>

In the latter case, noticeably 
faster execution can be expected, however the resultant
stepper is less versatile as bound to the given grid size.
If number of dimensions is supplied only, the integration
might take longer, yet same instance of the
stepper can be used for different grids.  

Since creating an instance of the [``Stepper``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/stepper.html) class
involves time-consuming compilation of the algorithm code,
the class is equipped with a cache logic - subsequent
calls with same arguments return references to previously
instantiated objects. Instances of [``Stepper``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/stepper.html) contain no
mutable data and are (thread-)safe to be reused.

The init method of [``Stepper``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/stepper.html) has an optional
``non_unit_g_factor`` argument which is a Boolean flag 
enabling handling of the G factor term which can be used to 
represent coordinate transformations and/or variable fluid density. 

Optionally, the number of threads to use for domain decomposition
in the first (non-contiguous) dimension during 2D and 3D calculations
may be specified using the optional ``n_threads`` argument with a
default value of ``numba.get_num_threads()``. The multi-threaded
logic of PyMPDATA depends thus on settings of numba, namely on the
selected threading layer (either via ``NUMBA_THREADING_LAYER`` env 
var or via ``numba.config.THREADING_LAYER``) and the selected size of the 
thread pool (``NUMBA_NUM_THREADS`` env var or ``numba.config.NUMBA_NUM_THREADS``).


#### Solver

Instances of the [``Solver``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/solver.html) class are used to control
the integration and access solution data. During instantiation, 
additional memory required by the solver is 
allocated according to the options provided. 

The only method of the [``Solver``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/solver.html) class besides the
init is [``advance(n_steps, mu_coeff, ...)``](https://open-atmos.github.io/PyMPDATA/PyMPDATA/solver.html#Solver.advance) 
which advances the solution by ``n_steps`` timesteps, optionally
taking into account a given diffusion coefficient ``mu_coeff``.

Solution state is accessible through the ``Solver.advectee`` property.
Multiple solver[s] can share a single stepper, e.g., as exemplified in the shallow-water system solution in the examples package.

Continuing with the above code snippets, instantiating
a solver and making 75 integration steps looks as follows:
<details>
<summary>Julia code (click to expand)</summary>

```Julia
Solver = pyimport("PyMPDATA").Solver
solver = Solver(stepper=stepper, advectee=advectee, advector=advector)
solver.advance(n_steps=75)
state = solver.advectee.get()
```
</details>
<details>
<summary>Matlab code (click to expand)</summary>

```Matlab
Solver = py.importlib.import_module('PyMPDATA').Solver;
solver = Solver(pyargs('stepper', stepper, 'advectee', advectee, 'advector', advector));
solver.advance(pyargs('n_steps', 75));
state = solver.advectee.get();
```
</details>
<details open>
<summary>Python code (click to expand)</summary>

```Python
from PyMPDATA import Solver

solver = Solver(stepper=stepper, advectee=advectee, advector=advector)
state_0 = solver.advectee.get().copy()
solver.advance(n_steps=75)
state = solver.advectee.get()
```
</details>

Now let's plot the results using `matplotlib` roughly as in Fig.&nbsp;5 in [Arabas et al. 2014](https://doi.org/10.3233/SPR-140379):