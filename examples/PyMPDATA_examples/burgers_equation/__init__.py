"""
PyMPDATA 1D Burgers' equation example with gif creation.

burgers-equation.ipynb:
.. include:: ./burgers_equation.ipynb.badges.md
"""

from .burgers_equation import (
    NT,
    NX,
    T_MAX,
    T_RANGE,
    T_SHOCK,
    X_ANALYTIC,
    calculate_analytical_solutions,
    run_numerical_simulation,
    analytical_solution,
)
