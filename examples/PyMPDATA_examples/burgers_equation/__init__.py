"""
PyMPDATA 1D Burgers' equation example with gif creation.

burgers-equation.ipynb:
.. include:: ./burgers_equation.ipynb.badges.md
"""
from .burgers_equation import (
    calculate_analytical_solutions,
    plot_analytical_solutions,
    plot_numerical_vs_analytical,
    plot_gif,
    run_numerical_simulation,
    NT, NX, T_MAX, T_RANGE, T_SHOCK,
)
