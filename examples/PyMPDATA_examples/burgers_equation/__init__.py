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
    calculate_analytical_solutions,
    plot_analytical_solutions,
    plot_gif,
    plot_numerical_vs_analytical,
    run_numerical_simulation,
)
