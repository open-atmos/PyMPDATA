"""Provides base _Scenario base class that every scenario should inherit"""

from PyMPDATA import Solver
from PyMPDATA.impl.enumerations import INNER, OUTER


class _Scenario:  # pylint: disable=too-few-public-methods
    """Base class for every Scenario. Provides logic for advance() function"""

    # pylint: disable=too-many-arguments
    def __init__(self, *, mpi_dim, stepper, advectee, advector, g_factor=None):
        self.mpi_dim = mpi_dim
        self.solver = Solver(
            stepper=stepper, advectee=advectee, advector=advector, g_factor=g_factor
        )

    def advance(self, dataset, output_steps, mpi_range):
        """Logic for performing simulation. Returns wall time of one timestep (in clock ticks)"""
        steps_done = 0
        wall_time = 0
        for index, output_step in enumerate(output_steps):
            n_steps = output_step - steps_done
            if n_steps > 0:
                wall_time_per_timestep = self.solver.advance(n_steps=n_steps)
                wall_time += wall_time_per_timestep * n_steps
                steps_done += n_steps
            data = self.solver.advectee.get()
            dataset[
                (
                    mpi_range if self.mpi_dim == OUTER else slice(None),
                    mpi_range if self.mpi_dim == INNER else slice(None),
                    slice(index, index + 1),
                )
            ] = data.reshape((data.shape[0], data.shape[1], 1))
        return wall_time
