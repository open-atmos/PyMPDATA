# pylint: disable=too-few-public-methods
""" Provides base _Scenario base class that every scenario should inherit """
from PyMPDATA import Solver


class _Scenario:
    """Base class for every Scenario. Provides logic for advance() function"""

    def __init__(self, *, stepper, advectee, advector, g_factor=None):
        self.solver = Solver(
            stepper=stepper, advectee=advectee, advector=advector, g_factor=g_factor
        )

    def advance(self, dataset, output_steps, x_range):
        """Logic for performing simulation. Returns wall time of one timestep (in clock ticks)"""
        steps_done = 0
        wall_time = 0
        for index, output_step in enumerate(output_steps):
            n_steps = output_step - steps_done
            if n_steps > 0:
                wall_time_per_timestep = self.solver.advance(n_steps=n_steps)
                wall_time += wall_time_per_timestep * n_steps
                steps_done += n_steps
            dataset[x_range, :, index] = self.solver.advectee.get()
        return wall_time
