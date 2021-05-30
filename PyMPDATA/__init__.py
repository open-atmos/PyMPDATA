from .options import Options
from .solver import Solver
from .stepper import Stepper

from .factories import Factories
from .arakawa_c.scalar_field import ScalarField
from .arakawa_c.vector_field import VectorField

from .arakawa_c.boundary_condition.periodic_boundary_condition import PeriodicBoundaryCondition
from .arakawa_c.boundary_condition.constant_boundary_condition import ConstantBoundaryCondition
from .arakawa_c.boundary_condition.extrapolated_boundary_condition import ExtrapolatedBoundaryCondition
from .arakawa_c.boundary_condition.polar_boundary_condition import PolarBoundaryCondition

from pkg_resources import get_distribution, DistributionNotFound, VersionConflict
try:
    __version__ = get_distribution(__name__).version
except (DistributionNotFound, VersionConflict):
    # package is not installed
    pass
