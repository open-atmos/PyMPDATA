from PySDM import Builder
from PySDM.backends import CPU
from PySDM.impl.mesh import Mesh
import numpy as np
from PySDM.initialisation.sampling import spatial_sampling


class Env():
    def __init__(self, grid, size):
        self.mesh = Mesh(grid=grid, size=size)

    def register(self, builder):
        self.particulator = builder.particulator

    def init_attributes(self, n_sd, spatial_discretisation):
        positions = spatial_discretisation.sample(self.mesh.grid, n_sd)

        attributes = {
            'n': np.ones(n_sd),
            'volume': np.full(n_sd, np.nan),
        }
        (
            attributes["cell id"],
            attributes["cell origin"],
            attributes["position in cell"],
        ) = self.mesh.cellular_attributes(positions)
        return attributes


def test_domain3d(n_sd=44, grid=(5, 6, 7), size=(50, 60, 70)):
    # arrange
    builder = Builder(n_sd=n_sd, backend=CPU())
    env = Env(grid, size)
    builder.set_environment(env)
    attributes = env.init_attributes(n_sd, spatial_discretisation=spatial_sampling.Pseudorandom())

    # act
    particulator = builder.build(attributes=attributes, products=())

    # assert
    # TODO !
