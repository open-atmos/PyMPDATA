"""
Created at 07.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from .field import Field
from ...arakawa_c.impl import vector_field_2d
import numpy as np

from ...utils import debug_flag
if debug_flag.VALUE:
    import MPyDATA.utils.fake_numba as numba
else:
    import numba


def make_scalar_field_2d(arg_data: np.ndarray, arg_halo: int):
    halo = int(arg_halo)
    shape = (arg_data.shape[0] + 2 * halo, arg_data.shape[1] + 2 * halo)

    @numba.jitclass([('data', numba.float64[:, :])])
    class ScalarField2D:
        def __init__(self, data: np.ndarray):
            self.data = np.zeros((shape[0], shape[1]), dtype=np.float64)
            self.get()[:, :] = data[:, :]

        def clone(self):
            return ScalarField2D(self.get().copy())

        def apply_1arg(self, arg1: Field.Impl, ext: int):
            # t0 = time()
            loop = True
            init = 0
            dims = range(2 if loop else 1)
            for i in range(-ext, shape[0] - 2 * halo + ext):
                for j in range(-ext, shape[1] - 2 * halo + ext):
                    # self.focus(i, j)
                    _i = i + halo
                    _j = j + halo

                    # arg1.focus(i, j)
                    arg1_i = i + halo - 1
                    arg1_j = j + halo - 1

                    self.data[_i, _j] = init
                    for dim in dims:
                        # self.axis = dim
                        # arg1.axis = dim
                        self.data[_i, _j] += -1 * (
                                vector_field_2d.at(arg1._data_0, arg1._data_1, arg1_i, arg1_j, dim, +.5, 0) -
                                vector_field_2d.at(arg1._data_0, arg1._data_1, arg1_i, arg1_j, dim, -.5, 0)
                        )
            # print(time() - t0, "apply_1arg()")

        @property
        def dimension(self) -> int:
            return 2

        def get(self):
            results = self.data[
                halo: self.data.shape[0] - halo,
                halo: self.data.shape[1] - halo
            ]
            return results

        # TODO: replace Nones with actual numbers (which are constant)
        # TODO: consider saving instances of the slices?
        def left_halo(self, d: int):
            if d == 0: return 0, halo, 0, shape[1]
            if d == 1: return 0, shape[0], 0, halo
            # raise ValueError()

        def right_halo(self, d: int):
            if d == 0: return -halo, shape[0], 0, shape[1]
            if d == 1: return 0, shape[0], -halo, shape[1]
            # raise ValueError()

        def left_edge(self, d: int):
            if d == 0: return halo, 2 * halo, 0, shape[1]
            if d == 1: return 0, shape[0], halo, 2 * halo
            # raise ValueError()

        def right_edge(self, d: int):
            if d == 0: return -2 * halo, -halo, 0, shape[1]
            if d == 1: return 0, shape[0], -2 * halo, -halo
            # raise ValueError()

    return ScalarField2D(data=arg_data)


@numba.njit()
def at(data, _i, _j, axis, arg1: int, arg2: int):

    if axis == 1:
        return data[_i + arg2, _j + arg1]
    else:
        return data[_i + arg1, _j + arg2]


