"""
Created at 07.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""
from .vector_fields_utils import _is_integral
from .field import Field
import numpy as np

from ...utils import debug_flag

if debug_flag.VALUE:
    import MPyDATA.utils.fake_numba as numba
else:
    import numba


def make_vector_field_1d(arg_data_0: np.ndarray, arg_halo: int):
    assert arg_halo > 0

    halo = int(arg_halo)
    shape_0 = int(arg_data_0.shape[0] - 1)

    @numba.jitclass([
        ('_data_0', numba.float64[:]),
        ('_i', numba.int64),
        ('axis', numba.int64)
    ])
    class VectorField1D:
        def __init__(self, data_0: np.ndarray):
            self.axis = 0
            self._data_0 = np.zeros((shape_0 + 1 + 2 * (halo - 1)), dtype=np.float64)
            self._i = 0
            self.get_component(0)[:] = data_0[:]

        def clone(self):
            return VectorField1D(self.get_component(0).copy())

        def data(self, i) -> np.ndarray:
            if i == 0:
                return self._data_0
            else:
                raise ValueError()

        @property
        def dimension(self):
            return 1

        def focus(self, i):
            self._i = i + halo - 1

        def at(self, item: float, _):
            idx = self.__idx(item)
            return self._data_0[idx]

        def __idx(self, item: float):
            if _is_integral(item):
                raise ValueError()
            return self._i + int(item + .5)

        def get_item(self, focus: int, relative: float):
            self.focus(focus if focus >= 0 else shape_0 + focus)
            return self.at(relative, -1)

        def get_component(self, _):
            return self._data_0[halo - 1: self._data_0.shape[0] - halo + 1]

        def apply_1arg(self, function: callable, arg_1: Field.Impl, ext: int):
            for i in range(-1 - ext, shape_0 + ext):
                self.focus(i)
                arg_1.focus(i)

                idx = self.__idx(+.5)
                self._data_0[idx] = function(arg_1)

        def apply_2arg(self, function: callable, arg_1: Field.Impl, arg_2: Field.Impl, ext: int):
            for i in range(-1 - ext, shape_0 + ext):
                self.focus(i)
                arg_1.focus(i)
                arg_2.focus(i)

                idx = self.__idx(+.5)
                self._data_0[idx] = function(arg_1, arg_2)

        def apply_3arg(self, function: callable, arg_1: Field.Impl, arg_2: Field.Impl, arg_3: Field.Impl, ext: int):
            for i in range(-1 - ext, shape_0 + ext):
                self.focus(i)
                arg_1.focus(i)
                arg_2.focus(i)
                arg_3.focus(i)

                idx = self.__idx(+.5)
                self._data_0[idx] = function(arg_1, arg_2, arg_3)

        def left_halo(self, _, __):
            return slice(0, halo - 1)

        def left_edge(self, _, __):
            return slice(halo - 1, 2 * (halo - 1))

        def right_edge(self, _, __):
            return slice((shape_0 + 1), (shape_0 + 1) + halo - 1)

        def right_halo(self, _, __):
            return slice((shape_0 + 1) + halo - 1, (shape_0 + 1) + 2 * (halo - 1))

    return VectorField1D(data_0=arg_data_0)
