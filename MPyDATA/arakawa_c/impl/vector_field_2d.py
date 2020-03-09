"""
Created at 07.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from .field import Field
from .vector_fields_utils import _is_integral, _is_fractional
from ...arakawa_c.impl import scalar_field_2d
import numpy as np

from ...utils import debug_flag
from MPyDATA.clock import time

if debug_flag.VALUE:
    import MPyDATA.utils.fake_numba as numba
else:
    import numba


@numba.njit()
def at(data_0, data_1, _i, _j, axis, arg1: [int, float], arg2: [int, float]):

    d, idx1, idx2 = idx(axis, _i, _j, arg1, arg2)

    return data(d, data_0, data_1)[idx1, idx2]


@numba.njit()
def data(i, data_0, data_1) -> np.ndarray:
    if i == 0:
        return data_0
    elif i == 1:
        return data_1
    else:
        raise ValueError()


@numba.njit()
def idx(axis, _i, _j, arg1: [int, float], arg2: [int, float]):
    if axis == 1:
        arg1, arg2 = arg2, arg1
    if _is_integral(arg1) and _is_fractional(arg2):
        d = 1
        idx1 = arg1 + 1
        idx2 = int(arg2 + .5)
    elif _is_integral(arg2) and _is_fractional(arg1):
        d = 0
        idx1 = int(arg1 + .5)
        idx2 = arg2 + 1
    # else:
    #     raise NotImplementedError()

    return d, int(_i + idx1), int(_j + idx2)


def make_vector_field_2d(arg_data, arg_halo: int):
    assert arg_data[0].shape[0] == arg_data[1].shape[0] + 1
    assert arg_data[0].shape[1] == arg_data[1].shape[1] - 1
    assert arg_halo >= 0

    halo = int(arg_halo)
    shape = (arg_data[1].shape[0], arg_data[0].shape[1])

    @numba.jitclass([
        ('_data_0', numba.float64[:, :]),
        ('_data_1', numba.float64[:, :]),
    ])
    class VectorField2D:
        def __init__(self, data_0: np.ndarray, data_1: np.ndarray):
            self._data_0 = np.full((
                data_0.shape[0] + 2 * (halo - 1),
                data_0.shape[1] + 2 * halo
            ), np.nan, dtype=np.float64)
            self._data_1 = np.full((
                data_1.shape[0] + 2 * halo,
                data_1.shape[1] + 2 * (halo - 1)
            ), np.nan, dtype=np.float64)
            self.get_component(0)[:, :] = data_0[:, :]
            self.get_component(1)[:, :] = data_1[:, :]

        def clone(self):
            return VectorField2D(
                self.get_component(0).copy(),
                self.get_component(1).copy()
            )

        def data(self, i):
            return data(i, self._data_0, self._data_1)

        @property
        def dimension(self) -> int:
            return 2

        def get_component(self, i: int) -> np.ndarray:
            domain = (
                slice(
                    halo - 1,
                    halo - 1 + shape[0] + 1
                ),
                slice(
                    halo,
                    halo + shape[1]
                )
            ) if i == 0 else (
                slice(
                    halo,
                    halo + shape[0]
                ),
                slice(
                    halo - 1,
                    halo - 1 + shape[1] + 1
                )
            )
            return data(i, self._data_0, self._data_1)[domain]

        def apply_2arg(self, arg_1: Field.Impl, arg_2: Field.Impl, ext: int):
            # t0 = time()
            for i in range(-1-ext, shape[0]+ext):
                for j in range(-1-ext, shape[1]+ext):
                    # self.focus(i, j)
                    _i = i + halo - 1
                    _j = j + halo - 1

                    # arg_1.focus(i, j)
                    arg_1_i = i + halo
                    arg_1_j = j + halo

                    # arg_2.focus(i, j)
                    arg_2_i = i + halo - 1
                    arg_2_j = j + halo - 1

                    for dd in range(2):
                        if (i == -1 and dd == 1) or (j == -1 and dd == 0):
                            continue
                        d, idx_i, idx_j = idx(dd, _i, _j, +.5, 0)
                        self.data(d)[idx_i, idx_j] = (
                                np.maximum(0, at(arg_2._data_0, arg_2._data_1, arg_2_i, arg_2_j, dd, +.5, 0)) *
                                scalar_field_2d.at(arg_1.data, arg_1_i, arg_1_j, dd, 0, 0) +
                                np.minimum(0, at(arg_2._data_0, arg_2._data_1, arg_2_i, arg_2_j, dd, +.5, 0)) *
                                scalar_field_2d.at(arg_1.data, arg_1_i, arg_1_j, dd, 1, 0)
                        )
            # print(time() - t0, "apply_2arg()")


        # TODO: replace Nones with actual numbers (which are constant)
        def left_halo(self, a: int, c: int):
            if c == 0:
                if a == 0: return slice(0, halo - 1), slice(None, None)
                if a == 1: return slice(None, None), slice(0, halo)
            if c == 1:
                if a == 0: return slice(0, halo), slice(None, None)
                if a == 1: return slice(None, None), slice(0, halo - 1)
            raise ValueError()

        def left_edge(self, a: int, c: int):
            if c == 0:
                if a == 0: return slice(halo - 1,2 * (halo - 1)), slice(None, None)
                if a == 1: return slice(None, None), slice(halo, 2 * halo)
            if c == 1:
                if a == 0: return slice(halo, 2 * halo), slice(None, None)
                if a == 1: return slice(None, None), slice(halo - 1, 2 * (halo - 1))
            raise ValueError()

        def right_halo(self, a: int, c: int):
            if c == 0:
                if a == 0: return slice(-(halo - 1), None), slice(None, None)
                if a == 1: return slice(None, None), slice(-halo, None)
            if c == 1:
                if a == 0: return slice(-halo, None), slice(None, None)
                if a == 1: return slice(None, None), slice(-(halo - 1), None)
            raise ValueError()

        def right_edge(self, a: int, c: int):
            if c == 0:
                if a == 0: return slice(-2 * (halo - 1), -(halo - 1)), slice(None, None)
                if a == 1: return slice(None, None), slice(-2 * halo, -halo)
            if c == 1:
                if a == 0: return slice(-2 * halo, -halo), slice(None, None)
                if a == 1: return slice(None, None), slice(-2 * (halo - 1), -(halo - 1))
            raise ValueError()

    return VectorField2D(data_0=arg_data[0], data_1=arg_data[1])



