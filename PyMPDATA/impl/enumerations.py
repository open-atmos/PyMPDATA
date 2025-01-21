""" common constants named with the intention of improving code readibility
(mostly integer indices used for indexing tuples) """

import numpy as np

ARG_FOCUS, ARG_DATA, ARG_DATA_OUTER, ARG_DATA_MID3D, ARG_DATA_INNER = 0, 1, 1, 2, 3
""" indices within tuple passed in fill_halos boundary-condition calls """

MAX_DIM_NUM = 3
""" maximal number of dimensions supported by the package """

OUTER, MID3D, INNER = 0, 1, -1
""" labels for identifying 1st, 2nd, and 3rd dimensions """

IMPL_META_AND_DATA, IMPL_BC = 0, 1
""" indices of "meta and data" and "bc" elements of the impl tuple in Field instances """

META_AND_DATA_META, META_AND_DATA_DATA = 0, 1
""" indices of "meta" and "data" elements of the "meta and data" impl Field property """

SIGN_LEFT, SIGN_RIGHT = +1, -1
""" left-hand and right-hand domain sides as used in boundary conditions logic """

RNG_START, RNG_STOP, RNG_STEP = 0, 1, 2
""" indices of elements in range-expressing tuples """

INVALID_INDEX = -44
""" value with which never-to-be-used unused-dimension tuples are filled with in traversals """

INVALID_INIT_VALUE, BUFFER_DEFAULT_VALUE = np.nan, np.nan
""" values with which arrays are filled at initialisation """

INVALID_HALO_VALUE = 666
""" value used when constructing never-to-be-used instances of Constant boundary condition """

INVALID_NULL_VALUE = 0.0
""" value with which the never-to-be-used "null" fields are populated """

ONE_FOR_STAGGERED_GRID = 1
""" used for explaining the purpose of +1 index addition if related to Arakawa-C grid shift """
