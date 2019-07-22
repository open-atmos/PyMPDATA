"""
Created at 22.07.2019

@author: Michael Olesik
@author: Sylwester Arabas
"""

from coord import x_id, x_ln, x_p2
from mpdata import MPDATA


# RUNNER START
def run(setup, si):
    s = setup

    result = {}

    for fn in (x_id, x_p2, x_ln):

        algos = (
            MPDATA(s.nr, s.r_min, s.r_max, s.dt, s.cdf0, fn(si), {"n_it": 1, "dfl": 0, "iga": 0, "tot": 0, "fct": 0}),
            # TODO
            # MPDATA(nr, r_min, r_max, dt, cdf0, fn(si), {"n_it": 2, "dfl": 0, "iga": 0, "tot": 0, "fct": 0}),
            # MPDATA(nr, r_min, r_max, dt, cdf0, fn(si), {"n_it":3, "dfl":0, "iga":0, "tot":0, "fct":0}),
            # MPDATA(nr, r_min, r_max, dt, cdf0, fn(si), {"n_it":2, "dfl":1, "iga":0, "tot":0, fct":0}),
            # MPDATA(nr, r_min, r_max, dt, cdf0, fn(si), {"n_it":2, "dfl":0, "iga":1, "tot":0, "fct":0}),
            # MPDATA(nr, r_min, r_max, dt, cdf0, fn(si), {"n_it":2, "dfl":1, "iga":1, "tot":1, "fct":0}),
            # MPDATA(nr, r_min, r_max, dt, cdf0, fn(si), {"n_it": 3, "dfl": 1, "iga": 1, "tot": 1, "fct": 0}),
            MPDATA(s.nr, s.r_min, s.r_max, s.dt, s.cdf0, fn(si), {"n_it": 3, "dfl": 1, "iga": 1, "tot": 1, "fct": 1})
        )

        for algo in algos:
            for t in range(s.nt):
                algo.step(s.drdt)

        result[fn] = algos

    return result
# RUNNER END
