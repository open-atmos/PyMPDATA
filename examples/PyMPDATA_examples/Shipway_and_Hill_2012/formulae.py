from collections import namedtuple
import numpy as np
from scipy import constants


def convert_to(value, unit):
    value /= unit


si = namedtuple(
    'si',
    ('kg', 'm', 's', 'metres', 'second', 'um', 'hPa', 'micrometre',
     'minutes', 'km', 'dimensionless', 'kelvin', 'mg')
)(
    kg=1.,
    m=1.,
    s=1.,
    metres=1.,
    second=1.,
    um=1e-6,
    hPa=100.,
    micrometre=1e-6,
    minutes=60.,
    km=1000.,
    dimensionless=1.,
    kelvin=1.,
    mg=1e-6
)

_Mv = 0.018015
_Md = 0.028970

const = namedtuple(
    'const',
    ('eps', 'g', 'p1000', 'Rd', 'Rv', 'c_pd', 'c_pv', 'lv',
     'rho_l', 'T0', 'ARM_C1', 'ARM_C2', 'ARM_C3')
)(
    eps=_Mv / _Md,
    g=constants.g,
    p1000=1000 * si.hPa,
    Rd=287.0027,
    Rv=constants.R / _Mv,
    c_pd=1005,
    c_pv=1850,
    lv=2.5e6,
    rho_l=1e3 * si.kg / si.m**3,
    T0=constants.zero_Celsius,
    ARM_C1=6.1094 * si.hPa,
    ARM_C2=17.625 * si.dimensionless,
    ARM_C3=243.04 * si.kelvin,
)


def rho_d(p, qv, theta_std):
    return p * (1 - 1 / (1 + const.eps / qv)) / (
                np.power(p / const.p1000, const.Rd / const.c_pd) * const.Rd * theta_std)


def drho_dz(g, p, T, qv, lv, dql_dz=0):
    Rq = const.Rv / (1 / qv + 1) + const.Rd / (1 + qv)
    cp = const.c_pv / (1 / qv + 1) + const.c_pd / (1 + qv)
    rho = p / Rq / T
    return (g / T * rho * (Rq / cp - 1) - p * lv / cp / T ** 2 * dql_dz) / Rq


# A14 in libcloudph++ 1.0 paper
def temperature(rhod, thd):
    return thd * np.power(
        rhod * thd / const.p1000 * const.Rd,
        const.Rd / const.c_pd / (1 - const.Rd / const.c_pd)
    )


# A15 in libcloudph++ 1.0 paper
def pressure(rhod, T, qv):
    return rhod * (1 + qv) * (const.Rv / (1 / qv + 1) + const.Rd / (1 + qv)) * T


def th_dry(th_std, qv):
    return th_std * np.power(1 + qv / const.eps, const.Rd / const.c_pd)


def pvs_Celsius(T):
    return const.ARM_C1 * np.exp((const.ARM_C2 * T) / (T + const.ARM_C3))


def pv(p, qv):
    return p * qv / (qv + const.eps)
