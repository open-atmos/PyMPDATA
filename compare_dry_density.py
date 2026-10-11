# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
from PyMPDATA_examples.Shipway_and_Hill_2012 import si, settings

# %%
RHOD_VERTVELO = 3*si.m/si.s * si.kg/si.m**3
T_MAX = 15*si.minutes
P0 = 1000*si.hPa
Z_MAX = 3200*si.m
nr = 32
dt = 1.25*si.s
dz = 25*si.m
R_MIN = 1 * si.um
R_MAX = 20.2 * si.um
apprx_drhod_dz = False

settings_true = settings.Settings(
                    rhod_w_const=RHOD_VERTVELO, nr=nr, dt=dt, dz=dz, t_max=T_MAX, 
                    r_min=R_MIN, r_max=R_MAX, p0=P0, z_max=Z_MAX, apprx_drhod_dz=apprx_drhod_dz
                )
settings_false = settings.Settings(
                    rhod_w_const=RHOD_VERTVELO, nr=nr, dt=dt, dz=dz, t_max=T_MAX, 
                    r_min=R_MIN, r_max=R_MAX, p0=P0, z_max=Z_MAX, apprx_drhod_dz=apprx_drhod_dz, use_max_step=False,
                )
# %%
z = z_points = np.linspace(0.0, 2 * settings_true.nz + 1)
plt.plot(settings_false.rhod(z) - settings_true.rhod(z), z)
plt.legend()
plt.ylabel("z / m")
plt.xlabel("difference in dry density compared to using max_step / kg/m$^{-3}$")
plt.show()
# %%
