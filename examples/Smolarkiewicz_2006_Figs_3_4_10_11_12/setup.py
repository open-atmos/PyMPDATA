import numpy as np

nt = 1600
dt = 1
nx = 500
C = 0.5
x_min = -250
x_max = 250


def cdf_cosine(x):
    x_mid = -150
    f = 2/12
    amplitude = 2

    pdf = np.where(
        np.abs(x-x_mid) < 10,
        amplitude * np.cos(f*(x-x_mid)),
        0)
    return np.cumsum(pdf)


def cdf_rect(x):
    x_mid = -150
    amplitude = 2
    offset = 2

    pdf = offset + np.where(
        np.abs(x-x_mid) <= 10,
        amplitude,
        0)
    return np.cumsum(pdf)

