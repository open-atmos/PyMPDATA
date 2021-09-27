import numpy as np
from scipy import integrate


def from_pdf_2d(pdf: callable, xrange: list, yrange: list, gridsize: list):
    z = np.empty(gridsize)
    dx, dy = (xrange[1] - xrange[0]) / gridsize[0], (yrange[1] - yrange[0]) / gridsize[1]
    for i in range(gridsize[0]):
        for j in range(gridsize[1]):
            z[i, j] = integrate.nquad(pdf, ranges=(
                (xrange[0] + dx*i, xrange[0] + dx*(i+1)),
                (yrange[0] + dy*j, yrange[0] + dy*(j+1))
            ))[0] / dx / dy
    x = np.linspace(xrange[0] + dx / 2, xrange[1] - dx / 2, gridsize[0])
    y = np.linspace(yrange[0] + dy / 2, yrange[1] - dy / 2, gridsize[1])
    return x, y, z


def from_cdf_1d(cdf: callable, x_min: float, x_max: float, nx: int):
    dx = (x_max - x_min) / nx
    x = np.linspace(x_min + dx / 2, x_max - dx / 2, nx)
    xh = np.linspace(x_min, x_max, nx + 1)
    y = np.diff(cdf(xh)) / dx
    return x, y


def discretised_analytical_solution(rh, pdf_t, midpoint_value=False, r=None):
    if midpoint_value:
        assert r is not None
    else:
        assert r is None
    output = np.empty(rh.shape[0]-1)
    for i in range(output.shape[0]):
        if midpoint_value:
            output[i] = pdf_t(r[i])
        else:
            dcdf, _ = integrate.quad(pdf_t, rh[i], rh[i + 1])  # TODO #206: handle other output values
            output[i] = dcdf / (rh[i + 1] - rh[i])
    return output

