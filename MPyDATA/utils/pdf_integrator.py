import numpy as np
from scipy import integrate


def discretised_analytical_solution(rh, pdf_t):
    output = np.empty(rh.shape[0]-1)
    for i in range(output.shape[0]):
        dcdf, _ = integrate.quad(pdf_t, rh[i], rh[i+1]) # TODO: handle other output values
        output[i] = dcdf / (rh[i+1] - rh[i])
    return output