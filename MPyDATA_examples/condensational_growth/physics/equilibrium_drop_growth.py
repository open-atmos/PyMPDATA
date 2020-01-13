import numpy as np


class DrDt:
    """ eq. 7.20 in Rogers and Yau 1989 """
    def __init__(self, ksi_1, S):
        self.ksi = (S - 1) * ksi_1

    def __call__(self, r):
        return self.ksi / r


class PdfEvolver:
    """ eq. 7.32 in Rogers and Yau 1989 """
    def __init__(self, pdf, drdt: DrDt, t):
        self.t = t
        self.pdf = pdf
        self.drdt = drdt

    def __call__(self, r):
        with np.errstate(invalid='ignore'):
            arg = np.sqrt(r ** 2 - 2 * self.drdt.ksi * self.t)
        return r / arg * self.pdf(arg)
