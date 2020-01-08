import numpy as np


class Rogers_drdt:
    def __init__(self, ksi_1, S):
        # Rogers and Yau p. 104
        self.ksi = (S - 1) * ksi_1

    def __call__(self, r):
        return self.ksi / r


class Rogers_pdf:
    def __init__(self, pdf, drdt: Rogers_drdt, t):
        self.t = t
        self.pdf = pdf
        self.drdt = drdt

    def __call__(self, r):
        with np.errstate(invalid='ignore'):
            arg = np.sqrt(r ** 2 - 2 * self.drdt.ksi * self.t)
        return r / arg * self.pdf(arg)