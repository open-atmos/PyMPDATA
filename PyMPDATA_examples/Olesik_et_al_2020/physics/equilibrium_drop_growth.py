import numpy as np


class DrDt:
    """ eq. 7.20 in Rogers and Yau 1989 """
    def __init__(self, ksi_1, S):
        self.ksi = (S - 1) * ksi_1

    def __call__(self, r):
        return self.ksi / r

    def mean(self, r1, r2):
        return self.ksi * np.log(r2/r1) / (r2 - r1)


class PdfEvolver:
    """ eq. 7.32 in Rogers and Yau 1989 """
    def __init__(self, pdf, drdt: DrDt, t):
        self.t = t
        self.pdf = pdf
        self.drdt = drdt

    def __call__(self, r):
        with np.errstate(invalid='ignore'):
            arg = np.sqrt(r ** 2 - 2 * self.drdt.ksi * self.t)
        result = r / arg * self.pdf(arg)

        if isinstance(result.magnitude, np.ndarray):
            result = np.where(np.isfinite(result.magnitude), result.magnitude, 0) * result.units
        else:
            if not np.isfinite(result):
                result = 0 * result.units

        return result
