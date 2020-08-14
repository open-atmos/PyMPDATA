import numpy as np


def L2(numerical, analytical, nt): # TODO: change name
    assert numerical.shape == analytical.shape
    N = analytical.size
    err2 = np.log(
        np.sqrt(
            sum(pow(numerical - analytical, 2)) / nt / N
        )
    ) / np.log(2)
    return err2

def Smolarkiewicz_Grabowski_1990_eq21(numerical, analytical, T):
    assert numerical.shape == analytical.shape
    NX = analytical.size
    err = np.sqrt(
            sum(pow(numerical - analytical, 2)) / NX
        ) / T
    return err

def Smolarkiewicz_Rasch_r0(numerical, analytical, g_factor):
    err = np.sqrt(sum(pow(numerical - analytical, 2)) * g_factor) / np.max(analytical)
    return err

def Smolarkiewicz_Rasch_r1(numerical, analytical, g_factor):
    err = sum(numerical * g_factor) / sum(analytical * g_factor) - 1
    return err

def Smolarkiewicz_Rasch_r2(numerical, analytical, g_factor):
    err = sum(numerical**2 * g_factor) / sum(analytical**2 * g_factor) - 1
    return err

