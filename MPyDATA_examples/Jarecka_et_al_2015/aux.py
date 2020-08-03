from numpy import gradient
import numpy as np


def extrapolate_in_time(u_old, u_new):
    """
    extrapolation to value^{t+0.5}
    """
    u_mid = (3. * u_new - u_old) * .5
    return u_mid

def grad(mass, dx, dy=None): 
    """
    first arg correspond to dx instead of dy.
    """
    if dy:
        grad_y = np.array(gradient(mass, axis=0)) / dy
        grad_x = np.array(gradient(mass, axis=1)) / dx 
        return grad_y, grad_x
    else:
        return gradient(mass) / dx

def interpolate_in_space(uh, h, axis=None): 
    u = None
    result = np.array(uh)

    zero_h = h==0.
    non_zero_h = h!=0.
    result[non_zero_h] /= h[non_zero_h]
    result[zero_h] = 0.

    if axis=='x':
        ny, nx = uh.shape
        u = np.zeros((ny, nx + 1))
        for i in range(ny):
            for j in range(1, nx):
                u[i][j] = (result[i][j-1] + result[i][j]) / 2.
        return u

    elif axis=='y':
        ny, nx = uh.shape
        u = np.zeros((ny + 1, nx))
        for j in range(nx):
            for i in range(1, ny):
                u[i][j] = (result[i-1][j] + result[i][j]) / 2.
        return u

    else:
        u = np.zeros((uh.shape[0] + 1))
        for i in range(1, u.shape[0] - 1):
            u[i] = (result[i-1] + result[i]) / 2.

        return u



coriolis_p = 2. * np.pi / 86164. # * sine(90)
