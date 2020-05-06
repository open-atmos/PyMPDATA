import numpy as np
import matplotlib.pyplot as plt
from MPyDATA_examples.Olesik_et_al_2020.demo_plot_convergence  import plot
from MPyDATA_examples.utils.show_plot import show_plot

test = np.array([[ 3.20000000e+01,  3.20000000e+01,  3.20000000e+01,
         8.00000000e+01,  8.00000000e+01,  8.00000000e+01,
         1.28000000e+02,  1.28000000e+02,  1.28000000e+02],
       [ 5.00000000e-02,  5.00000000e-01,  9.50000000e-01,
         5.00000000e-02,  5.00000000e-01,  9.50000000e-01,
         5.00000000e-02,  5.00000000e-01,  9.50000000e-01],
       [-5.56969959e+00, -5.89217713e+00, -6.96836274e+00,
        -6.81251853e+00, -7.05469036e+00, -8.89597104e+00,
        -7.54207726e+00, -7.75538222e+00, -9.69176495e+00]])
r0 = test[0]
r1 = test[1]
r2 = test[2]

# r2 = np.random.random(len(r0))

def test_plot(plot = False):
    if plot:
        plot(r0, r1, r2)
        plt.show()




