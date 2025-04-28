# Introduction
PyMPDATA examples are bundled with PyMPDATA and located in the examples subfolder.
They constitute a separate PyMPDATA_examples Python package which is also available at PyPI.
The examples have additional dependencies listed in PyMPDATA_examples package setup.py file.
Running the examples requires the PyMPDATA_examples package to be installed.

We recommend you look through the example gallery below to see the examples in action.

# Example gallery
Unless stated otherwise the following examples solve the <mark>basic advection equation</mark>:
$$ \partial_t (\psi) + \nabla \cdot (u \psi) = 0 $$

The examples are grouped by the dimensionality of the computational grid.

## in 1D
| tags                                                                                                                                                                                              | link                                                     |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------|
| <mark>advection-diffusion equation</mark><br/>$$ \partial_t (\psi) + \nabla \cdot (u \psi) + \mu \Delta (\psi) = 0 $$                                                                             | `PyMPDATA_examples.advection_diffusion_1d`*              |
| <mark>Black-Scholes equation</mark>, option pricing<br>$$  \frac{\partial f}{\partial t} + rS \frac{\partial f}{\partial S} + \frac{\sigma^2}{2} S^2 \frac{\partial^2 f}{\partial S^2} - rf = 0$$ | `PyMPDATA_examples.Arabas_and_Farhat_2020`*              |
| <mark>advection equation</mark>, homogeneous, several algorithm variants comparison: infinite-gauge, flux-corrected,..                                                                            | `PyMPDATA_examples.Smolarkiewicz_2006_Figs_3_4_10_11_12` |
| <mark>Size-spectral advection</mark>, particle population condensational growth, coordinate transformation<br>$$ \partial_t (G \psi) + \nabla \cdot (Gu \psi) = 0 $$                              | `PyMPDATA_examples.Olesik_et_al_2022`*                   |
| <mark>advection equation</mark>, [double-pass donor-cell option](https://osti.gov/biblio/7049237)                                                                                                                       | `PyMPDATA_examples.DPDC`                                 |

## in 2D
| tags                                                                                                                                                                                                                                                                                                    | link                                                                                                                                                                                                                                                                                                  |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <mark>Boussinesq system</mark> for buoyancy-driven flow<br/>$$\begin{eqnarray}\partial_t \vec{v}+\nabla\cdot\left(\vec{v}\otimes\vec{v}\right)=-\nabla\pi-\vec{g}\frac{\theta^\prime}{\theta_0}~\\\ \partial_t\theta+\nabla\cdot\left(\vec{v}\theta\right)=0~\\\ \nabla\cdot\\vec{v}=0\end{eqnarray}$$  | `PyMPDATA_examples.Jaruga_et_al_2015`<br/><img src="https://github.com/open-atmos/PyMPDATA/releases/download/tip/boussinesq_2d_anim.gif" width="50%" alt="boussinesq-2d">                                                                                                                             |
| <mark>advection-diffusion equation</mark><br/>$$ \partial_t (\psi) + \nabla \cdot (u \psi) + \mu \Delta (\psi) = 0 $$                                                                                                                                                                                   | `PyMPDATA_examples.advection_diffusion_2d`*<br/><img src="https://github.com/open-atmos/PyMPDATA/releases/download/tip/advection_diffusion.gif" width="50%" alt="adv-diff">                                                                                                                           |
| <mark>Spectral-spatial advection</mark>, particle population condensational growth in a vertical column of air, time dependent flow                                                                                                                                                                     | `PyMPDATA_examples.Shipway_and_Hill_2012`<br/><img src="https://github.com/open-atmos/PyMPDATA/wiki/files/KiD-1D_PyMPDATA_n_iters=3.gif" width="50%" alt="spectral-spatial">                                                                                                                          |
| <mark>shallow-water equations</mark><br/>$$\begin{eqnarray} \partial_t h + \partial_x (uh) + \partial_y (vh) &=& 0~  \\\ \partial_t (uh) + \partial_x ( uuh) + \partial_y (vuh) &=& - h \partial_x h~ \\\ \partial_t (vh) + \partial_x ( uvh) + \partial_y (vvh) &=& - h \partial_y h~ \end{eqnarray}$$ | `PyMPDATA_examples.Jarecka_et_al_2015`*                                                                                                                                                                                                                                                               |
| <mark>advection equation</mark>, solid body rotation                                                                                                                                                                                                                                                    | `PyMPDATA_examples.Molenkamp_test_as_in_Jaruga_et_al_2015_Fig_12`*                                                                                                                                                                                                                                    |
| <mark>advection equation</mark>, solid body rotation                                                                                                                                                                                                                                                    | `PyMPDATA_examples.wikipedia_example`*<br /><img src="https://github.com/open-atmos/PyMPDATA/releases/download/tip/wikipedia_example.gif" width="50%" alt="Wikipedia example">                                                                                                                        |
| <mark>advection equation</mark>, coordinate transformation, spherical coordinates, see also examples in [PyMPDATA-MPI](https://pypi.org/project/PyMPDATA-MPI/) $$ \partial_t (G \psi) + \nabla \cdot (Gu \psi) = 0 $$                                                                                   | `PyMPDATA_examples.Williamson_and_Rasch_1989_as_in_Jaruga_et_al_2015_Fig_14`<br><img src="https://github.com/open-atmos/PyMPDATA-MPI/releases/download/latest-generated-plots/n_iters.1_rank_0_size_1_c_field_.0.5.0.25._mpi_dim_0_n_threads_1-SphericalScenario-anim.gif" width="50%" alt="mpi-gif"> |
| <mark>advection equation</mark>, comparison against DG solution using [Trixi.jl](https://trixi-framework.github.io/) ([Ranocha et al. 2022](https://doi.org/10.21105/jcon.00077))                                                                                                                       | `PyMPDATA_examples.trixi_comparison`                                                                                                                                                                                                                                                                  |
| <mark>Black-Scholes equation</mark>, option pricing<br>$$  \frac{\partial f}{\partial t} + rS \frac{\partial f}{\partial S} + \frac{\sigma^2}{2} S^2 \frac{\partial^2 f}{\partial S^2} + \frac{dA}{dt}\frac{\partial f}{\partial A} - rf = 0$$                                                          | `PyMPDATA_examples.Magnuszewski_et_al_2025`                                                                                                                                                                                                                                                           |

## in 3D
| tags                                                                                                                                | link                                   |
|:------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------|
| <mark>homogeneous advection equation</mark>                                                                                         | `PyMPDATA_examples.Smolarkiewicz_1984` |
| <mark>homogeneous advection equation</mark>, performance comparison against libmpdata++, scalability analysis in respect to threads | `PyMPDATA_examples.Bartman_et_al_2022` |

\* - with comparison against analytic solution

# Installation
Since the examples package includes Jupyter notebooks (and their execution requires write access), the suggested install and launch steps are:

```
git clone https://github.com/open-atmos/PyMPDATA-examples.git
cd PyMPDATA-examples
pip install -e .
jupyter-notebook
```

Alternatively, one can also install the examples package from pypi.org by using
```
pip install PyMPDATA-examples.
```
