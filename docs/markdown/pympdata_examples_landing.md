# Introduction
PyMPDATA examples are bundled with PyMPDATA and located in the examples subfolder.
They constitute a separate PyMPDATA_examples Python package which is also available at PyPI.
The examples have additional dependencies listed in PyMPDATA_examples package setup.py file.
Running the examples requires the PyMPDATA_examples package to be installed.

Below is an example of how to use the PyMPDATA_examples package to run a simple advection-diffusion in 2D
`PyMPDATA_examples.advection_diffusion_2d`
![adv_diff](https://github.com/open-atmos/PyMPDATA/releases/download/tip/advection_diffusion.gif)

# Example gallery

## in 1D
| tags                                                                                                              | link                                                     |
|:------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------|
| <mark>advection-diffusion</mark><br/>$$ \partial_t (G \psi) + \nabla \cdot (Gu \psi) + \mu \Delta (G \psi) = 0 $$ |`PyMPDATA_examples.advection_diffusion_1d` * |
| <mark>Black-Scholes</mark>, option pricing                                                                        | `PyMPDATA_examples.Arabas_and_Farhat_2020`               |
| particle population condensational growth                                                                         | `PyMPDATA_examples.Olesik_et_al_2022`                    |
| <mark>advection</mark>, homogeneous, infinite-gauge, flux-corrected                                               | `PyMPDATA_examples.Smolarkiewicz_2006_Figs_3_4_10_11_12` |
## in 2D
| tags                 | link                                                                                                                                                                       |
|:---------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| advection-diffusion  | `PyMPDATA_examples.advection_diffusion_2d`<br/><img src="https://github.com/open-atmos/PyMPDATA/releases/download/tip/advection_diffusion.gif" width="50%" alt="adv-diff"> |
| droplet condensation | `PyMPDATA_examples.Shipway_and_Hill_2012`                                                                                                                                  |

## in 3D
| tags               | link                                       |
|:-------------------|:-------------------------------------------|
| advection          | `PyMPDATA_examples.Smolarkiewicz_1984` |
\* - with analytic solution

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
