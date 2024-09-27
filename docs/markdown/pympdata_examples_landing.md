# Introduction
PyMPDATA examples are bundled with PyMPDATA and located in the examples subfolder.
They constitute a separate PyMPDATA_examples Python package which is also available at PyPI.
The examples have additional dependencies listed in PyMPDATA_examples package setup.py file.
Running the examples requires the PyMPDATA_examples package to be installed.

Below is an example of how to use the PyMPDATA_examples package to run a simple advection-diffusion in 2D
`PyMPDATA_examples.advection_diffusion_2d`
![adv_diff](https://github.com/open-atmos/PyMPDATA/releases/download/tip/advection_diffusion.gif)

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
