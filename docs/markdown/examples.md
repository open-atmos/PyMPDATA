## Examples (Jupyter notebooks reproducing results from literature):

PyMPDATA examples are bundled with PyMPDATA and located in the `examples` subfolder.
They constitute a separate [``PyMPDATA_examples``](https://pypi.org/p/PyMPDATA-examples) Python package which is also available at PyPI.
The examples have additional dependencies listed in [``PyMPDATA_examples`` package ``setup.py``](https://github.com/open-atmos/PyMPDATA/blob/main/examples/setup.py) file.
Running the examples requires the ``PyMPDATA_examples`` package to be installed.
Since the examples package includes Jupyter notebooks (and their execution requires write access), the suggested install and launch steps are:
```
git clone https://github.com/open-atmos/PyMPDATA-examples.git
cd PyMPDATA-examples
pip install -e .
jupyter-notebook
```
Alternatively, one can also install the examples package from pypi.org by using ``pip install PyMPDATA-examples``.
