fourier_radiator
================

📡 Fast Fourier-Based Radiation Calculation for Classical Electrons

**`fourier_radiator`** is a high-performance toolkit for computing classical radiation spectra from electron trajectories using Fourier methods. It supports far-field and near-field configurations, GPU acceleration (via OpenCL), and high-resolution spectrum analysis.

.. image:: https://img.shields.io/pypi/v/fourier_radiator.svg
    :target: https://pypi.python.org/pypi/fourier_radiator
    :alt: PyPI Version

.. image:: https://img.shields.io/travis/zhazhajust/fourier_radiator.svg
    :target: https://travis-ci.com/zhazhajust/fourier_radiator
    :alt: Build Status

.. image:: https://readthedocs.org/projects/fourier_radiator/badge/?version=latest
    :target: https://fourier_radiator.readthedocs.io/en/latest/?version=latest
    :alt: Documentation Status

Documentation: https://fourier_radiator.readthedocs.io

---

Features
--------

✅ **Fourier transform–based radiation computation**  
✅ **Far-field and near-field support**  
✅ **GPU-accelerated with PyOpenCL**  
✅ **Customizable frequency & angular grids**  
✅ **Logarithmic or linear spectral resolution**  
✅ **Compatible with real and simulated particle tracks**  
✅ **MPI-ready parallelization support**

---

Installation
------------

You can install with pip (after building locally or publishing):

.. code-block:: bash

    pip install fourier_radiator

Dependencies:

- numpy
- scipy
- pyopencl
- hickle
- mako
- tqdm
- mpi4py (optional)

---

Usage Example
-------------

.. code-block:: python

    from fourier_radiator import FourierRadiator

    config = {
        "grid": [
            (1e-3, 1.0),     # Frequency normalized to ω₁
            (0, 0.1),        # Theta
            (0, 2*np.pi),    # Phi
            (512, 32, 32),   # Grid size
        ],
        "mode": "far",
        "dtype": "float",
    }

    radiator = FourierRadiator(config)
    radiator.calculate_spectrum(particleTracks, timeStep=dt, nSnaps=1)
    spectrum = radiator.Data["radiation"]["total"]

---

Documentation
-------------

Full documentation is available at:  
https://fourier_radiator.readthedocs.io

---

Credits
-------

Created by `zhazhajust`, based on the `cookiecutter` template by Audrey Roy Greenfeld.

Project template:  
https://github.com/audreyr/cookiecutter-pypackage

---

License
-------

This project is licensed under the **GNU General Public License v3 (GPLv3)**.
