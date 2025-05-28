Installation
============

Requirements
------------

* Python 3.9+
* Qiskit 2.0+
* Qiskit IBM Runtime 0.28.0+
* IBM Quantum account (for real hardware)

Install from PyPI
-----------------

.. code-block:: bash

    pip install gadd

Install from Source
-------------------

.. code-block:: bash

    git clone https://github.com/qiskit-community/gadd.git
    cd gadd
    pip install -e .

Development Installation
------------------------

.. code-block:: bash

    git clone https://github.com/qiskit-community/gadd.git
    cd gadd
    pip install -e ".[dev]"

Verification
------------

Test your installation:

.. code-block:: python

    import gadd
    print(gadd.__version__)
