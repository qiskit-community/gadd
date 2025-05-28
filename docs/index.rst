GADD: Genetic Algorithm for Dynamical Decoupling
================================================

**GADD** is a Python package for empirically optimizing dynamical decoupling (DD) sequences on quantum processors using genetic algorithms. Based on the research paper `"Empirical learning of dynamical decoupling on quantum processors" <https://arxiv.org/abs/2403.02294>`_.

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.9+

.. image:: https://img.shields.io/badge/Qiskit-2.0+-blue.svg
   :target: https://qiskit.org/
   :alt: Qiskit

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
   :alt: License

Quick Start
-----------

Install GADD:

.. code-block:: bash

    pip install gadd

Basic usage:

.. code-block:: python

    from gadd import GADD, TrainingConfig
    from qiskit import QuantumCircuit
    from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

    # Create quantum circuit
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)

    # Setup GADD
    service = QiskitRuntimeService()
    backend = service.least_busy()
    config = TrainingConfig(pop_size=16, n_iterations=10)
    gadd = GADD(backend=backend, config=config)

    # Train DD sequences
    def utility_function(circuit, result):
        counts = result.get_counts()
        return counts.get('000', 0) / sum(counts.values())

    best_strategy, result = gadd.train(
        sampler=Sampler(),
        training_circuit=circuit,
        utility_function=utility_function
    )

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/gadd
   api/sequences
   api/utility_functions
   api/group_operations
   api/circuit_padding

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Research Paper
--------------

If you use GADD in your research, please cite:

.. code-block:: bibtex

    @article{tong2024empirical,
      title={Empirical learning of dynamical decoupling on quantum processors},
      author={Tong, Christopher and Zhang, Helena and Pokharel, Bibek},
      journal={arXiv preprint arXiv:2403.02294},
      year={2024}
    }
