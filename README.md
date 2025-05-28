# GADD: Genetic Algorithm for Dynamical Decoupling optimization

<!-- START_DOCS_INCLUDE -->

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit](https://img.shields.io/badge/Qiskit-2.0+-blue.svg)](https://qiskit.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Coverage Status](https://coveralls.io/repos/github/qiskit-community/gadd/badge.svg?branch=main)](https://coveralls.io/github/qiskit-community/gadd?branch=main)

**GADD** is a Python package for empirically optimizing dynamical decoupling (DD) sequences on quantum processors using a genetic algorithm as described in the research paper ["Empirical learning of dynamical decoupling on quantum processors"](https://arxiv.org/abs/2403.02294).

## Key Features

- **Empirical Optimization**: Learn DD sequences directly from quantum hardware feedback
- **Hardware Agnostic**: Works with any quantum backend that supports Qiskit
- **Genetic Algorithm**: Efficient search through large DD sequence spaces
- **Multiple Utility Functions**: Built-in metrics for various quantum applications
- **Comparative Analysis**: Benchmark against standard DD sequences (XY4, CPMG, EDD, URDD)
- **Comprehensive Results**: Detailed training progression and performance analytics

## Installation

### Requirements

This package is designed to be used with [Qiskit](https://github.com/Qiskit/qiskit) and the [Qiskit Runtime IBM Client](https://github.com/Qiskit/qiskit-ibm-runtime). To run on IBM hardware, you will need an IBM Quantum account.

### Install from PyPI

```bash
pip install gadd
```

### Install from Source

```bash
git clone https://github.com/qiskit-community/gadd.git
cd gadd
pip install -e .
```

## Usage Examples

The core class `GADD` runs the genetic algorithm training process
on training circuits, outputting the best sequence and intermediate training data, if desired:

### 1. Basic training

```python
from gadd import GADD, TrainingConfig
from gadd.utility_functions import success_probability_utility
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

service = QiskitRuntimeService()
backend = service.least_busy()

# Create a simple quantum circuit
circuit = QuantumCircuit(3)
circuit.h(0)
circuit.cx(0, 1)
circuit.cx(1, 2)
circuit.measure_all()

# Simple configuration
config = TrainingConfig(pop_size=16, n_iterations=10)
gadd = GADD(backend=backend, config=config)

best_strategy, result = gadd.train(
    sampler=sampler,
    training_circuit=circuit,
    utility_function=success_probability_utility
)
```

### 2. Custom parameters

The package supports customizing training parameters and utility functions. Plotting
the training progression and running the circuit on the target circuit are also supported.

```python
from qiskit.visualization import timeline_drawer

config = TrainingConfig(
    pop_size=32,              # Larger population
    sequence_length=8,        # 8-gate DD sequences
    n_iterations=50,          # More iterations
    mutation_probability=0.8, # Higher mutation rate
    shots=8000,              # More shots per evaluation
    num_colors=3,            # 3-coloring for qubit graph
    dynamic_mutation=True    # Adaptive mutation
)

gadd = GADD(
    backend=backend, 
    config=config,
    seed=42
)

# Train with checkpointing and comparing against canonical sequences
best_strategy, result = gadd.train(
    sampler=sampler,
    training_circuit=circuit,
    utility_function=utility_function,
    save_path="./checkpoints",
    comparison_seqs=['cpmg', 'cpmg_staggered', 'xy4', 'xy4_staggered', 'edd', 'edd_staggered']
)

# Plot results
gadd.plot_training_progress(result, save_path="training_plot.png")

# Visualize the target circuit with DD sequences
circuit_with_dd = gadd.apply_dd(
    strategy = best_strategy,
    target_circuit = target_circuit,
    backend = backend
)
timeline_drawer(circuit_with_dd)

# Run the best sequence on the target circuit
result = gadd.run(
    strategy = best_strategy,
    target_circuit=target_circuit,
    sampler=sampler
)
```

### 3. Save and resume training

```python
# Resume from previous training
previous_state = gadd.load_training_state("./checkpoints/checkpoint_iter_20.json")

best_strategy, result = gadd.train(
    sampler=sampler,
    training_circuit=circuit,
    utility_function=utility_function,
    resume_from_state=previous_state
)
```

### 4. Custom utility functions

```python
def custom_fidelity(circuit, result):
    """Custom fidelity-based utility function."""
    counts = result.get_counts()
    total_shots = sum(counts.values())
    
    # Expected distribution for your specific circuit
    expected_dist = {'000': 0.5, '111': 0.5}  # GHZ state
    
    # Calculate 1-norm distance
    observed_dist = {state: count/total_shots for state, count in counts.items()}
    
    fidelity = 1 - 0.5 * sum(abs(expected_dist.get(state, 0) - observed_dist.get(state, 0)) 
                             for state in set(expected_dist.keys()) | set(observed_dist.keys()))
    
    return fidelity

# Use custom utility function
best_strategy, result = gadd.train(
    sampler=sampler,
    training_circuit=circuit,
    utility_function=UtilityFactory.custom(custom_fidelity, "Custom Fidelity")
)
```

## Directory structure

```text
gadd/
├── README.md                      # This file
├── pyproject.toml                 # Package configuration
├── LICENSE                        # Apache 2.0 license
├── CITATION.bib                   # BibTeX citation file
│
├── gadd/                          # Package directory
│   ├── __init__.py                # Package initialization
│   ├── gadd.py                    # Main GADD algorithm implementation
│   ├── experiments.py             # GADD experiments from the paper
│   ├── sequences.py               # DD sequence definitions
│   ├── utility_functions.py       # Utility function implementations
│   ├── group_operations.py        # Group theory operations
│   └── circuit_padding.py         # Circuit padding utilities
|
└── tests/                         # Test suite
    ├── __init__.py                # Test initialization
    ├── test_gadd.py               # Core algorithm tests
    ├── test_sequences.py          # DD sequence tests
    ├── test_utility_functions.py  # Utility function tests
    ├── test_group_operations.py   # Group operation tests
    └── test_circuit_padding.py    # DD sequence tests

```

<!-- END_DOCS_INCLUDE -->

## Citation

If you use GADD in your work, please cite as per the included [BibTeX file](CITATION.bib).
