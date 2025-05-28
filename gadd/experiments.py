"""
Pre-configured experiments from the GADD paper.
"""

from typing import Optional, Dict, Any
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import Sampler

from .gadd import GADD
from .utility_functions import UtilityFactory


def create_bernstein_vazirani_circuit(bitstring: str) -> QuantumCircuit:
    """
    Create Bernstein-Vazirani circuit for given bitstring.

    Args:
        bitstring: Hidden bitstring to encode.

    Returns:
        Quantum circuit implementing BV algorithm.
    """
    n = len(bitstring)
    circuit = QuantumCircuit(n + 1, n)

    # Initialize ancilla in |->
    circuit.x(n)
    circuit.h(n)

    # Apply Hadamard to all qubits
    for i in range(n):
        circuit.h(i)

    # Oracle
    for i, bit in enumerate(bitstring):
        if bit == "1":
            circuit.cx(i, n)

    # Final Hadamards
    for i in range(n):
        circuit.h(i)

    # Measure
    for i in range(n):
        circuit.measure(i, i)

    return circuit


def create_ghz_circuit(n_qubits: int) -> QuantumCircuit:
    """
    Create GHZ state preparation circuit.

    Args:
        n_qubits: Number of qubits.

    Returns:
        Circuit preparing GHZ state.
    """
    circuit = QuantumCircuit(n_qubits, n_qubits)

    circuit.h(0)
    for i in range(n_qubits - 1):
        circuit.cx(i, i + 1)

    circuit.measure_all()
    return circuit


def run_bv_experiment(
    gadd: GADD, sampler: Sampler, n: int = 24, bitstring: Optional[str] = None, **kwargs
) -> Dict[str, Any]:
    """
    Run Bernstein-Vazirani experiment as in the paper.

    Args:
        gadd: GADD instance.
        sampler: Qiskit Runtime sampler.
        n: Problem size.
        bitstring: Hidden bitstring (defaults to all-ones).
        **kwargs: Additional arguments for training.

    Returns:
        Dictionary with experiment results.
    """
    if bitstring is None:
        bitstring = "1" * n

    # Create circuit
    circuit = create_bernstein_vazirani_circuit(bitstring)

    # Create utility function
    utility_function = UtilityFactory.success_probability(bitstring)

    # Standard sequences to compare
    comparison_seqs = kwargs.pop(
        "comparison_seqs",
        [
            "cpmg",
            "cpmg_staggered",
            "xy4",
            "xy4_staggered",
            "edd",
            "edd_staggered",
            "urdd",
        ],
    )

    # Train
    best_strategy, result = gadd.train(
        sampler=sampler,
        training_circuit=circuit,
        utility_function=utility_function,
        comparison_seqs=comparison_seqs,
        **kwargs,
    )

    return {
        "best_strategy": best_strategy,
        "training_result": result,
        "circuit": circuit,
        "utility_function": utility_function,
    }


def run_ghz_experiment(
    gadd: GADD, sampler: Sampler, n_qubits: int = 50, **kwargs
) -> Dict[str, Any]:
    """
    Run GHZ state preparation experiment as in the paper.

    Args:
        gadd: GADD instance.
        sampler: Qiskit Runtime sampler.
        n_qubits: Number of qubits for GHZ state.
        **kwargs: Additional arguments for training.

    Returns:
        Dictionary with experiment results.
    """
    # Create circuit
    circuit = create_ghz_circuit(n_qubits)

    # Create utility function
    utility_function = UtilityFactory.ghz_state(n_qubits)

    # Standard sequences to compare
    comparison_seqs = kwargs.pop(
        "comparison_seqs",
        ["cpmg", "cpmg_staggered", "xy4", "xy4_staggered", "edd", "edd_staggered"],
    )

    # Train
    best_strategy, result = gadd.train(
        sampler=sampler,
        training_circuit=circuit,
        utility_function=utility_function,
        comparison_seqs=comparison_seqs,
        **kwargs,
    )

    return {
        "best_strategy": best_strategy,
        "training_result": result,
        "circuit": circuit,
        "utility_function": utility_function,
    }
