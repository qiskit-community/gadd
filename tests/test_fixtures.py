"""
Mock backend and sampler for testing.
"""

from unittest.mock import Mock

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler import Target
from qiskit.circuit.library import CXGate, XGate, SXGate, RZGate
from qiskit.transpiler import InstructionProperties


class MockBackend:
    """Mock backend for testing compatible with Qiskit 2.0."""

    def __init__(self, num_qubits=5):
        self.num_qubits = num_qubits
        self.coupling_map = self._create_coupling_map()
        self.name = "mock_backend"
        self.version = "1.0.0"

    def _create_coupling_map(self):
        """Create a simple linear coupling map."""

        edges = [[i, i + 1] for i in range(self.num_qubits - 1)]
        return CouplingMap(edges)

    @property
    def target(self):
        """Mock target for Qiskit 2.0 compatibility."""

        target = Target()

        # Add single qubit gates
        for i in range(self.num_qubits):
            target.add_instruction(
                XGate(), {(i,): InstructionProperties(duration=35.5e-9)}
            )
            target.add_instruction(
                SXGate(), {(i,): InstructionProperties(duration=35.5e-9)}
            )
            target.add_instruction(RZGate(0), {(i,): InstructionProperties(duration=0)})

        # Add two qubit gates based on coupling map
        for edge in self.coupling_map.get_edges():
            target.add_instruction(
                CXGate(), {edge: InstructionProperties(duration=519e-9)}
            )
            target.add_instruction(
                CXGate(), {edge[::-1]: InstructionProperties(duration=519e-9)}
            )

        return target


class MockSampler:
    """Mock sampler for testing compatible with Qiskit Runtime."""

    def __init__(self, counts_function=None):
        self.counts_function = counts_function or self._default_counts
        self.jobs_run = []
        self._options = {"shots": 4000}

    def _default_counts(self, circuit, shots):
        """Default counts function returns mostly zeros."""
        zero_state = "0" * circuit.num_qubits
        one_state = "1" * circuit.num_qubits
        # Return quasi-distribution format
        return {
            int(zero_state, 2): 0.75,  # 75% probability
            int(one_state, 2): 0.25,  # 25% probability
        }

    def set_options(self, **options):
        """Set options for the sampler."""
        self._options.update(options)

    def run(self, circuits, shots=None):
        """Mock run method compatible with Qiskit Runtime Sampler."""
        if shots is None:
            shots = self._options.get("shots", 4000)

        # Handle single circuit or list
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]

        job = Mock()
        result = Mock()

        # Generate counts for each circuit
        quasi_dists = []
        for circuit in circuits:
            counts = self.counts_function(circuit, shots)
            quasi_dists.append(counts)

        # Mock result methods
        result.quasi_dists = quasi_dists
        result.metadata = [{"shots": shots} for _ in circuits]

        job.result = lambda: result
        job.job_id = lambda: f"mock_job_{len(self.jobs_run)}"

        self.jobs_run.append((circuits, shots))

        return job
