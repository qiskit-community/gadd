"""
Mock fixtures for testing.
"""

from unittest.mock import Mock

from qiskit import QuantumCircuit
from qiskit.transpiler import InstructionDurations
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke


class MockBackend(FakeSherbrooke):
    """Mock backend for testing compatible with Qiskit 2.0."""

    def __init__(self):
        super().__init__()

        # Ensure we have instruction durations for common gates
        self.target._instruction_durations = self._get_instruction_durations()

    def _get_instruction_durations(self) -> InstructionDurations:
        """Get instruction durations including DD gates."""
        durations = []

        # Basic gate durations in dt units
        for qubit in range(self.num_qubits):
            durations.extend(
                [
                    ("id", qubit, 0),
                    ("x", qubit, 160),
                    ("y", qubit, 160),
                    ("z", qubit, 0),
                    ("h", qubit, 160),
                    ("sx", qubit, 160),
                    ("rz", qubit, 0),
                    ("rx", qubit, 160),
                    ("ry", qubit, 160),
                    ("u1", qubit, 0),
                    ("u2", qubit, 160),
                    ("u3", qubit, 160),
                    ("measure", qubit, 4000),
                    ("reset", qubit, 1000),
                ]
            )

        # Two-qubit gate durations
        if hasattr(self, "coupling_map") and self.coupling_map:
            for edge in self.coupling_map.get_edges():
                durations.extend(
                    [
                        ("cx", edge, 800),
                        ("cx", edge[::-1], 800),
                        ("cz", edge, 800),
                        ("cz", edge[::-1], 800),
                        ("ecr", edge, 800),
                        ("ecr", edge[::-1], 800),
                    ]
                )
        else:
            # Two-qubit gate durations - add for ALL possible pairs in both directions
            for i in range(self.num_qubits):
                for j in range(self.num_qubits):
                    if i != j:
                        durations.extend(
                            [
                                ("cx", [i, j], 800),
                                ("cz", [i, j], 800),
                                ("ecr", [i, j], 800),
                            ]
                        )
        return InstructionDurations(durations)


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
