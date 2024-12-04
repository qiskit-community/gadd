from dataclasses import dataclass
from typing import List, Dict, Optional
from qiskit.circuit.library import IGate, XGate, YGate, U1Gate, RZGate, RXGate, RYGate
from qiskit import QuantumCircuit, pulse
import numpy as np

# Default gateset for DD pulses
xbcircuit = QuantumCircuit(1, name="xb")
xbcircuit.append(U1Gate(np.pi), [0])
xbcircuit.x(0)
xbcircuit.append(U1Gate(np.pi), [0])
XbGate = xbcircuit.to_instruction()

ybcircuit = QuantumCircuit(1, name="yb")
ybcircuit.append(U1Gate(np.pi), [0])
ybcircuit.y(0)
ybcircuit.append(U1Gate(np.pi), [0])
YbGate = ybcircuit.to_instruction()

default_gateset = {
    0: IGate(),
    1: IGate(),
    2: XGate(),
    3: XbGate,
    4: YGate(),
    5: YbGate,
    6: RZGate(np.pi),
    7: RZGate(-np.pi),
}


@dataclass
class DDSequence:
    """Dynamical Decoupling sequence representation."""

    gates: List[str]

    def __len__(self):
        return len(self.gates)

    def __getitem__(self, idx):
        return self.gates[idx]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DDSequence):
            return NotImplemented
        return self.gates == other.gates

    def copy(self) -> "DDSequence":
        """Create a deep copy of the sequence."""
        return DDSequence(self.gates.copy())


class StandardSequences:
    """Standard DD sequence implementations."""

    def __init__(self):
        self._sequences = {
            "xy4": DDSequence(["X", "Y", "X", "Y"]),
            "cpmg": DDSequence(["X", "X"]),
            "edd": DDSequence(["X", "Y", "X", "Y", "Y", "X", "Y", "X"]),
            "baseline": DDSequence(["I"]),
        }

    def get(self, name: str) -> DDSequence:
        """Get a standard sequence by name."""
        return self._sequences.get(name)

    def list_available(self) -> List[str]:
        """List available standard sequences."""
        return list(self._sequences.keys())


class DDStrategy:
    """Collection of DD sequences assigned to different colors."""

    def __init__(self, sequences: Dict[int, DDSequence]):
        """
        Initialize DD strategy with sequences for each color.

        Args:
            sequences: Dictionary mapping color indices to DD sequences

        Raises:
            ValueError: If sequences dict is empty
        """
        if not sequences:
            raise ValueError("Must provide at least one sequence")
        self.sequences = sequences

    @classmethod
    def from_single_sequence(cls, sequence: DDSequence, n_colors: int = 1) -> "DDStrategy":
        """Create strategy by repeating single sequence for all colors."""
        return cls({i: sequence.copy() for i in range(n_colors)})

    def get_sequence(self, color: int) -> DDSequence:
        """Get sequence for specific color."""
        return self.sequences[color]

    def __len__(self) -> int:
        return len(self.sequences)


class ColorAssignment:
    """Assignment of device qubits to colors."""

    def __init__(self, assignments: Dict[int, List[int]]):
        """
        Initialize color assignment.

        Args:
            assignments: Dictionary mapping colors to lists of qubit indices
        """
        self.assignments = assignments
        self._validate()

    def _validate(self):
        """Validate color assignments."""
        # Check for overlapping qubit assignments
        assigned = set()
        for qubits in self.assignments.values():
            overlap = assigned.intersection(qubits)
            if overlap:
                raise ValueError(f"Qubits {overlap} assigned to multiple colors")
            assigned.update(qubits)

    def get_color(self, qubit: int) -> Optional[int]:
        """Get color assigned to a qubit."""
        for color, qubits in self.assignments.items():
            if qubit in qubits:
                return color
        return None

    def get_qubits(self, color: int) -> List[int]:
        """Get qubits assigned to a color."""
        return self.assignments.get(color, [])

    @property
    def n_colors(self) -> int:
        """Number of colors used in assignment."""
        return len(self.assignments)
