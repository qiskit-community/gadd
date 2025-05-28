from dataclasses import dataclass
from typing import List, Dict, Optional, Union
from qiskit.circuit.library import IGate, XGate, YGate, U1Gate, RZGate, RXGate, RYGate
from qiskit import QuantumCircuit
import numpy as np


# Default gateset for DD pulses
def create_xb_gate():
    """Create Xb gate (X with additional phase)."""
    xbcircuit = QuantumCircuit(1, name="xb")
    xbcircuit.append(U1Gate(np.pi), [0])
    xbcircuit.x(0)
    xbcircuit.append(U1Gate(np.pi), [0])
    return xbcircuit.to_instruction()


def create_yb_gate():
    """Create Yb gate (Y with additional phase)."""
    ybcircuit = QuantumCircuit(1, name="yb")
    ybcircuit.append(U1Gate(np.pi), [0])
    ybcircuit.y(0)
    ybcircuit.append(U1Gate(np.pi), [0])
    return ybcircuit.to_instruction()


XbGate = create_xb_gate()
YbGate = create_yb_gate()

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

    def __post_init__(self):
        """Validate sequence after initialization."""
        if not isinstance(self.gates, list):
            raise TypeError("gates must be a list of strings")
        if not all(isinstance(g, str) for g in self.gates):
            raise TypeError("All gates must be strings")
        # Validate gate names if needed
        valid_gates = {"I", "X", "Y", "Z", "Ip", "Im", "Xp", "Xm", "Yp", "Ym", "Zp", "Zm"}
        for gate in self.gates:
            if gate not in valid_gates:
                raise ValueError(f"Invalid gate: {gate}. Must be one of {valid_gates}")

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

    def to_indices(self) -> List[int]:
        """Convert gate names to group element indices."""
        from .group_operations import GROUP_ELEMENTS

        # Map simplified names to full names
        name_map = {"I": "Ip", "X": "Xp", "Y": "Yp", "Z": "Zp"}

        indices = []
        for gate in self.gates:
            full_name = name_map.get(gate, gate)
            if full_name in GROUP_ELEMENTS:
                indices.append(GROUP_ELEMENTS[full_name])
            else:
                raise ValueError(f"Unknown gate: {gate}")
        return indices


class StandardSequences:
    """Standard DD sequence implementations."""

    def __init__(self):
        self._sequences = {
            # Non-staggered sequences
            "xy4": DDSequence(["X", "Y", "X", "Y"]),
            "cpmg": DDSequence(["X", "X"]),
            "edd": DDSequence(["X", "Y", "X", "Y", "Y", "X", "Y", "X"]),
            "baseline": DDSequence(["I"]),
            "urdd": DDSequence(["X", "Y", "X", "Y"]),  # Simplified URDD
            # For staggered versions, we use the same sequences
            # The staggering is applied during circuit padding
            "xy4_staggered": DDSequence(["X", "Y", "X", "Y"]),
            "cpmg_staggered": DDSequence(["X", "X"]),
            "edd_staggered": DDSequence(["X", "Y", "X", "Y", "Y", "X", "Y", "X"]),
        }

        # Track which sequences should be applied with staggering
        self._staggered_sequences = {"xy4_staggered", "cpmg_staggered", "edd_staggered"}

    def get(self, name: str) -> DDSequence:
        """Get a standard sequence by name."""
        if name.lower() not in self._sequences:
            raise ValueError(f"Unknown sequence: {name}. Available: {self.list_available()}")
        return self._sequences[name.lower()].copy()

    def is_staggered(self, name: str) -> bool:
        """Check if a sequence should be applied with staggering."""
        return name.lower() in self._staggered_sequences

    def list_available(self) -> List[str]:
        """List available standard sequences."""
        return list(self._sequences.keys())


class DDStrategy:
    """Collection of DD sequences assigned to different colors."""

    def __init__(self, sequences: Union[List[DDSequence], Dict[int, DDSequence]]):
        """
        Initialize DD strategy with sequences for each color.

        Args:
            sequences: Either a list of DDSequence objects (assigned to colors 0, 1, 2, ...)
                      or a dictionary mapping color indices to DD sequences

        Raises:
            ValueError: If sequences is empty or invalid type
        """
        if isinstance(sequences, list):
            if not sequences:
                raise ValueError("Must provide at least one sequence")
            self.sequences = {i: seq for i, seq in enumerate(sequences)}
        elif isinstance(sequences, dict):
            if not sequences:
                raise ValueError("Must provide at least one sequence")
            self.sequences = sequences
        else:
            raise TypeError("sequences must be a list or dict")

        self._validate()

    def _validate(self):
        """Validate sequences."""
        for color, seq in self.sequences.items():
            if not isinstance(seq, DDSequence):
                raise TypeError(f"Sequence for color {color} must be a DDSequence")
            if not isinstance(color, int) or color < 0:
                raise ValueError(f"Color index must be a non-negative integer, got {color}")

    @classmethod
    def from_single_sequence(cls, sequence: DDSequence, n_colors: int = 1) -> "DDStrategy":
        """Create strategy by repeating single sequence for all colors."""
        if n_colors <= 0:
            raise ValueError("n_colors must be positive")
        return cls([sequence.copy() for _ in range(n_colors)])

    def get_sequence(self, color: int) -> DDSequence:
        """Get sequence for specific color."""
        if color not in self.sequences:
            raise KeyError(f"No sequence defined for color {color}")
        return self.sequences[color]

    def __len__(self) -> int:
        return len(self.sequences)

    def to_dict(self) -> Dict[str, any]:
        """Convert strategy to dictionary for serialization."""
        return {"sequences": {color: seq.gates for color, seq in self.sequences.items()}}

    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> "DDStrategy":
        """Create strategy from dictionary."""
        sequences = {int(color): DDSequence(gates) for color, gates in data["sequences"].items()}
        return cls(sequences)


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
        # Create reverse mapping for efficiency
        self._qubit_to_color = {}
        for color, qubits in assignments.items():
            for qubit in qubits:
                self._qubit_to_color[qubit] = color

    def _validate(self):
        """Validate color assignments."""
        # Check for overlapping qubit assignments
        assigned = set()
        for color, qubits in self.assignments.items():
            if not isinstance(qubits, list):
                raise TypeError(f"Qubits for color {color} must be a list")
            overlap = assigned.intersection(qubits)
            if overlap:
                raise ValueError(f"Qubits {overlap} assigned to multiple colors")
            assigned.update(qubits)

    def get_color(self, qubit: int) -> Optional[int]:
        """Get color assigned to a qubit."""
        return self._qubit_to_color.get(qubit)

    def get_qubits(self, color: int) -> List[int]:
        """Get qubits assigned to a color."""
        return self.assignments.get(color, [])

    @property
    def n_colors(self) -> int:
        """Number of colors used in assignment."""
        return len(self.assignments)
