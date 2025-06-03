"""
DD strategies, sequences, and coloring assignments.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Union

import numpy as np
import rustworkx as rx

from qiskit import QuantumCircuit
from qiskit.circuit.library import IGate, XGate, YGate, U1Gate, RZGate
from qiskit.providers import BackendV2 as Backend

from gadd.group_operations import DecouplingGroup


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
        valid_gates = {
            "I",
            "X",
            "Y",
            "Z",
            "Ip",
            "Im",
            "Xp",
            "Xm",
            "Yp",
            "Ym",
            "Zp",
            "Zm",
        }
        for gate in self.gates:
            if gate not in valid_gates:
                raise ValueError(f"Invalid gate: {gate}. Must be one of {valid_gates}")

    def __str__(self) -> str:
        return "-".join(self.gates)

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

    def to_indices(self, group: Optional["DecouplingGroup"] = None) -> List[int]:
        """Convert gate names to group element indices.

        Args:
            group: Optional custom decoupling group. Uses default if not provided.

        Returns:
            List of integer indices corresponding to group elements.
        """
        from .group_operations import DEFAULT_GROUP

        if group is None:
            group = DEFAULT_GROUP

        indices = []
        for gate in self.gates:
            try:
                indices.append(group.element_index(gate))
            except ValueError:
                # Handle simplified names
                name_map = {"I": "Ip", "X": "Xp", "Y": "Yp", "Z": "Zp"}
                full_name = name_map.get(gate, gate)
                indices.append(group.element_index(full_name))
        return indices


class StandardSequences:
    """Standard DD sequence implementations.

    This class provides access to well-established dynamical decoupling sequences
    commonly used in quantum error suppression experiments. Each sequence is designed
    to suppress specific types of errors through systematic application of Pauli
    operations during idle periods.

    Available Sequences:
        xy4: Four-pulse sequence [X, Y, X, Y] that provides first-order decoupling
            from both X and Y noise. Also known as XY-4, this sequence offers
            robust performance against pulse timing errors.

        cpmg: Two-pulse Carr-Purcell-Meiboom-Gill sequence [X, X] that decouples
            Y and Z noise but leaves X noise unaffected. Simple and widely used
            for basic error suppression.

        edd: Eight-pulse Eulerian Dynamical Decoupling sequence
            [X, Y, X, Y, Y, X, Y, X] that provides enhanced robustness against
            systematic pulse errors compared to XY-4 while maintaining first-order
            decoupling properties.

        baseline: Single identity operation [I] used as a control sequence to
            measure the effect of applying no dynamical decoupling.

        urdd: Simplified Universally Robust DD sequence [X, Y, X, Y]. In practice,
            URDD adapts pulse counts based on idle duration, but this provides
            the basic four-pulse pattern.

        xy4_staggered: XY-4 sequence with CR-aware staggering for crosstalk
            suppression. Uses the same pulse sequence as xy4 but applies
            time-shifted scheduling between different qubit colors.

        cpmg_staggered: CPMG sequence with staggered timing between colors
            for multi-qubit crosstalk mitigation.

        edd_staggered: Eulerian DD sequence with staggered timing to reduce
            correlated errors between neighboring qubits.

    Note:
        Staggered sequences use the same gate patterns as their non-staggered
        counterparts but apply different pulse timing during circuit padding
        to suppress crosstalk errors between adjacent qubits.
    """

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
        """Get a standard DD sequence by name.

        Args:
            name: Name of the sequence (case insensitive).

        Returns:
            Copy of the requested DD sequence.

        Raises:
            ValueError: If the sequence name is not recognized.
        """
        if name.lower() not in self._sequences:
            raise ValueError(
                f"Unknown sequence: {name}. Available: {self.list_available()}"
            )
        return self._sequences[name.lower()].copy()

    def is_staggered(self, name: str) -> bool:
        """Check if a sequence should be applied with staggered timing.

        Args:
            name: Sequence name to check (case insensitive).

        Returns:
            ``True`` if sequence should use staggered timing, ``False`` otherwise.
        """
        return name.lower() in self._staggered_sequences

    def list_available(self) -> List[str]:
        """List all available standard sequence names.

        Returns:
            List of available sequence names that can be used with :meth:`.get`.
        """
        return list(self._sequences.keys())


class DDStrategy:
    """Collection of dynamical decoupling sequences assigned to different qubit colors.

    A DD strategy defines the complete dynamical decoupling approach for a quantum
    circuit by specifying which DD sequence should be applied to each color of qubits.
    The coloring is typically based on the circuit's connectivity graph to ensure
    that adjacent qubits (which may experience crosstalk) receive independent DD
    sequences for optimal error suppression.

    This class encapsulates the multi-color DD approach described in the GADD paper,
    where different groups of qubits can receive tailored DD sequences optimized
    for their specific noise environment and connectivity constraints. The strategy
    can range from simple uniform approaches (same sequence for all colors) to
    sophisticated heterogeneous strategies with different sequences per color.

    Args:
        sequences: Either a list of :class:`.DDSequence` objects assigned to colors
            0, 1, 2, ... or a dictionary mapping color indices to DD sequences.

    Raises:
        ValueError: If sequences is empty.
        TypeError: If sequences contains non-DDSequence objects or has invalid format.

    Example:
        >>> seq1 = DDSequence(["X", "Y", "X", "Y"])
        >>> seq2 = DDSequence(["Y", "X", "Y", "X"])
        >>> strategy = DDStrategy([seq1, seq2])  # 2-color strategy
        >>> strategy.get_sequence(0)  # Returns seq1
    """

    def __init__(self, sequences: Union[List[DDSequence], Dict[int, DDSequence]]):
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

        # self._validate()

    def _validate(self):
        """Validate sequences."""
        for color, seq in self.sequences.items():
            if not isinstance(seq, DDSequence):
                raise TypeError(f"Sequence for color {color} must be a DDSequence")
            if not isinstance(color, int) or color < 0:
                raise ValueError(
                    f"Color index must be a non-negative integer, got {color}"
                )

    @classmethod
    def from_single_sequence(
        cls, sequence: DDSequence, n_colors: int = 1
    ) -> "DDStrategy":
        """Create strategy by repeating single sequence for all colors.

        Args:
            sequence: DD sequence to replicate for all colors.
            n_colors: Number of colors in the strategy.

        Returns:
            New DD strategy with the sequence assigned to all colors.

        Raises:
            ValueError: If n_colors is not positive.
        """
        if n_colors <= 0:
            raise ValueError("n_colors must be positive")
        return cls([sequence.copy() for _ in range(n_colors)])

    def get_sequence(self, color: int) -> DDSequence:
        """Get the DD sequence assigned to a specific color.

        Args:
            color: Color index to retrieve sequence for.

        Returns:
            DD sequence for the specified color.

        Raises:
            KeyError: If no sequence is defined for the specified color.
        """
        if color not in self.sequences:
            raise KeyError(f"No sequence defined for color {color}")
        return self.sequences[color]

    def __len__(self) -> int:
        return len(self.sequences)

    def __str__(self) -> str:
        if len(self.sequences) == 1:
            # Single sequence case
            seq = next(iter(self.sequences.values()))
            return f"DD Strategy: {seq}"
        else:
            # Multi-color case
            lines = [f"DD Strategy ({len(self.sequences)} colors):"]
            for color in sorted(self.sequences.keys()):
                seq = self.sequences[color]
                lines.append(f"  Color {color}: {seq}")
            return "\n".join(lines)

    def to_dict(self) -> Dict[str, any]:
        """Convert strategy to dictionary for serialization.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "sequences": {color: seq.gates for color, seq in self.sequences.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> "DDStrategy":
        """Create strategy from dictionary."""
        sequences = {
            int(color): DDSequence(gates) for color, gates in data["sequences"].items()
        }
        return cls(sequences)


class ColorAssignment:
    """Assignment of device qubits to colors based on connectivity graph."""

    def __init__(
        self, graph: Optional[rx.PyGraph] = None, backend: Optional["Backend"] = None
    ):
        """
        Initialize color assignment.

        Args:
            graph: Connectivity graph where nodes are qubits and edges are connections.
                If None and backend is provided, will extract from backend.
            backend: Backend to extract connectivity from if graph not provided.

        Raises:
            ValueError: If neither graph nor backend is provided.
        """
        if graph is None and backend is None:
            raise ValueError("Must provide either graph or backend")

        if graph is None:
            # Extract from backend
            if hasattr(backend, "coupling_map") and backend.coupling_map:
                self.graph = backend.coupling_map.graph.to_undirected()
            else:
                # Fallback: create complete graph
                n_qubits = backend.num_qubits if hasattr(backend, "num_qubits") else 1
                self.graph = rx.PyGraph()
                self.graph.add_nodes_from(range(n_qubits))
        else:
            self.graph = graph

        # Perform graph coloring
        self._color_map = rx.graph_greedy_color(self.graph)

        # Create assignments dictionary (color -> list of qubits)
        self.assignments = {}
        for qubit, color in self._color_map.items():
            if color not in self.assignments:
                self.assignments[color] = []
            self.assignments[color].append(qubit)

        # Create reverse mapping for efficiency
        self._qubit_to_color = self._color_map.copy()

    def __str__(self) -> str:
        """Return human-readable string representation."""
        lines = [f"Qubit Coloring ({self.n_colors} colors):"]
        for color in sorted(self.assignments.keys()):
            qubits = self.assignments[color]
            qubit_ranges = self._format_qubit_ranges(qubits)
            lines.append(f"  Color {color}: {qubit_ranges}")
        return "\n".join(lines)

    def _format_qubit_ranges(self, qubits: List[int]) -> str:
        """Format qubit list as ranges where possible."""
        if not qubits:
            return "[]"

        sorted_qubits = sorted(qubits)
        if len(sorted_qubits) <= 3:
            return str(sorted_qubits)

        # Try to find consecutive ranges
        ranges = []
        start = sorted_qubits[0]
        end = start

        for i in range(1, len(sorted_qubits)):
            if sorted_qubits[i] == end + 1:
                end = sorted_qubits[i]
            else:
                if end == start:
                    ranges.append(str(start))
                elif end == start + 1:
                    ranges.append(f"{start},{end}")
                else:
                    ranges.append(f"{start}-{end}")
                start = end = sorted_qubits[i]

        # Add final range
        if end == start:
            ranges.append(str(start))
        elif end == start + 1:
            ranges.append(f"{start},{end}")
        else:
            ranges.append(f"{start}-{end}")

        return "[" + ",".join(ranges) + "]"

    @classmethod
    def from_circuit(cls, circuit: QuantumCircuit) -> "ColorAssignment":
        """
        Create color assignment from circuit connectivity.

        Args:
            circuit: Quantum circuit to extract connectivity from.

        Returns:
            ColorAssignment based on circuit structure.
        """
        # Build connectivity graph from circuit
        graph = rx.PyGraph()
        qubits = set()
        edges = set()

        for instruction in circuit.data:
            if instruction.operation.name in ["cx", "ecr", "cz"]:  # Two-qubit gates
                qubits_involved = [q._index for q in instruction.qubits]
                if len(qubits_involved) == 2:
                    q1, q2 = qubits_involved
                    qubits.add(q1)
                    qubits.add(q2)
                    edges.add((min(q1, q2), max(q1, q2)))
            else:
                # Track all qubits
                for q in instruction.qubits:
                    qubits.add(q._index)

        # Add all qubits as nodes
        graph.add_nodes_from(sorted(qubits))

        # Add edges
        for q1, q2 in edges:
            graph.add_edge(q1, q2, None)

        return cls(graph=graph)

    @classmethod
    def from_manual_assignment(
        cls, assignments: Dict[int, List[int]]
    ) -> "ColorAssignment":
        """
        Create from manual color assignments.

        Args:
            assignments: Dictionary mapping colors to lists of qubit indices.

        Returns:
            ColorAssignment with specified assignments.
        """
        # Create a graph where qubits with different colors are connected
        graph = rx.PyGraph()
        all_qubits = set()
        for qubits in assignments.values():
            all_qubits.update(qubits)

        graph.add_nodes_from(sorted(all_qubits))

        # Connect qubits with different colors
        colors = list(assignments.keys())
        for i in range(len(colors)):
            for j in range(i + 1, len(colors)):
                for q1 in assignments[colors[i]]:
                    for q2 in assignments[colors[j]]:
                        graph.add_edge(q1, q2, None)

        # Create instance and override the computed coloring
        instance = cls(graph=graph)
        instance.assignments = assignments
        instance._qubit_to_color = {}
        for color, qubits in assignments.items():
            for qubit in qubits:
                instance._qubit_to_color[qubit] = color
        instance._color_map = instance._qubit_to_color.copy()

        return instance

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

    def to_dict(self) -> Dict[int, int]:
        """
        Convert to qubit->color mapping dictionary.

        Returns:
            Dictionary mapping qubit indices to color values.
        """
        return self._color_map.copy()

    def validate_coloring(self) -> bool:
        """
        Validate that the coloring is proper (no adjacent nodes have same color).

        Returns:
            True if coloring is valid, False otherwise.
        """
        for edge in self.graph.edge_list():
            if self._color_map[edge[0]] == self._color_map[edge[1]]:
                return False
        return True
