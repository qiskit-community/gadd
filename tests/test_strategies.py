"""Test DD strategies."""

import unittest
from gadd.strategies import DDSequence, StandardSequences, ColorAssignment, DDStrategy
import rustworkx as rx
from qiskit import QuantumCircuit
from .fixtures import MockBackend


class TestDDSequence(unittest.TestCase):
    """Test DDSequence class."""

    def test_init_valid(self):
        """Test valid initialization."""
        seq = DDSequence(["X", "Y", "X", "Y"])
        self.assertEqual(len(seq), 4)
        self.assertEqual(seq[0], "X")
        self.assertEqual(seq.gates, ["X", "Y", "X", "Y"])

    def test_init_invalid_type(self):
        """Test initialization with invalid types."""
        with self.assertRaises(TypeError):
            DDSequence("XYXY")  # String instead of list

        with self.assertRaises(TypeError):
            DDSequence([1, 2, 3])  # Numbers instead of strings

    def test_sequence_creation(self):
        """Test basic sequence creation and properties."""
        gates = ["X", "Y", "X", "Y"]
        seq = DDSequence(gates)
        self.assertEqual(len(seq), 4)
        self.assertEqual(seq.gates, gates)
        self.assertEqual(seq[0], "X")

    def test_init_invalid_gates(self):
        """Test initialization with invalid gate names."""
        with self.assertRaises(ValueError):
            DDSequence(["X", "Y", "Q"])  # Q is not valid

    def test_len_and_getitem(self):
        """Test length and indexing."""
        seq = DDSequence(["Xp", "Ym", "Zp"])
        self.assertEqual(len(seq), 3)
        self.assertEqual(seq[0], "Xp")
        self.assertEqual(seq[1], "Ym")
        self.assertEqual(seq[2], "Zp")
        self.assertEqual(seq[-1], "Zp")

    def test_equality(self):
        """Test equality comparison."""
        seq1 = DDSequence(["X", "Y"])
        seq2 = DDSequence(["X", "Y"])
        seq3 = DDSequence(["Y", "X"])

        self.assertEqual(seq1, seq2)
        self.assertNotEqual(seq1, seq3)
        self.assertNotEqual(seq1, "XY")  # Different type

    def test_copy(self):
        """Test deep copy."""
        seq1 = DDSequence(["X", "Y"])
        seq2 = seq1.copy()

        # Should be equal but different objects
        self.assertEqual(seq1, seq2)
        self.assertIsNot(seq1, seq2)
        self.assertIsNot(seq1.gates, seq2.gates)

        # Modifying copy shouldn't affect original
        seq2.gates.append("Z")
        self.assertEqual(len(seq1), 2)
        self.assertEqual(len(seq2), 3)

    def test_to_indices(self):
        """Test conversion to group element indices."""
        seq = DDSequence(["X", "Y", "Xm", "Yp"])
        indices = seq.to_indices()
        self.assertEqual(indices, [2, 4, 3, 4])  # Xp=2, Yp=4, Xm=3

    def test_dd_strategy_init_dict(self):
        """Test DDStrategy initialization with dictionary."""
        seq1 = DDSequence(["X", "Y"])
        seq2 = DDSequence(["Y", "X"])

        # Init with dict
        strategy = DDStrategy({0: seq1, 2: seq2})  # Non-contiguous colors
        self.assertEqual(len(strategy), 2)
        self.assertEqual(strategy.get_sequence(0), seq1)
        self.assertEqual(strategy.get_sequence(2), seq2)

    def test_dd_strategy_validation(self):
        """Test DDStrategy validation."""
        # Invalid type for sequences
        with self.assertRaises(TypeError):
            DDStrategy("not a list or dict")

        # Empty sequences
        with self.assertRaises(ValueError):
            DDStrategy([])

        with self.assertRaises(ValueError):
            DDStrategy({})

        # Non-DDSequence in list
        with self.assertRaises(TypeError):
            DDStrategy([["X", "Y"]])  # List instead of DDSequence

        # Invalid color index (negative)
        seq = DDSequence(["X"])
        with self.assertRaises(ValueError):
            DDStrategy({-1: seq})

        # Invalid color index (non-integer)
        with self.assertRaises(ValueError):
            DDStrategy({"a": seq})

    def test_dd_strategy_serialization(self):
        """Test DDStrategy to_dict and from_dict."""
        seq1 = DDSequence(["X", "Y", "X", "Y"])
        seq2 = DDSequence(["Y", "X", "Y", "X"])
        strategy = DDStrategy([seq1, seq2])

        # Test to_dict
        data = strategy.to_dict()
        expected = {"sequences": {0: ["X", "Y", "X", "Y"], 1: ["Y", "X", "Y", "X"]}}
        self.assertEqual(data, expected)

        # Test from_dict
        strategy2 = DDStrategy.from_dict(data)
        self.assertEqual(len(strategy2), 2)
        self.assertEqual(strategy2.get_sequence(0).gates, seq1.gates)
        self.assertEqual(strategy2.get_sequence(1).gates, seq2.gates)

    def test_dd_strategy_get_sequence_error(self):
        """Test DDStrategy get_sequence with invalid color."""
        seq = DDSequence(["X", "Y"])
        strategy = DDStrategy([seq])

        # Valid color
        self.assertEqual(strategy.get_sequence(0), seq)

        # Invalid color
        with self.assertRaises(KeyError) as context:
            strategy.get_sequence(1)
        self.assertIn("No sequence defined for color 1", str(context.exception))


class TestStandardSequences(unittest.TestCase):
    """Test StandardSequences class."""

    def setUp(self):
        self.std_seqs = StandardSequences()

    def test_get_valid(self):
        """Test getting valid sequences."""
        xy4 = self.std_seqs.get("xy4")
        self.assertIsInstance(xy4, DDSequence)
        self.assertEqual(xy4.gates, ["X", "Y", "X", "Y"])

        cpmg = self.std_seqs.get("cpmg")
        self.assertEqual(cpmg.gates, ["X", "X"])

    def test_get_invalid(self):
        """Test getting invalid sequence."""
        with self.assertRaises(ValueError):
            self.std_seqs.get("invalid_seq")

    def test_get_returns_copy(self):
        """Test that get returns a copy."""
        xy4_1 = self.std_seqs.get("xy4")
        xy4_2 = self.std_seqs.get("xy4")

        self.assertEqual(xy4_1, xy4_2)
        self.assertIsNot(xy4_1, xy4_2)

    def test_list_available(self):
        """Test listing available sequences."""
        available = self.std_seqs.list_available()
        self.assertIsInstance(available, list)
        self.assertIn("xy4", available)
        self.assertIn("cpmg", available)
        self.assertIn("edd", available)

    def test_standard_sequences_edge_cases(self):
        """Test StandardSequences edge cases."""
        std_seqs = StandardSequences()

        # Test case insensitive get
        xy4_lower = std_seqs.get("xy4")
        xy4_upper = std_seqs.get("XY4")
        self.assertEqual(xy4_lower.gates, xy4_upper.gates)

        # Test is_staggered
        self.assertTrue(std_seqs.is_staggered("xy4_staggered"))
        self.assertTrue(std_seqs.is_staggered("XY4_STAGGERED"))  # Case insensitive
        self.assertFalse(std_seqs.is_staggered("xy4"))
        self.assertFalse(std_seqs.is_staggered("unknown_seq"))


class TestColorAssignment(unittest.TestCase):
    def test_color_assignment_from_backend(self):
        """Test ColorAssignment from backend."""

        backend = MockBackend(num_qubits=5)
        colors = ColorAssignment(backend=backend)

        # Should have created a coloring
        self.assertGreater(colors.n_colors, 0)
        self.assertLessEqual(colors.n_colors, 5)

        # All qubits should be colored
        all_colored_qubits = set()
        for color in range(colors.n_colors):
            all_colored_qubits.update(colors.get_qubits(color))
        self.assertEqual(len(all_colored_qubits), 5)

        # Validate coloring
        self.assertTrue(colors.validate_coloring())

    def test_color_assignment_from_circuit(self):
        """Test ColorAssignment from circuit structure."""
        # Create a circuit with specific connectivity
        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)

        colors = ColorAssignment.from_circuit(qc)

        # Linear chain should need at most 2 colors
        self.assertLessEqual(colors.n_colors, 2)

        # Adjacent qubits should have different colors
        self.assertNotEqual(colors.get_color(0), colors.get_color(1))
        self.assertNotEqual(colors.get_color(1), colors.get_color(2))
        self.assertNotEqual(colors.get_color(2), colors.get_color(3))

    def test_color_assignment_from_manual(self):
        """Test manual color assignment."""
        assignments = {0: [0, 2, 4], 1: [1, 3], 2: [5]}
        colors = ColorAssignment.from_manual_assignment(assignments)

        self.assertEqual(colors.n_colors, 3)
        self.assertEqual(colors.get_color(0), 0)
        self.assertEqual(colors.get_color(1), 1)
        self.assertEqual(colors.get_color(5), 2)
        self.assertEqual(colors.get_qubits(0), [0, 2, 4])

    def test_color_assignment_validation(self):
        """Test color assignment validation."""
        # Create graph with triangle
        graph = rx.PyGraph()
        graph.add_nodes_from([0, 1, 2])
        graph.add_edges_from([(0, 1, None), (1, 2, None), (2, 0, None)])

        colors = ColorAssignment(graph=graph)

        # Should need 3 colors for triangle
        self.assertEqual(colors.n_colors, 3)
        self.assertTrue(colors.validate_coloring())

    def test_color_assignment_to_dict(self):
        """Test conversion to dictionary."""
        assignments = {0: [0, 2], 1: [1, 3]}
        colors = ColorAssignment.from_manual_assignment(assignments)

        color_dict = colors.to_dict()
        self.assertEqual(color_dict[0], 0)
        self.assertEqual(color_dict[1], 1)
        self.assertEqual(color_dict[2], 0)
        self.assertEqual(color_dict[3], 1)

    def test_color_assignment_error_handling(self):
        """Test error handling in ColorAssignment."""
        # No graph or backend provided
        with self.assertRaises(ValueError):
            ColorAssignment()


class TestHelperFunctions(unittest.TestCase):
    def test_create_gate_functions(self):
        """Test the gate creation helper functions."""
        from gadd.strategies import create_xb_gate, create_yb_gate

        # Test that functions return instruction objects
        xb_gate = create_xb_gate()
        self.assertEqual(xb_gate.name, "xb")
        self.assertEqual(xb_gate.num_qubits, 1)

        yb_gate = create_yb_gate()
        self.assertEqual(yb_gate.name, "yb")
        self.assertEqual(yb_gate.num_qubits, 1)


if __name__ == "__main__":
    unittest.main()
