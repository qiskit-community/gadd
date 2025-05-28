"""Test DD sequences."""

import unittest
from gadd.sequences import DDSequence, StandardSequences


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


if __name__ == "__main__":
    unittest.main()
