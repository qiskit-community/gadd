import unittest
import numpy as np
from gadd.sequences import DDSequence, StandardSequences, DDStrategy

"""Test the DD sequence classes."""


class TestDDSequence(unittest.TestCase):
    def test_sequence_creation(self):
        """Test basic sequence creation and properties."""
        gates = ["X", "Y", "X", "Y"]
        seq = DDSequence(gates)
        self.assertEqual(len(seq), 4)
        self.assertEqual(seq.gates, gates)
        self.assertEqual(seq[0], "X")

    def test_sequence_equality(self):
        """Test sequence equality comparison."""
        seq1 = DDSequence(["X", "Y", "X", "Y"])
        seq2 = DDSequence(["X", "Y", "X", "Y"])
        seq3 = DDSequence(["Y", "X", "Y", "X"])

        self.assertEqual(seq1, seq2)
        self.assertNotEqual(seq1, seq3)

    # def test_sequence_copy(self):
    #     """Test sequence copying."""
    #     seq1 = DDSequence(["X", "Y", "X", "Y"])
    #     seq2 = seq1.copy()

    #     self.assertEqual(seq1, seq2)
    #     seq2.gates[0] = "Z"
    #     self.assertNotEqual(seq1, seq2)


class TestStandardSequences(unittest.TestCase):
    def setUp(self):
        self.std_seqs = StandardSequences()

    def test_standard_sequences_available(self):
        """Test that all standard sequences are available."""
        expected = {"xy4", "cpmg", "edd", "baseline"}
        available = set(self.std_seqs.list_available())
        self.assertEqual(available, expected)

    def test_sequence_retrieval(self):
        """Test retrieving specific sequences."""
        xy4 = self.std_seqs.get("xy4")
        self.assertEqual(xy4.gates, ["X", "Y", "X", "Y"])

        cpmg = self.std_seqs.get("cpmg")
        self.assertEqual(cpmg.gates, ["X", "X"])

    def test_invalid_sequence(self):
        """Test requesting invalid sequence."""
        invalid = self.std_seqs.get("invalid_name")
        self.assertIsNone(invalid)


if __name__ == "__main__":
    unittest.main()
