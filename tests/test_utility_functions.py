"""Tests for utility functions."""

import unittest
from qiskit.result import QuasiDistribution, Counts
from gadd.utility_functions import (
    normalize_counts,
    UtilityFactory,
    SuccessProbability,
    OneNormDistance,
    GHZUtility,
    CustomUtility,
)


class TestUtilityFunctions(unittest.TestCase):
    """
    Test the utility function class and helper functions.
    """

    def test_normalize_counts_edge_cases(self):
        """Test normalize_counts with edge cases."""
        # Test empty counts
        with self.assertRaises(ValueError) as context:
            normalize_counts({})
        self.assertIn("Empty counts", str(context.exception))

        # Test zero total counts
        with self.assertRaises(ValueError) as context:
            normalize_counts({"00": 0, "11": 0})
        self.assertIn(
            "Total counts/probabilities must be positive", str(context.exception)
        )

        # Test integer keys with different bit lengths
        counts = {0: 100, 15: 200, 7: 300}  # 0b0, 0b1111, 0b111
        normalized = normalize_counts(counts)
        self.assertIn("0000", normalized)  # Should pad to 4 bits (max needed)
        self.assertIn("1111", normalized)
        self.assertIn("0111", normalized)
        self.assertAlmostEqual(sum(normalized.values()), 1.0)

    def test_success_probability(self):
        """Test success probability utility function."""
        utility = UtilityFactory.success_probability("00")
        # Testing different data formats
        counts = QuasiDistribution({"00": 0.8, "01": 0.1, "10": 0.1})
        counts2 = Counts({"00": 800, "01": 100, "10": 100})
        counts3 = {0: 800, 1: 100, 2: 100}
        self.assertAlmostEqual(utility.compute(counts), 0.8)
        self.assertAlmostEqual(utility.compute(counts2), 0.8)
        self.assertAlmostEqual(utility.compute(counts3), 0.8)
        self.assertEqual(utility.get_name(), "Success Probability (|00⟩)")

    def test_one_norm_distance(self):
        """Test 1-norm distance utility function."""
        ideal = {"00": 0.5, "11": 0.5}
        utility = UtilityFactory.one_norm(ideal)

        # Perfect match
        counts1 = QuasiDistribution({"00": 0.5, "11": 0.5})
        self.assertAlmostEqual(utility.compute(counts1), 1.0)

        # Complete mismatch
        counts2 = QuasiDistribution({"01": 0.5, "10": 0.5})
        self.assertAlmostEqual(utility.compute(counts2), 0.0)

        # Partial match
        counts3 = QuasiDistribution({"00": 0.25, "11": 0.25, "01": 0.5})
        self.assertAlmostEqual(utility.compute(counts3), 0.5)

    def test_ghz_utility(self):
        """Test GHZ state utility function."""
        utility = UtilityFactory.ghz_state(2)  # 2-qubit GHZ

        # Perfect GHZ
        counts1 = QuasiDistribution({"00": 0.5, "11": 0.5})
        self.assertAlmostEqual(utility.compute(counts1), 1.0)

        # Imperfect GHZ
        counts2 = QuasiDistribution({"00": 0.4, "11": 0.4, "01": 0.1, "10": 0.1})
        self.assertAlmostEqual(utility.compute(counts2), 0.8)

    def test_custom_utility(self):
        """Test custom utility function."""

        def custom_func(counts):
            return counts.get("00", 0.0)

        utility = UtilityFactory.custom(custom_func, "Test Utility")
        counts = QuasiDistribution({"00": 0.75, "11": 0.25})

        self.assertAlmostEqual(utility.compute(counts), 0.75)
        self.assertEqual(utility.get_name(), "Test Utility")

    def test_verify_state(self):
        """Test verify_state method."""
        utility = SuccessProbability("00")

        # Test valid state
        counts = {"00": 0.5, "11": 0.5}
        utility.verify_state("00", counts)  # Should not raise

        # Test mismatched length
        with self.assertRaises(ValueError) as context:
            utility.verify_state("000", counts)
        self.assertIn("length", str(context.exception))

        # Test non-binary string
        with self.assertRaises(ValueError) as context:
            utility.verify_state("0x", counts)
        self.assertIn("binary string", str(context.exception))

        # Test empty counts
        with self.assertRaises(ValueError) as context:
            utility.verify_state("00", {})
        self.assertIn("Empty counts", str(context.exception))

    def test_success_probability_integer_target(self):
        """Test SuccessProbability with integer target state."""
        # Test with integer target
        utility = SuccessProbability(3)  # Binary: 11
        counts = {"00": 0.25, "01": 0.25, "10": 0.25, "11": 0.25}
        self.assertAlmostEqual(utility.compute(counts), 0.25)

        # Test that it pads correctly
        utility2 = SuccessProbability(1)  # Binary: 1
        counts2 = {"000": 0.5, "001": 0.5}
        self.assertAlmostEqual(utility2.compute(counts2), 0.5)

    def test_success_probability_invalid_target(self):
        """Test SuccessProbability with invalid target."""
        with self.assertRaises(ValueError) as context:
            SuccessProbability("0a1")
        self.assertIn("binary string", str(context.exception))

    def test_one_norm_dimension_mismatch(self):
        """Test OneNormDistance with mismatched dimensions."""
        utility = OneNormDistance({"00": 0.5, "11": 0.5})

        # Test with different length states
        counts = {"000": 0.5, "111": 0.5}
        with self.assertRaises(ValueError) as context:
            utility.compute(counts)
        self.assertIn("dimensions don't match", str(context.exception))

    def test_ghz_utility_invalid_qubits(self):
        """Test GHZUtility with invalid qubit number."""
        with self.assertRaises(ValueError) as context:
            GHZUtility(0)
        self.assertIn("positive", str(context.exception))

        with self.assertRaises(ValueError) as context:
            GHZUtility(-1)
        self.assertIn("positive", str(context.exception))

    def test_counts_type_variations(self):
        """Test various count input types."""
        utility = SuccessProbability("00")

        # Test with Counts object (mock it since we don't want to import from qiskit.result)
        class MockCounts(dict):
            pass

        mock_counts = MockCounts({"00": 800, "01": 200})
        result = utility.compute(mock_counts)
        self.assertAlmostEqual(result, 0.8)

    def test_all_utility_names(self):
        """Test get_name methods for all utility types."""
        # SuccessProbability
        util1 = SuccessProbability("101")
        self.assertEqual(util1.get_name(), "Success Probability (|101⟩)")

        # OneNormDistance
        util2 = OneNormDistance({"00": 1.0})
        self.assertEqual(util2.get_name(), "1-Norm Distance")

        # GHZUtility
        util3 = GHZUtility(5)
        self.assertEqual(util3.get_name(), "GHZ State Fidelity")

        # CustomUtility with custom name
        util4 = CustomUtility(lambda x: 0.5, "My Custom Utility")
        self.assertEqual(util4.get_name(), "My Custom Utility")


if __name__ == "__main__":
    unittest.main()
