# File: tests/test_utility_functions.py

import unittest
from qiskit.result import QuasiDistribution, Counts
from gadd.utility_functions import (
    UtilityFactory,
    SuccessProbability,
    OneNormDistance,
    GHZUtility,
    CustomUtility,
)


class TestUtilityFunctions(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
