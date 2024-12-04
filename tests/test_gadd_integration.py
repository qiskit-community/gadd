# File: tests/test_gadd_integration.py

import unittest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import FakeBackend
from qiskit_ibm_runtime import Sampler
from gadd import GADD
from gadd.utility_functions import UtilityFactory


class TestGADDIntegration(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.backend = FakeBackend()
        self.sampler = Sampler()

    def test_training_basic(self):
        """Test basic training functionality."""
        # Create simple test circuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        # Create GADD instance with success probability utility
        utility = UtilityFactory.success_probability("00")
        gadd = GADD(backend=self.backend, utility_function=utility)

        # Configure for quick test
        gadd.config.population_size = 4
        gadd.config.n_iterations = 2

        # Train
        seq, results = gadd.train(sampler=self.sampler, training_circuit=qc, save_iterations=True)

        # Basic checks
        self.assertIsNotNone(seq)
        self.assertEqual(len(results.iteration_data), 2)
        self.assertTrue(0 <= max(results.iteration_data[1]["population"].values()) <= 1)

    def test_sequence_application(self):
        """Test applying trained sequence to new circuit."""
        # Create training and target circuits
        train_qc = QuantumCircuit(2)
        train_qc.h([0, 1])
        train_qc.measure_all()

        target_qc = QuantumCircuit(2)
        target_qc.h(0)
        target_qc.cx(0, 1)
        target_qc.measure_all()

        # Train on simple circuit
        utility = UtilityFactory.ghz_state(2)
        gadd = GADD(backend=self.backend, utility_function=utility)
        gadd.config.population_size = 4
        gadd.config.n_iterations = 2

        seq, _ = gadd.train(sampler=self.sampler, training_circuit=train_qc)

        # Apply to target circuit
        results = gadd.run(sequence=seq, target_circuit=target_qc, sampler=self.sampler)

        # Check results format
        self.assertIn("success_probability", results)
        self.assertTrue(0 <= results["success_probability"] <= 1)

    def test_comparison_sequences(self):
        """Test comparison with standard sequences."""
        qc = QuantumCircuit(2)
        qc.h([0, 1])
        qc.measure_all()

        utility = UtilityFactory.ghz_state(2)
        gadd = GADD(backend=self.backend, utility_function=utility)
        gadd.config.population_size = 4
        gadd.config.n_iterations = 2

        seq, results = gadd.train(
            sampler=self.sampler, training_circuit=qc, comparison_seqs=["xy4", "cpmg"]
        )

        # Check comparison data
        self.assertIn("xy4", results.comparison_data)
        self.assertIn("cpmg", results.comparison_data)


if __name__ == "__main__":
    unittest.main()
