"""Integration tests."""

import unittest, os
from unittest.mock import patch

from qiskit import QuantumCircuit

from gadd import GADD, TrainingConfig, TrainingResult
from gadd.strategies import DDStrategy, ColorAssignment, DDSequence
from gadd.utility_functions import UtilityFactory

from .fixtures import MockBackend, MockSampler


class TestIntegration(unittest.TestCase):
    """Integration tests for complete GADD workflows."""

    def setUp(self):
        """Set up test fixtures."""
        self.backend = MockBackend(num_qubits=5)
        self.sampler = MockSampler()

    def test_basic_training_example(self):
        """Test basic training example from README."""
        # Create a simple quantum circuit
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.measure_all()

        # Simple configuration
        config = TrainingConfig(pop_size=4, n_iterations=2)
        gadd = GADD(backend=self.backend, config=config)

        # Use success probability utility
        utility_function = UtilityFactory.success_probability("000")

        best_strategy, result = gadd.train(
            sampler=self.sampler,
            training_circuit=circuit,
            utility_function=utility_function,
        )

        # Verify outputs
        self.assertIsInstance(best_strategy, DDStrategy)
        self.assertEqual(len(result.final_population), config.pop_size)
        self.assertEqual(len(result.iteration_data), config.n_iterations)
        self.assertEqual(result.config.pop_size, 4)

    def test_custom_parameters_example(self):
        """Test custom parameters example from README."""
        config = TrainingConfig(
            pop_size=8,
            sequence_length=4,
            n_iterations=3,
            mutation_probability=0.8,
            shots=1000,
            num_colors=2,
            dynamic_mutation=True,
        )

        gadd = GADD(backend=self.backend, config=config, seed=42)

        # Create test circuit
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        # Mock utility function that improves over iterations
        iteration_count = 0

        def improving_utility(normalized_counts):
            nonlocal iteration_count
            iteration_count += 1
            return 0.5 + (iteration_count * 0.1)

        utility_function = UtilityFactory.custom(improving_utility, "Test Utility")

        # Train with comparison sequences
        best_strategy, result = gadd.train(
            sampler=self.sampler,
            training_circuit=circuit,
            utility_function=utility_function,
            comparison_seqs=["xy4", "cpmg"],
        )

        # Verify training completed
        self.assertEqual(len(result.iteration_data), 3)
        self.assertIn("xy4", result.comparison_data)
        self.assertIn("cpmg", result.comparison_data)

        # Test applying DD to target circuit
        target_circuit = QuantumCircuit(2)
        target_circuit.h(0)
        target_circuit.delay(100, 0, unit="dt")
        target_circuit.cx(0, 1)

        circuit_with_dd = gadd.apply_dd(
            strategy=best_strategy, target_circuit=target_circuit
        )

        self.assertIsInstance(circuit_with_dd, QuantumCircuit)
        self.assertIn("DD", circuit_with_dd.name)

        # Test running the strategy
        run_result = gadd.run(
            strategy=best_strategy, target_circuit=target_circuit, sampler=self.sampler
        )

        self.assertIn("counts", run_result)
        self.assertIn("utility", run_result)
        self.assertIn("padded_circuit", run_result)

    def test_save_and_resume_training(self):
        """Test save and resume functionality."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(pop_size=4, n_iterations=4)
            gadd = GADD(backend=self.backend, config=config)

            circuit = QuantumCircuit(2)
            circuit.h(0)
            circuit.cx(0, 1)

            utility_function = UtilityFactory.success_probability("00")

            # Train for 2 iterations
            gadd.config.n_iterations = 2
            _, partial_result = gadd.train(
                sampler=self.sampler,
                training_circuit=circuit,
                utility_function=utility_function,
                save_path=tmpdir,
            )

            # Check checkpoint was saved
            checkpoint_file = os.path.join(tmpdir, "checkpoint_iter_2.json")
            self.assertTrue(os.path.exists(checkpoint_file))

            # Load checkpoint
            previous_state = gadd.load_training_state(checkpoint_file)
            self.assertEqual(previous_state.iteration, 2)

            # Resume training
            gadd.config.n_iterations = 4
            best_strategy, final_result = gadd.train(
                sampler=self.sampler,
                training_circuit=circuit,
                utility_function=utility_function,
                resume_from_state=previous_state,
            )

            # Should have completed 4 iterations total
            self.assertEqual(len(final_result.iteration_data), 4)
            self.assertEqual(final_result.iteration_data[0]["iteration"], 0)
            self.assertEqual(final_result.iteration_data[-1]["iteration"], 3)

    def test_custom_utility_function(self):
        """Test custom utility function example."""
        # Create GHZ circuit
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)

        def custom_fidelity(normalized_counts):
            """Custom fidelity-based utility function."""
            # Expected distribution for GHZ state
            expected_dist = {"000": 0.5, "111": 0.5}

            # Calculate 1-norm distance
            fidelity = 1.0
            for state in set(expected_dist.keys()) | set(normalized_counts.keys()):
                fidelity -= 0.5 * abs(
                    expected_dist.get(state, 0) - normalized_counts.get(state, 0)
                )

            return fidelity

        utility_function = UtilityFactory.custom(custom_fidelity, "GHZ Fidelity")

        config = TrainingConfig(pop_size=4, n_iterations=2)
        gadd = GADD(backend=self.backend, config=config)

        best_strategy, result = gadd.train(
            sampler=self.sampler,
            training_circuit=circuit,
            utility_function=utility_function,
        )

        # Verify custom utility was used
        self.assertIsInstance(best_strategy, DDStrategy)
        self.assertGreater(result.best_score, 0)

    def test_circuit_based_coloring(self):
        """Test coloring based on circuit structure."""
        # Create circuit with specific connectivity
        circuit = QuantumCircuit(5)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.cx(2, 3)
        circuit.cx(3, 4)

        # Create coloring from circuit
        color_assignment = ColorAssignment.from_circuit(circuit)

        config = TrainingConfig(pop_size=4, n_iterations=2)
        gadd = GADD(backend=self.backend, config=config, coloring=color_assignment)

        # Verify coloring was applied
        self.assertIsInstance(gadd.coloring, ColorAssignment)
        self.assertLessEqual(
            gadd.coloring.n_colors, 2
        )  # Linear chain needs at most 2 colors

        # Train and verify it works with custom coloring
        utility_function = UtilityFactory.success_probability("00000")

        best_strategy, result = gadd.train(
            sampler=self.sampler,
            training_circuit=circuit,
            utility_function=utility_function,
        )

        self.assertIsInstance(best_strategy, DDStrategy)

    def test_different_initialization_modes(self):
        """Test different population initialization modes."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        utility_function = UtilityFactory.success_probability("00")

        # Test random mode
        config_random = TrainingConfig(pop_size=4, n_iterations=2, mode="random")
        gadd_random = GADD(backend=self.backend, config=config_random, seed=42)

        _, result_random = gadd_random.train(
            sampler=self.sampler,
            training_circuit=circuit,
            utility_function=utility_function,
        )

        # Test uniform mode
        config_uniform = TrainingConfig(pop_size=4, n_iterations=2, mode="uniform")
        gadd_uniform = GADD(backend=self.backend, config=config_uniform, seed=42)

        _, result_uniform = gadd_uniform.train(
            sampler=self.sampler,
            training_circuit=circuit,
            utility_function=utility_function,
        )

        # Both should complete successfully
        self.assertEqual(len(result_random.final_population), 4)
        self.assertEqual(len(result_uniform.final_population), 4)

    @patch("matplotlib.pyplot.show")
    def test_plotting(self, mock_show):
        """Test plotting functionality."""
        # Create mock result

        result = TrainingResult(
            best_sequence=DDStrategy.from_single_sequence(
                DDSequence(["X", "Y", "X", "Y"])
            ),
            best_score=0.95,
            iteration_data=[
                {
                    "iteration": 0,
                    "best_score": 0.8,
                    "mean_score": 0.7,
                    "std_score": 0.1,
                },
                {
                    "iteration": 1,
                    "best_score": 0.9,
                    "mean_score": 0.8,
                    "std_score": 0.05,
                },
                {
                    "iteration": 2,
                    "best_score": 0.95,
                    "mean_score": 0.85,
                    "std_score": 0.05,
                },
            ],
            comparison_data={"xy4": 0.85, "cpmg": 0.8},
            final_population=["seq1", "seq2", "seq3", "seq4"],
            config=TrainingConfig(),
            training_time=123.45,
        )

        gadd = GADD(backend=self.backend)

        # Test plotting without save
        gadd.plot_training_progress(result)
        mock_show.assert_called_once()

        # Test plotting with save
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            gadd.plot_training_progress(result, save_path=tmp.name)
            self.assertTrue(os.path.exists(tmp.name))

    def test_custom_decoupling_group(self):
        """Test GADD with custom decoupling group."""
        from gadd.group_operations import DecouplingGroup

        # Create simple 4-element group
        custom_group = DecouplingGroup(
            elements={"I": 0, "X": 1, "Y": 2, "Z": 3},
            names={0: "I", 1: "X", 2: "Y", 3: "Z"},
            multiplication=[
                [0, 1, 2, 3],
                [1, 0, 3, 2],
                [2, 3, 0, 1],
                [3, 2, 1, 0],
            ],
            inverse_map={0: 0, 1: 1, 2: 2, 3: 3},
        )

        config = TrainingConfig(
            pop_size=4, n_iterations=2, decoupling_group=custom_group, sequence_length=4
        )

        gadd = GADD(backend=self.backend, config=config)

        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        utility_function = UtilityFactory.success_probability("00")

        best_strategy, result = gadd.train(
            sampler=self.sampler,
            training_circuit=circuit,
            utility_function=utility_function,
        )

        # Check that sequences use the custom group
        gates = gadd._decode_sequence(result.final_population[0])
        for gate in gates:
            self.assertIn(gate, ["I", "X", "Y", "Z"])


if __name__ == "__main__":
    unittest.main()
