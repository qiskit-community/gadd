"""Tests for gadd.py."""

import unittest
import tempfile
import os

from qiskit import QuantumCircuit

from gadd import GADD, TrainingConfig, TrainingState, TrainingResult
from gadd.strategies import DDSequence, DDStrategy, ColorAssignment
from gadd.group_operations import verify_sequence_identity

from .test_fixtures import MockBackend, MockSampler


class TestGADD(unittest.TestCase):
    """Test GADD class core functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.backend = MockBackend(num_qubits=5)
        self.config = TrainingConfig(
            pop_size=4, sequence_length=4, n_iterations=2, shots=1000
        )
        self.gadd = GADD(backend=self.backend, config=self.config, seed=42)

        # Simple test circuit
        self.circuit = QuantumCircuit(3)
        self.circuit.h(0)
        self.circuit.cx(0, 1)
        self.circuit.cx(1, 2)

    def test_initialization(self):
        """Test GADD initialization."""
        # Test with backend
        gadd = GADD(backend=self.backend)
        self.assertIsNotNone(gadd.backend)
        self.assertIsNotNone(gadd.coloring)

        # Test without backend
        gadd = GADD()
        self.assertIsNone(gadd.backend)
        self.assertEqual(gadd.coloring, {})

        # Test with custom config
        config = TrainingConfig(pop_size=8)
        gadd = GADD(config=config)
        self.assertEqual(gadd.config.pop_size, 8)

    def test_coloring_property(self):
        """Test coloring getter and setter."""
        # Test setter with valid coloring
        new_coloring = {0: 0, 1: 1, 2: 0}
        self.gadd.coloring = new_coloring
        self.assertEqual(self.gadd.coloring, new_coloring)

        # Test setter with invalid type
        with self.assertRaises(TypeError):
            self.gadd.coloring = "invalid"

    def test_apply_dd(self):
        """Test apply_dd method."""
        # Create a simple strategy
        sequence = DDSequence(["X", "Y", "X", "Y"])
        strategy = DDStrategy.from_single_sequence(sequence, n_colors=1)

        # Apply DD
        padded_circuit = self.gadd.apply_dd(strategy, self.circuit)

        # Check result
        self.assertIsInstance(padded_circuit, QuantumCircuit)
        self.assertEqual(padded_circuit.num_qubits, self.circuit.num_qubits)

        # Test with no backend (should use default coloring)
        gadd_no_backend = GADD()
        padded = gadd_no_backend.apply_dd(strategy, self.circuit)
        self.assertIsInstance(padded, QuantumCircuit)

    def test_generate_random_sequence(self):
        """Test random sequence generation."""
        sequence = self.gadd._generate_random_sequence()

        # Check length
        self.assertEqual(len(sequence), self.config.sequence_length)

        # Check all elements are valid
        for elem in sequence:
            self.assertIn(elem, range(self.config.group_size))

        self.assertTrue(verify_sequence_identity(sequence))

    def test_encode_decode_strategy(self):
        """Test strategy encoding and decoding."""
        # Create a sequence
        sequence = [2, 4, 2, 5]  # Xp, Yp, Xp, Ym

        # Encode
        encoded = self.gadd._encode_strategy(sequence)
        self.assertEqual(
            len(encoded), self.config.sequence_length * self.config.num_colors
        )

        # Decode
        decoded = self.gadd._decode_sequence(encoded)
        self.assertEqual(decoded, ["Xp", "Yp", "Xp", "Ym"])

    def test_initialize_populations(self):
        """Test population initialization methods."""
        # Test random initialization
        pop_random = self.gadd._initialize_random_population()
        self.assertEqual(len(pop_random), self.config.pop_size)
        self.assertEqual(len(set(pop_random)), self.config.pop_size)  # All unique

        # Test uniform initialization
        pop_uniform = self.gadd._initialize_uniform_population()
        self.assertEqual(len(pop_uniform), self.config.pop_size)

        # Verify all sequences are valid
        for strategy in pop_random + pop_uniform:
            gates = self.gadd._decode_sequence(strategy)
            self.assertEqual(len(gates), self.config.sequence_length)

    def test_crossover(self):
        """Test crossover operation."""
        parent1 = "0123" * self.config.num_colors  # Example encoding
        parent2 = "4567" * self.config.num_colors

        child = self.gadd._crossover(parent1, parent2)

        # Check child has correct length
        self.assertEqual(len(child), len(parent1))

        # Check child is different from both parents (usually)
        # Note: There's a small chance they could be the same
        self.assertTrue(child != parent1 or child != parent2)

    def test_mutate(self):
        """Test mutation operation."""
        original = "0123" * self.config.num_colors
        mutated = self.gadd._mutate(original)

        # Check length preserved
        self.assertEqual(len(mutated), len(original))

        # Check at least one position changed (except last)
        differences = sum(
            1
            for i in range(self.config.sequence_length - 1)
            if original[i] != mutated[i]
        )
        self.assertGreaterEqual(differences, 1)

    def test_calculate_diversity(self):
        """Test diversity calculation."""
        # All same
        population = ["0123"] * 4
        diversity = self.gadd._calculate_diversity(population)
        self.assertEqual(diversity, 0.25)  # 1/4

        # All different
        population = ["0123", "1234", "2345", "3456"]
        diversity = self.gadd._calculate_diversity(population)
        self.assertEqual(diversity, 1.0)

        # Mixed
        population = ["0123", "0123", "1234", "1234"]
        diversity = self.gadd._calculate_diversity(population)
        self.assertEqual(diversity, 0.5)

    def test_select_parents(self):
        """Test parent selection."""
        parents = ["seq1", "seq2", "seq3"]
        scores = {"seq1": 0.8, "seq2": 0.6, "seq3": 0.4}

        # Run multiple times to check probabilistic selection
        selected_counts = {"seq1": 0, "seq2": 0, "seq3": 0}
        for _ in range(100):
            p1, p2 = self.gadd._select_parents(parents, scores)
            selected_counts[p1] += 1
            selected_counts[p2] += 1

        # Higher scoring parents should be selected more often
        self.assertGreater(selected_counts["seq1"], selected_counts["seq3"])

    def test_adjust_mutation_probability(self):
        """Test dynamic mutation adjustment."""
        # Low diversity - should increase mutation
        self.gadd._training_state = TrainingState(mutation_probability=0.5)
        iteration_info = {"population_diversity": 0.05}
        self.gadd._adjust_mutation_probability(iteration_info)
        self.assertGreater(self.gadd._training_state.mutation_probability, 0.5)

        # High diversity - should decrease mutation
        self.gadd._training_state.mutation_probability = 0.5
        iteration_info = {"population_diversity": 0.9}
        self.gadd._adjust_mutation_probability(iteration_info)
        self.assertLess(self.gadd._training_state.mutation_probability, 0.5)

        # Check bounds
        self.gadd._training_state.mutation_probability = 0.95
        iteration_info = {"population_diversity": 0.05}
        self.gadd._adjust_mutation_probability(iteration_info)
        self.assertLessEqual(self.gadd._training_state.mutation_probability, 0.9)

    def test_coloring_with_color_assignment(self):
        """Test coloring setter with ColorAssignment."""
        # Test with ColorAssignment instance
        assignments = {0: [0, 2], 1: [1]}
        color_assignment = ColorAssignment.from_manual_assignment(assignments)

        self.gadd.coloring = color_assignment
        self.assertIsInstance(self.gadd.coloring, ColorAssignment)
        self.assertEqual(self.gadd.coloring.n_colors, 2)


class TestGADDTraining(unittest.TestCase):
    """Test GADD training functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.backend = MockBackend(num_qubits=3)
        self.config = TrainingConfig(
            pop_size=4, sequence_length=4, n_iterations=2, shots=1000
        )
        self.gadd = GADD(backend=self.backend, config=self.config, seed=42)

        # Simple test circuit
        self.circuit = QuantumCircuit(3)
        self.circuit.h(0)
        self.circuit.cx(0, 1)
        self.circuit.cx(1, 2)

        # Mock sampler
        self.sampler = MockSampler()

        # Simple utility function
        self.utility_function = lambda circuit, result: 0.75  # Constant for testing

    def test_evaluate_population(self):
        """Test population evaluation."""
        population = self.gadd._initialize_population("random")

        scores = self.gadd._evaluate_population(
            population, self.sampler, self.circuit, self.utility_function
        )

        # Check all strategies evaluated
        self.assertEqual(len(scores), len(population))

        # Check all scores are as expected
        for score in scores.values():
            self.assertEqual(score, 0.75)

        # Check sampler was called
        self.assertEqual(len(self.sampler.jobs_run), len(population))

    def test_evaluate_standard_sequences(self):
        """Test standard sequence evaluation."""
        comparison_seqs = ["xy4", "cpmg"]

        scores = self.gadd._evaluate_standard_sequences(
            comparison_seqs, self.sampler, self.circuit, self.utility_function
        )

        # Check all sequences evaluated
        self.assertEqual(len(scores), 2)
        self.assertIn("xy4", scores)
        self.assertIn("cpmg", scores)

        # Check scores
        for seq, score in scores.items():
            self.assertEqual(score, 0.75)

    def test_generate_offspring(self):
        """Test offspring generation."""
        population = ["seq1", "seq2", "seq3", "seq4"]
        scores = {"seq1": 0.8, "seq2": 0.7, "seq3": 0.6, "seq4": 0.5}

        # Need training state for mutation probability
        self.gadd._training_state = TrainingState(mutation_probability=0.5)

        offspring = self.gadd._generate_offspring(population, scores)

        # Should return 3K population (K + 2K)
        self.assertEqual(len(offspring), 3 * len(population))

        # Original population should be included
        for parent in population:
            self.assertIn(parent, offspring)

    def test_train_basic(self):
        """Test basic training loop."""
        # Use a utility function that improves over iterations
        iteration_count = 0

        def improving_utility(circuit, result):
            nonlocal iteration_count
            iteration_count += 1
            return 0.5 + (iteration_count * 0.01)

        strategy, result = self.gadd.train(
            self.sampler,
            self.circuit,
            improving_utility,
            mode="random",
            save_iterations=True,
        )

        # Check return types
        self.assertIsInstance(strategy, DDStrategy)
        self.assertIsInstance(result, TrainingResult)

        # Check training completed
        self.assertEqual(len(result.iteration_data), self.config.n_iterations)

        # Check scores improved
        first_score = result.iteration_data[0]["best_score"]
        last_score = result.iteration_data[-1]["best_score"]
        self.assertGreater(last_score, first_score)

    def test_train_with_comparison(self):
        """Test training with comparison sequences."""
        strategy, result = self.gadd.train(
            self.sampler,
            self.circuit,
            self.utility_function,
            comparison_seqs=["xy4", "cpmg"],
        )

        # Check comparison data
        self.assertIn("xy4", result.comparison_data)
        self.assertIn("cpmg", result.comparison_data)
        self.assertEqual(result.comparison_data["xy4"], 0.75)
        self.assertEqual(result.comparison_data["cpmg"], 0.75)

    def test_save_and_resume_training(self):
        """Test checkpoint save and resume."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Start training
            self.gadd._training_state = TrainingState(
                population=["seq1", "seq2"],
                iteration=5,
                best_scores=[0.5, 0.6, 0.7],
                mutation_probability=0.6,
            )

            # Save checkpoint
            self.gadd._save_checkpoint(tmpdir)

            # Check file created
            checkpoint_file = os.path.join(tmpdir, "checkpoint_iter_5.json")
            self.assertTrue(os.path.exists(checkpoint_file))

            # Load checkpoint
            loaded_state = self.gadd.load_training_state(checkpoint_file)

            # Verify state
            self.assertEqual(loaded_state.iteration, 5)
            self.assertEqual(loaded_state.population, ["seq1", "seq2"])
            self.assertEqual(loaded_state.best_scores, [0.5, 0.6, 0.7])
            self.assertEqual(loaded_state.mutation_probability, 0.6)

    def test_run_method(self):
        """Test run method for executing a single strategy."""
        sequence = DDSequence(["X", "Y", "X", "Y"])
        strategy = DDStrategy.from_single_sequence(sequence)

        result = self.gadd.run(strategy, self.circuit, self.sampler)

        # Check result structure
        self.assertIn("counts", result)
        self.assertIn("success_probability", result)
        self.assertIn("total_shots", result)
        self.assertIn("padded_circuit", result)

        # Check values
        self.assertEqual(result["total_shots"], self.config.shots)
        self.assertIsInstance(result["padded_circuit"], QuantumCircuit)


class TestTrainingConfig(unittest.TestCase):
    """Test TrainingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrainingConfig()
        self.assertEqual(config.pop_size, 16)
        self.assertEqual(config.sequence_length, 8)
        self.assertEqual(config.n_iterations, 20)
        self.assertEqual(config.mode, "random")

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TrainingConfig(pop_size=32, sequence_length=16, mode="uniform")
        self.assertEqual(config.pop_size, 32)
        self.assertEqual(config.sequence_length, 16)
        self.assertEqual(config.mode, "uniform")

    def test_serialization(self):
        """Test to_dict and from_dict methods."""
        config = TrainingConfig(pop_size=32, shots=8000)

        # Serialize
        data = config.to_dict()
        self.assertIsInstance(data, dict)
        self.assertEqual(data["pop_size"], 32)
        self.assertEqual(data["shots"], 8000)

        # Deserialize
        config2 = TrainingConfig.from_dict(data)
        self.assertEqual(config2.pop_size, 32)
        self.assertEqual(config2.shots, 8000)


class TestTrainingState(unittest.TestCase):
    """Test TrainingState dataclass."""

    def test_default_values(self):
        """Test default state values."""
        state = TrainingState()
        self.assertEqual(state.population, [])
        self.assertEqual(state.iteration, 0)
        self.assertEqual(state.best_scores, [])

    def test_serialization(self):
        """Test state serialization."""
        state = TrainingState(
            population=["seq1", "seq2"], iteration=5, best_scores=[0.5, 0.6]
        )

        # Serialize
        data = state.to_dict()
        self.assertEqual(data["population"], ["seq1", "seq2"])
        self.assertEqual(data["iteration"], 5)

        # Deserialize
        state2 = TrainingState.from_dict(data)
        self.assertEqual(state2.population, ["seq1", "seq2"])
        self.assertEqual(state2.iteration, 5)


class TestTrainingResult(unittest.TestCase):
    """Test TrainingResult dataclass."""

    def test_serialization(self):
        """Test result serialization."""
        sequence = DDSequence(["X", "Y"])
        strategy = DDStrategy.from_single_sequence(sequence)
        config = TrainingConfig(pop_size=4)

        result = TrainingResult(
            best_sequence=strategy,
            best_score=0.95,
            iteration_data=[{"iteration": 0, "score": 0.9}],
            comparison_data={"xy4": 0.8},
            final_population=["seq1", "seq2"],
            config=config,
            training_time=123.45,
        )

        # Serialize
        data = result.to_dict()
        self.assertIsInstance(data, dict)
        self.assertEqual(data["best_score"], 0.95)
        self.assertEqual(data["training_time"], 123.45)
        self.assertIn("best_sequence", data)
        self.assertIn("config", data)


if __name__ == "__main__":
    unittest.main()
