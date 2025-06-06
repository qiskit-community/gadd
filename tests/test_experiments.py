"""Tests for experiments.py."""

import unittest
from unittest.mock import Mock, patch

from qiskit import QuantumCircuit
from qiskit.result import QuasiDistribution

from gadd.experiments import (
    create_bernstein_vazirani_circuit,
    create_ghz_circuit,
    run_bv_experiment,
    run_ghz_experiment,
)
from gadd import GADD, TrainingConfig
from gadd.strategies import DDStrategy

from .fixtures import MockBackend, MockSampler


class TestCircuitCreation(unittest.TestCase):
    """Test circuit creation functions."""

    def test_create_bernstein_vazirani_circuit(self):
        """Test BV circuit creation."""
        # Test with different bitstrings
        bitstring = "101"
        circuit = create_bernstein_vazirani_circuit(bitstring)

        # Check circuit properties
        self.assertIsInstance(circuit, QuantumCircuit)
        self.assertEqual(circuit.num_qubits, 4)  # n + 1 for ancilla
        self.assertEqual(circuit.num_clbits, 3)  # n classical bits

        # Check basic structure - should have Hadamards
        gate_names = [inst.operation.name for inst in circuit.data]
        self.assertIn("h", gate_names)
        self.assertIn("cx", gate_names)

        # Test with all-ones bitstring
        bitstring = "1111"
        circuit = create_bernstein_vazirani_circuit(bitstring)
        self.assertEqual(circuit.num_qubits, 5)
        self.assertEqual(circuit.num_clbits, 4)

        # Test with all-zeros bitstring
        bitstring = "0000"
        circuit = create_bernstein_vazirani_circuit(bitstring)
        self.assertEqual(circuit.num_qubits, 5)
        self.assertEqual(circuit.num_clbits, 4)

        # Count CX gates - should equal number of 1s in bitstring
        cx_count = sum(1 for inst in circuit.data if inst.operation.name == "cx")
        self.assertEqual(cx_count, 0)

    def test_create_bernstein_vazirani_circuit_structure(self):
        """Test detailed structure of BV circuit."""
        bitstring = "110"
        circuit = create_bernstein_vazirani_circuit(bitstring)

        # Extract gate sequence
        gates = [(inst.operation.name, inst.qubits) for inst in circuit.data]

        # First gate should be X on ancilla (last qubit)
        self.assertEqual(gates[0][0], "x")
        self.assertEqual(gates[0][1][0]._index, 3)  # Ancilla is qubit 3

        # Should have Hadamard on ancilla after X
        h_gates = [(i, g) for i, g in enumerate(gates) if g[0] == "h"]
        ancilla_h = [g for i, g in h_gates if g[1][0]._index == 3]
        self.assertGreater(len(ancilla_h), 0)

        # Check oracle implementation
        cx_gates = [(i, g) for i, g in enumerate(gates) if g[0] == "cx"]
        self.assertEqual(len(cx_gates), 2)  # Two 1s in "110"

        # Check measurements
        measure_gates = [g for g in gates if g[0] == "measure"]
        self.assertEqual(len(measure_gates), 3)  # Measure n qubits

    def test_create_ghz_circuit(self):
        """Test GHZ circuit creation."""
        # Test different sizes
        for n in [2, 3, 5, 10]:
            circuit = create_ghz_circuit(n)

            # Check properties
            self.assertIsInstance(circuit, QuantumCircuit)
            self.assertEqual(circuit.num_qubits, n)
            self.assertEqual(circuit.num_clbits, n)

            # Check structure
            gates = [inst.operation.name for inst in circuit.data]
            self.assertEqual(gates.count("h"), 1)  # One Hadamard
            self.assertEqual(gates.count("cx"), n - 1)  # n-1 CX gates
            self.assertEqual(gates.count("measure"), n)  # n measurements

    def test_create_ghz_circuit_structure(self):
        """Test detailed structure of GHZ circuit."""
        n = 4
        circuit = create_ghz_circuit(n)

        # Extract non-measurement gates
        gates = [
            (inst.operation.name, inst.qubits)
            for inst in circuit.data
            if inst.operation.name != "measure"
        ]

        # First gate should be H on qubit 0
        self.assertEqual(gates[0][0], "h")
        self.assertEqual(gates[0][1][0]._index, 0)

        # Check CX cascade
        cx_gates = [g for g in gates if g[0] == "cx"]
        for i, (_, qubits) in enumerate(cx_gates):
            control = qubits[0]._index
            target = qubits[1]._index
            self.assertEqual(control, i)
            self.assertEqual(target, i + 1)


class TestExperiments(unittest.TestCase):
    """Test experiment running functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.backend = MockBackend()
        self.config = TrainingConfig(pop_size=4, n_iterations=2)
        self.gadd = GADD(backend=self.backend, config=self.config, seed=42)

        # Mock sampler with good results
        def good_counts(circuit, shots):
            """Return counts that give high utility."""
            if "measure" in [inst.operation.name for inst in circuit.data]:
                n_qubits = circuit.num_clbits
                if n_qubits == 3:  # BV circuit
                    return {0b111: 0.9, 0b000: 0.1}  # "111" has high probability
                else:  # GHZ circuit
                    all_zeros = 0
                    all_ones = (1 << n_qubits) - 1
                    return {all_zeros: 0.45, all_ones: 0.45, 1: 0.1}
            return {0: 1.0}

        self.sampler = MockSampler(counts_function=good_counts)

    @patch("gadd.experiments.GADD.train")
    def test_run_bv_experiment_basic(self, mock_train):
        """Test basic BV experiment run."""
        # Mock training results
        mock_strategy = DDStrategy.from_single_sequence(Mock())
        mock_result = Mock(
            best_sequence=mock_strategy,
            best_score=0.95,
            iteration_data=[],
            benchmark_scores={},
            final_population=[],
            config=self.config,
            training_time=10.0,
        )
        mock_train.return_value = (mock_strategy, mock_result)

        # Run experiment
        result = run_bv_experiment(self.gadd, self.sampler, n=5)

        # Check call
        mock_train.assert_called_once()
        call_args = mock_train.call_args

        # Check circuit
        circuit = call_args[0][1]  # Second positional arg
        self.assertIsInstance(circuit, QuantumCircuit)
        self.assertEqual(circuit.num_qubits, 6)  # n + 1

        # Check result structure
        self.assertIn("best_strategy", result)
        self.assertIn("training_result", result)
        self.assertIn("circuit", result)
        self.assertIn("utility_function", result)

    @patch("gadd.experiments.GADD.train")
    def test_run_bv_experiment_custom_bitstring(self, mock_train):
        """Test BV experiment with custom bitstring."""
        mock_strategy = DDStrategy.from_single_sequence(Mock())
        mock_result = Mock()
        mock_train.return_value = (mock_strategy, mock_result)

        # Run with custom bitstring
        result = run_bv_experiment(self.gadd, self.sampler, n=4, bitstring="1010")

        # Verify circuit has correct size
        circuit = mock_train.call_args[0][1]
        self.assertEqual(circuit.num_qubits, 5)

    @patch("gadd.experiments.GADD.train")
    def test_run_bv_experiment_benchmark_strategies(self, mock_train):
        """Test BV experiment with custom benchmark strategies."""
        mock_strategy = DDStrategy.from_single_sequence(Mock())
        mock_result = Mock()
        mock_train.return_value = (mock_strategy, mock_result)

        # Run with custom benchmark strategies
        custom_seqs = ["xy4", "edd"]
        result = run_bv_experiment(
            self.gadd, self.sampler, n=3, benchmark_strategies=custom_seqs
        )

        # Check benchmark_strategies was passed
        kwargs = mock_train.call_args[1]
        self.assertEqual(kwargs["benchmark_strategies"], custom_seqs)

    @patch("gadd.experiments.GADD.train")
    def test_run_bv_experiment_kwargs(self, mock_train):
        """Test BV experiment forwards kwargs."""
        mock_strategy = DDStrategy.from_single_sequence(Mock())
        mock_result = Mock()
        mock_train.return_value = (mock_strategy, mock_result)

        # Run with additional kwargs
        result = run_bv_experiment(
            self.gadd, self.sampler, n=3, save_path="/tmp/test", custom_param=123
        )

        # Check kwargs were forwarded
        kwargs = mock_train.call_args[1]
        self.assertEqual(kwargs["save_path"], "/tmp/test")
        self.assertEqual(kwargs["custom_param"], 123)

    @patch("gadd.experiments.GADD.train")
    def test_run_ghz_experiment_basic(self, mock_train):
        """Test basic GHZ experiment run."""
        mock_strategy = DDStrategy.from_single_sequence(Mock())
        mock_result = Mock()
        mock_train.return_value = (mock_strategy, mock_result)

        # Run experiment
        result = run_ghz_experiment(self.gadd, self.sampler, n_qubits=5)

        # Check call
        mock_train.assert_called_once()
        call_args = mock_train.call_args

        # Check circuit
        circuit = call_args[0][1]
        self.assertIsInstance(circuit, QuantumCircuit)
        self.assertEqual(circuit.num_qubits, 5)

        # Check utility function is for GHZ
        utility_func = call_args[0][2]
        self.assertIsNotNone(utility_func)

    @patch("gadd.experiments.GADD.train")
    def test_run_ghz_experiment_large(self, mock_train):
        """Test GHZ experiment with large qubit count."""
        mock_strategy = DDStrategy.from_single_sequence(Mock())
        mock_result = Mock()
        mock_train.return_value = (mock_strategy, mock_result)

        # Run with 50 qubits (as in paper)
        result = run_ghz_experiment(self.gadd, self.sampler, n_qubits=50)

        # Verify circuit size
        circuit = mock_train.call_args[0][1]
        self.assertEqual(circuit.num_qubits, 50)

    def test_run_bv_experiment_integration(self):
        """Test full BV experiment integration."""
        # Use real training with minimal iterations
        self.gadd.config.n_iterations = 1
        self.gadd.config.pop_size = 2

        result = run_bv_experiment(self.gadd, self.sampler, n=3)

        # Check all components present
        self.assertIn("best_strategy", result)
        self.assertIn("training_result", result)
        self.assertIn("circuit", result)
        self.assertIn("utility_function", result)

        # Check types
        self.assertIsInstance(result["best_strategy"], DDStrategy)
        self.assertIsInstance(result["circuit"], QuantumCircuit)

        # Check training actually ran
        self.assertGreater(result["training_result"].training_time, 0)

    def test_run_ghz_experiment_integration(self):
        """Test full GHZ experiment integration."""
        # Use real training with minimal iterations
        self.gadd.config.n_iterations = 1
        self.gadd.config.pop_size = 2

        result = run_ghz_experiment(self.gadd, self.sampler, n_qubits=3)

        # Check all components present
        self.assertIn("best_strategy", result)
        self.assertIn("training_result", result)
        self.assertIn("circuit", result)
        self.assertIn("utility_function", result)

        # Check types
        self.assertIsInstance(result["best_strategy"], DDStrategy)
        self.assertIsInstance(result["circuit"], QuantumCircuit)

        # Check utility function
        utility = result["utility_function"]
        # Test it with mock GHZ-like counts
        ghz_counts = QuasiDistribution({"000": 0.5, "111": 0.5})
        score = utility.compute(ghz_counts)
        self.assertAlmostEqual(score, 1.0, places=2)


class TestExperimentEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_bv_circuit_empty_bitstring(self):
        """Test BV circuit with empty bitstring edge case."""
        # Should still create a valid circuit
        circuit = create_bernstein_vazirani_circuit("")
        self.assertEqual(circuit.num_qubits, 1)  # Just ancilla
        self.assertEqual(circuit.num_clbits, 0)

    def test_ghz_circuit_single_qubit(self):
        """Test GHZ circuit with single qubit."""
        circuit = create_ghz_circuit(1)
        self.assertEqual(circuit.num_qubits, 1)
        self.assertEqual(circuit.num_clbits, 1)

        # Should just be H and measure
        gates = [inst.operation.name for inst in circuit.data]
        self.assertIn("h", gates)
        self.assertNotIn("cx", gates)  # No CX for single qubit

    def test_bv_experiment_default_benchmark_strategies(self):
        """Test default benchmark strategies for BV experiment."""
        backend = MockBackend()
        gadd = GADD(backend=backend, config=TrainingConfig(n_iterations=1))
        sampler = MockSampler()

        with patch("gadd.experiments.GADD.train") as mock_train:
            mock_strategy = DDStrategy.from_single_sequence(Mock())
            mock_result = Mock()
            mock_train.return_value = (mock_strategy, mock_result)

            run_bv_experiment(gadd, sampler, n=3)

            # Check default benchmark strategies
            kwargs = mock_train.call_args[1]
            expected_seqs = [
                "cpmg",
                "cpmg_staggered",
                "xy4",
                "xy4_staggered",
                "edd",
                "edd_staggered",
                "urdd",
            ]
            self.assertEqual(kwargs["benchmark_strategies"], expected_seqs)

    def test_ghz_experiment_default_benchmark_strategies(self):
        """Test default benchmark strategies for GHZ experiment."""
        backend = MockBackend()
        gadd = GADD(backend=backend, config=TrainingConfig(n_iterations=1))
        sampler = MockSampler()

        with patch("gadd.experiments.GADD.train") as mock_train:
            mock_strategy = DDStrategy.from_single_sequence(Mock())
            mock_result = Mock()
            mock_train.return_value = (mock_strategy, mock_result)

            run_ghz_experiment(gadd, sampler, n_qubits=3)

            # Check default benchmark strategies (no URDD for GHZ)
            kwargs = mock_train.call_args[1]
            expected_seqs = [
                "cpmg",
                "cpmg_staggered",
                "xy4",
                "xy4_staggered",
                "edd",
                "edd_staggered",
            ]
            self.assertEqual(kwargs["benchmark_strategies"], expected_seqs)


if __name__ == "__main__":
    unittest.main()
