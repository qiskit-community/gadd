"""Tests for circuit padding."""

import unittest
from qiskit import QuantumCircuit
from qiskit.circuit import Delay
from qiskit.circuit.library import XGate, YGate, ZGate, CXGate

from gadd.circuit_padding import (
    DDPulse,
    get_instruction_duration,
    apply_dd_strategy,
)
from gadd.sequences import DDSequence, DDStrategy


class TestDDPulse(unittest.TestCase):
    """Test DDPulse class."""

    def test_init(self):
        """Test DDPulse initialization."""
        pulse = DDPulse("X", 0, 100.0)
        self.assertEqual(pulse.gate_name, "X")
        self.assertEqual(pulse.qubit, 0)
        self.assertEqual(pulse.time, 100.0)

    def test_to_gate_invalid(self):
        """Test conversion fails for invalid gate."""
        pulse = DDPulse("InvalidGate", 0, 0)
        with self.assertRaises(ValueError):
            pulse.to_gate()

    def test_repr(self):
        """Test string representation."""
        pulse = DDPulse("X", 2, 123.456)
        repr_str = repr(pulse)
        self.assertIn("X", repr_str)
        self.assertIn("q2", repr_str)
        self.assertIn("123.456", repr_str)


class TestInstructionDuration(unittest.TestCase):
    """Test instruction duration calculation."""

    def test_standard_gates(self):
        """Test duration for standard gates."""
        x_gate = XGate()
        duration = get_instruction_duration(x_gate, [0], unit="dt")
        self.assertEqual(duration, 160)  # Default X gate duration

        cx_gate = CXGate()
        duration = get_instruction_duration(cx_gate, [0, 1], unit="dt")
        self.assertEqual(duration, 800)  # Default CX gate duration

    def test_delay_instruction(self):
        """Test delay instruction duration."""
        delay = Delay(100, unit="dt")
        duration = get_instruction_duration(delay, [0], unit="dt")
        self.assertEqual(duration, 100)

        # Test unit conversion
        dt = 0.222e-9
        duration_s = get_instruction_duration(delay, [0], unit="s", dt=dt)
        self.assertAlmostEqual(duration_s, 100 * dt)

    def test_virtual_gates(self):
        """Test virtual gates have zero duration."""
        z_gate = ZGate()
        duration = get_instruction_duration(z_gate, [0], unit="dt")
        self.assertEqual(duration, 0)


class TestCircuitPadding(unittest.TestCase):
    """Test full circuit padding functionality."""

    def setUp(self):
        """Set up test circuits and sequences."""
        # Simple test circuit
        self.qc = QuantumCircuit(2)
        self.qc.x(0)
        self.qc.delay(200, 0, unit="dt")
        self.qc.x(0)
        self.qc.cx(0, 1)

        # Test DD sequence
        self.dd_seq = DDSequence(["X", "Y", "X", "Y"])

        # Simple coloring
        self.coloring = {0: 0, 1: 0}

    def test_apply_dd_strategy(self):
        """Test applying full DD strategy."""
        strategy = DDStrategy.from_single_sequence(self.dd_seq)
        padded = apply_dd_strategy(self.qc, strategy, self.coloring)

        # Check circuit properties
        self.assertIsInstance(padded, QuantumCircuit)
        self.assertEqual(padded.num_qubits, self.qc.num_qubits)

        # Check naming
        self.assertIn("DD", padded.name)

    def test_multi_color_strategy(self):
        """Test strategy with multiple colors."""
        # Create different sequences for different colors
        seq1 = DDSequence(["X", "X"])
        seq2 = DDSequence(["Y", "Y"])
        strategy = DDStrategy([seq1, seq2])

        # Color qubits differently
        coloring = {0: 0, 1: 1}

        padded = apply_dd_strategy(self.qc, strategy, coloring)

        # Both qubits should get different DD sequences
        self.assertIsInstance(padded, QuantumCircuit)

    def test_min_idle_duration(self):
        """Test minimum idle duration threshold."""
        # Create circuit with short idle period
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.delay(30, 0, unit="dt")  # Short delay
        qc.x(0)

        strategy = DDStrategy.from_single_sequence(self.dd_seq)
        coloring = {0: 0}

        # With default min_idle_duration, no DD should be inserted
        padded = apply_dd_strategy(qc, strategy, coloring, min_idle_duration=64)

        # Circuit should be essentially unchanged (just copied)
        # Note: In real implementation, we'd check that no DD gates were added
        self.assertIsInstance(padded, QuantumCircuit)


class TestIntegration(unittest.TestCase):
    """Integration tests for circuit padding."""

    def test_ghz_circuit_padding(self):
        """Test padding a GHZ state preparation circuit."""
        # Create GHZ circuit
        n_qubits = 4
        qc = QuantumCircuit(n_qubits)
        qc.h(0)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)

        # Apply DD
        dd_seq = DDSequence(["X", "Y", "X", "Y"])
        padded = apply_dd_strategy(qc, DDStrategy([dd_seq]), {0: 0, 1: 1, 2: 2, 3: 3})

        # Check result
        self.assertIsInstance(padded, QuantumCircuit)
        self.assertEqual(padded.num_qubits, n_qubits)

    def test_complex_circuit(self):
        """Test padding a more complex circuit."""
        qc = QuantumCircuit(3, 3)

        # Layer 1
        qc.h(0)
        qc.h(1)
        qc.h(2)

        # Layer 2 (creates idle times)
        qc.cx(0, 1)
        qc.delay(100, 2, unit="dt")

        # Layer 3
        qc.cx(1, 2)
        qc.x(0)

        # Measure
        qc.measure_all()

        # Create multi-color strategy
        xy4 = DDSequence(["X", "Y", "X", "Y"])
        cpmg = DDSequence(["X", "X"])
        strategy = DDStrategy([xy4, cpmg])

        # Color based on connectivity
        coloring = {0: 0, 1: 1, 2: 0}

        padded = apply_dd_strategy(qc, strategy, coloring)

        # Verify structure preserved
        self.assertEqual(padded.num_qubits, 3)
        self.assertEqual(padded.num_clbits, 3)

        # TODO - verify pulses placed correctly


if __name__ == "__main__":
    unittest.main()
