"""Tests for circuit padding."""

import unittest
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Delay
from qiskit.circuit.library import XGate, YGate, ZGate, CXGate, RXGate, RYGate, RZGate
from qiskit.transpiler import InstructionDurations

from gadd.circuit_padding import (
    DDPulse,
    get_instruction_duration,
    apply_dd_strategy,
)
from gadd.strategies import DDSequence, DDStrategy
from .fixtures import MockBackend


class TestDDPulse(unittest.TestCase):
    """Test DDPulse class."""

    def test_init(self):
        """Test DDPulse initialization."""
        pulse = DDPulse("X", 0, 100.0)
        self.assertEqual(pulse.gate_name, "X")
        self.assertEqual(pulse.qubit, 0)
        self.assertEqual(pulse.time, 100.0)

    def test_to_gate_all_gates(self):
        """Test conversion for all supported gates."""
        # Test all supported gate types
        gate_tests = [
            ("I", "id"),
            ("Ip", "id"),
            ("Im", "id"),
            ("X", "rx"),
            ("Xp", "rx"),
            ("Xm", "rx"),
            ("Y", "ry"),
            ("Yp", "ry"),
            ("Ym", "ry"),
            ("Z", "rz"),
            ("Zp", "rz"),
            ("Zm", "rz"),
        ]

        for gate_name, expected_type in gate_tests:
            pulse = DDPulse(gate_name, 0, 0)
            gate = pulse.to_gate()
            self.assertIsNotNone(gate)
            # Check gate type
            if expected_type == "id":
                self.assertEqual(gate.name, "id")
            elif expected_type == "rx":
                self.assertIsInstance(gate, RXGate)
            elif expected_type == "ry":
                self.assertIsInstance(gate, RYGate)
            elif expected_type == "rz":
                self.assertIsInstance(gate, RZGate)

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

    def test_additional_gates(self):
        """Test duration for additional gate types."""
        # Test Y gate
        y_gate = YGate()
        duration = get_instruction_duration(y_gate, [0], unit="dt")
        self.assertEqual(duration, 160)

        # Test Z gate (virtual)
        z_gate = ZGate()
        duration = get_instruction_duration(z_gate, [0], unit="dt")
        self.assertEqual(duration, 0)

        # Test RX, RY, RZ gates
        rx_gate = RXGate(3.14)
        duration = get_instruction_duration(rx_gate, [0], unit="dt")
        self.assertEqual(duration, 160)

        ry_gate = RYGate(3.14)
        duration = get_instruction_duration(ry_gate, [0], unit="dt")
        self.assertEqual(duration, 160)

        rz_gate = RZGate(3.14)
        duration = get_instruction_duration(rz_gate, [0], unit="dt")
        self.assertEqual(duration, 0)  # RZ is virtual

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
        self.qc.delay(2000, 0, unit="dt")
        self.qc.x(0)
        self.qc.cx(0, 1)

        self.backend = MockBackend()

        # Test DD sequence
        self.dd_seq = DDSequence(["X", "Y", "X", "Y"])

        # Simple coloring
        self.coloring = {0: 0, 1: 0}

    def test_apply_dd_strategy(self):
        """Test applying full DD strategy."""
        strategy = DDStrategy.from_single_sequence(self.dd_seq)
        padded = apply_dd_strategy(self.qc, strategy, self.coloring, self.backend)

        # Check DD gates were added
        self.assertEqual(padded.data[0].operation, XGate())
        self.assertEqual(padded.data[2].operation, RXGate(np.pi))
        self.assertEqual(padded.data[4].operation, RYGate(np.pi))
        self.assertEqual(padded.data[6].operation, RXGate(np.pi))
        self.assertEqual(padded.data[8].operation, RYGate(np.pi))
        self.assertEqual(padded.data[10].operation, XGate())

        # Check naming
        self.assertIn("DD", padded.name)

    def test_apply_dd_strategy_with_backend(self):
        """Test applying DD strategy with backend."""

        backend = MockBackend()
        strategy = DDStrategy.from_single_sequence(self.dd_seq)

        # Test with backend-provided instruction durations
        padded = apply_dd_strategy(self.qc, strategy, self.coloring, backend=backend)
        self.assertIsInstance(padded, QuantumCircuit)

    def test_apply_dd_strategy_with_custom_durations(self):
        """Test applying DD strategy with custom instruction durations."""
        # Create custom durations
        durations = InstructionDurations(
            [
                ("x", 0, 200),
                ("y", 0, 200),
                ("cx", [0, 1], 1000),
            ]
        )

        strategy = DDStrategy.from_single_sequence(self.dd_seq)
        padded = apply_dd_strategy(self.qc, strategy, self.coloring, self.backend)
        self.assertIsInstance(padded, QuantumCircuit)

    def test_apply_dd_strategy_staggered(self):
        """Test applying staggered DD strategy."""
        # Create circuit with multiple colors
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(1, 2)

        # Three-color assignment
        coloring = {0: 0, 1: 1, 2: 2}

        # Create strategy with different sequences per color
        seq1 = DDSequence(["X", "X"])
        seq2 = DDSequence(["Y", "Y"])
        seq3 = DDSequence(["X", "Y", "X", "Y"])
        strategy = DDStrategy([seq1, seq2, seq3])

        padded = apply_dd_strategy(qc, strategy, coloring, self.backend, staggered=True)
        self.assertIsInstance(padded, QuantumCircuit)
        self.assertIn("staggered", padded.name)

    def test_apply_dd_strategy_missing_color(self):
        """Test handling missing color in strategy."""
        # Strategy only has sequence for color 0
        strategy = DDStrategy({0: self.dd_seq})

        # But coloring has colors 0 and 1
        coloring = {0: 0, 1: 1}

        # Should still work, just skip color 1
        padded = apply_dd_strategy(self.qc, strategy, coloring, self.backend)
        self.assertIsInstance(padded, QuantumCircuit)

    def test_apply_dd_strategy_odd_sequence(self):
        """Test handling odd-length sequences."""
        # Create odd-length sequence
        odd_seq = DDSequence(["X", "Y", "X"])
        strategy = DDStrategy.from_single_sequence(odd_seq)

        # Should print warning and skip
        import io
        import sys

        captured_output = io.StringIO()
        sys.stdout = captured_output

        padded = apply_dd_strategy(self.qc, strategy, self.coloring, self.backend)

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        self.assertIn("Warning", output)
        self.assertIn("odd number of gates", output)

    def test_apply_dd_strategy_identity_only(self):
        """Test sequence with only identity gates."""
        # Create sequence with only identity gates
        id_seq = DDSequence(["I", "I"])
        strategy = DDStrategy.from_single_sequence(id_seq)

        padded = apply_dd_strategy(self.qc, strategy, self.coloring, self.backend)
        self.assertIsInstance(padded, QuantumCircuit)

    def test_multi_color_strategy(self):
        """Test strategy with multiple colors."""
        # Create different sequences for different colors
        seq1 = DDSequence(["X", "X"])
        seq2 = DDSequence(["Y", "Y"])
        strategy = DDStrategy([seq1, seq2])

        # Color qubits differently
        coloring = {0: 0, 1: 1}

        padded = apply_dd_strategy(self.qc, strategy, coloring, self.backend)
        print(padded)

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
        padded = apply_dd_strategy(
            qc, strategy, coloring, self.backend, {"min_idle_duration": 64}
        )

        # Circuit should be essentially unchanged (just copied)
        self.assertEqual(padded.data, qc.data)


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
        padded = apply_dd_strategy(
            qc, DDStrategy([dd_seq]), {0: 0, 1: 1, 2: 2, 3: 3}, self.backend
        )

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
