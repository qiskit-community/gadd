"""
Circuit padding utilities for applying dynamical decoupling sequences.

This module handles the insertion of DD pulses into quantum circuits
during idle periods, following the approach described in the GADD paper.
"""

from typing import List, Dict, Optional
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers import BackendV2 as Backend
from qiskit.circuit import Instruction, Gate, Delay
from qiskit.circuit.library import IGate, RXGate, RYGate, RZGate
from qiskit.transpiler import InstructionDurations, PassManager
from qiskit_ibm_runtime.transpiler.passes.scheduling import (
    ALAPScheduleAnalysis,
    PadDynamicalDecoupling,
)

from .strategies import DDStrategy


class DDPulse:
    """Represents a DD pulse with timing information."""

    def __init__(self, gate_name: str, qubit: int, time: float):
        self.gate_name = gate_name
        self.qubit = qubit
        self.time = time

    def to_gate(self) -> Gate:
        """Convert gate name to Qiskit gate object."""
        gate_map = {
            "I": IGate(),
            "Ip": IGate(),
            "Im": IGate(),  # Identity with phase - treated as identity in practice
            "X": RXGate(np.pi),
            "Xp": RXGate(np.pi),
            "Xm": RXGate(-np.pi),  # X gate with opposite rotation
            "Y": RYGate(np.pi),
            "Yp": RYGate(np.pi),
            "Ym": RYGate(-np.pi),  # Y gate with opposite rotation
            "Z": RZGate(np.pi),
            "Zp": RZGate(np.pi),
            "Zm": RZGate(-np.pi),  # Z gate with opposite rotation
        }

        gate = gate_map.get(self.gate_name)
        if gate is None:
            raise ValueError(f"Unknown gate name: {self.gate_name}")
        return gate

    def __repr__(self):
        return f"DDPulse({self.gate_name}, q{self.qubit}, t={self.time:.3f})"


def get_instruction_duration(
    instruction: Instruction,
    qubits: List[int],
    backend: Optional[Backend] = None,
    unit: str = "dt",
    dt: float = 0.222e-9,
) -> float:
    """
    Get duration of an instruction from backend or defaults.

    Args:
        instruction: The instruction to get duration for.
        qubits: Qubits the instruction acts on.
        backend: Backend to get durations from.
        unit: Time unit ('dt' or 's').
        dt: Duration of a single dt in seconds.

    Returns:
        Duration in specified units.
    """
    # Handle delay specially
    if isinstance(instruction, Delay):
        duration = instruction.duration
        if instruction.unit == "dt":
            return duration if unit == "dt" else duration * dt
        else:  # assumes 's'
            return duration / dt if unit == "dt" else duration

    # Try to get from backend first
    if backend is not None:
        try:
            if hasattr(backend, "instruction_durations"):
                durations = backend.instruction_durations
            else:
                durations = InstructionDurations.from_backend(backend)

            gate_name = instruction.name.lower()
            if len(qubits) == 1:
                duration_dt = durations.get(gate_name, qubits[0], "dt")
            else:
                duration_dt = durations.get(gate_name, qubits, "dt")

            if duration_dt is not None:
                return duration_dt if unit == "dt" else duration_dt * dt
        except:
            pass  # Fall back to defaults

    # Default durations in dt units
    default_durations = {
        "id": 0,  # Identity is virtual
        "x": 160,  # Single-qubit gate
        "y": 160,
        "z": 0,  # Z is virtual in most architectures
        "h": 160,
        "sx": 160,
        "rx": 160,
        "ry": 160,
        "rz": 0,  # RZ is virtual
        "u1": 0,  # U1 is virtual
        "u2": 160,
        "u3": 160,
        "cx": 800,  # Two-qubit gate
        "cz": 800,
        "ecr": 800,
        "measure": 4000,  # Measurement
        "reset": 1000,
    }

    gate_name = instruction.name.lower()
    duration_dt = default_durations.get(gate_name, 160)  # Default single-qubit duration

    if unit == "dt":
        return duration_dt
    else:
        return duration_dt * dt


def apply_dd_strategy(
    circuit: QuantumCircuit,
    strategy: DDStrategy,
    coloring: Dict[int, int],
    backend: Backend,
    staggered: bool = False,
    **dd_options: Optional[Dict],
) -> QuantumCircuit:
    """
    Apply a DD strategy to a quantum circuit using ``PadDynamicalDecoupling``.

    Args:
        circuit: Original quantum circuit.
        strategy: DD strategy containing sequences for each color.
        coloring: Mapping from qubit to color.
        backend: Backend to execute circuit on.
        staggered: Whether to apply CR-aware staggering for crosstalk suppression.
        dd_options: Additionl options to pass to ``PadDynamicalDecoupling``.

    Returns:
        Circuit with DD strategy applied.
    """
    if backend is not None:
        try:
            instruction_durations = InstructionDurations.from_backend(backend)
        except:  # pylint: disable=bare-except
            instruction_durations = InstructionDurations()
    else:
        instruction_durations = InstructionDurations()

    # Transpile the circuit itself first
    padded_circuit = transpile(circuit, backend)
    suffix = "_DD_staggered" if staggered else "_DD"
    padded_circuit.name = (
        f"{circuit.name}{suffix}" if circuit.name else f"circuit{suffix}"
    )
    print(padded_circuit)
    # Get unique colors and sort them
    unique_colors = sorted(set(coloring.values()))

    # Define staggered spacings if requested
    if staggered and len(unique_colors) >= 2:
        # Implement CR-aware staggering as described in the paper
        # Color 0: symmetric spacing [1/2n, 1/n, ..., 1/n, 1/2n]
        # Color 1: early spacing [1/n, 1/n, ..., 1/n, 0]
        # Color 2: late spacing [0, 1/n, ..., 1/n, 1/n]
        def get_staggered_spacing(
            color_idx: int, n_pulses: int
        ) -> Optional[List[float]]:
            if color_idx == 0 or n_pulses < 2:
                return None  # Use default symmetric

            spacing = [1.0 / n_pulses] * (n_pulses + 1)
            if color_idx == 1:
                # Early timing - combine first and last spacing
                spacing[0] = spacing[0] + spacing[-1]
                spacing[-1] = 0
            elif color_idx >= 2:
                # Late timing - combine first and last spacing
                spacing[0] = 0
                spacing[-1] = spacing[-1] + spacing[0]

            return spacing

    else:
        get_staggered_spacing = lambda color_idx, n_pulses: None

    # Apply DD for each color separately
    for color_idx, color in enumerate(unique_colors):
        # Get qubits for this color
        qubits_for_color = [q for q, c in coloring.items() if c == color]

        if not qubits_for_color:
            continue

        # Get DD sequence for this color
        try:
            dd_sequence = strategy.get_sequence(color)
        except KeyError:
            # No sequence for this color, skip
            continue

        # Convert sequence gates to Qiskit gates
        dd_gates = []
        for gate_name in dd_sequence.gates:
            if gate_name not in ["I", "Ip", "Im"]:  # Skip identity gates
                pulse = DDPulse(gate_name, 0, 0)  # Qubit and time don't matter here
                dd_gates.append(pulse.to_gate())

        if not dd_gates:
            continue

        # Get spacing for this color if staggered
        alt_spacings = (
            get_staggered_spacing(color_idx, len(dd_gates)) if staggered else None
        )

        # Create pass manager for this color's qubits
        pm = PassManager(
            [
                ALAPScheduleAnalysis(instruction_durations),
                PadDynamicalDecoupling(
                    durations=instruction_durations,
                    dd_sequences=dd_gates,
                    qubits=qubits_for_color,
                    # pulse_alignment=pulse_alignment,
                    insert_multiple_cycles=True,
                    coupling_map=None,
                    alt_spacings=alt_spacings,
                    skip_reset_qubits=False,
                    **dd_options,
                ),
            ]
        )

        # Apply the pass
        padded_circuit = pm.run(padded_circuit)

    return padded_circuit
