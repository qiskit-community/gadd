"""
Core GADD functionality.
"""

from typing import List, Tuple, Optional, Callable, Dict, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
import time
import os

import numpy as np
from numpy.random import BitGenerator, Generator, SeedSequence, default_rng

from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.transpiler import InstructionDurations
from qiskit_ibm_runtime import Sampler
import matplotlib.pyplot as plt

from .strategies import DDStrategy, DDSequence, StandardSequences, ColorAssignment
from .group_operations import (
    complete_sequence_to_identity,
    DecouplingGroup,
    DEFAULT_GROUP,
)
from .circuit_padding import apply_dd_strategy as _apply_dd_strategy
from .utility_functions import UtilityFunction, SuccessProbability


@dataclass
class TrainingConfig:
    """Configuration parameters for GADD training.

    This class encapsulates all hyperparameters and settings needed to configure the genetic algorithm
    optimization process for dynamical decoupling sequences. It provides sensible defaults based on the
    empirical findings from the GADD paper while allowing full customization of the training process.

    Args:
        pop_size: Size of the population (``K`` in the paper).
        sequence_length: Length of each DD sequence (``L`` in the paper).
        parent_fraction: Fraction of population to use as parents for reproduction.
        n_iterations: Number of GA iterations to run.
        mutation_probability: Initial probability of mutation.
        optimization_level: Qiskit transpilation optimization level.
        shots: Number of shots for quantum circuit execution.
        num_colors: Number of distinct sequences per strategy (``k`` in the paper).
        decoupling_group: The decoupling group to use from :class:`.DecouplingGroup`.
        mode: Mode for generating initial population (``uniform`` or ``random``).
        dynamic_mutation: Whether to dynamically adjust mutation probability.
        mutation_decay: Factor to adjust mutation probability.
    """

    pop_size: int = 16
    sequence_length: int = 8
    parent_fraction: float = 0.25
    n_iterations: int = 20
    mutation_probability: float = 0.75
    optimization_level: int = 1
    shots: int = 4000
    num_colors: int = 3
    decoupling_group: DecouplingGroup = field(default_factory=lambda: DEFAULT_GROUP)
    mode: str = "uniform"
    dynamic_mutation: bool = True
    mutation_decay: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        """Serializes training configuration to a dictionary for JSON export or checkpointing.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        data = asdict(self)
        # Convert DecouplingGroup to dict for serialization
        data["decoupling_group"] = {
            "elements": self.decoupling_group.elements,
            "names": self.decoupling_group.names,
            "multiplication": self.decoupling_group.multiplication,
            "inverse_map": self.decoupling_group.inverse_map,
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """Deserializes a dictionary object to a :class:`.TrainingConfig` object.

        Args:
            data: The serialized input dictionary.

        Returns:
            The deserialized :class:`.TrainingConfig` instance.
        """
        # Convert decoupling_group dict back to DecouplingGroup
        if "decoupling_group" in data and isinstance(data["decoupling_group"], dict):
            group_data = data["decoupling_group"]
            data["decoupling_group"] = DecouplingGroup(**group_data)
        return cls(**data)

    def __str__(self) -> str:
        """Return human-readable string representation."""
        lines = [
            "GADD Training Configuration:",
            f"  Population size: {self.pop_size}",
            f"  Sequence length: {self.sequence_length}",
            f"  Iterations: {self.n_iterations}",
            f"  Colors: {self.num_colors}",
            f"  Shots per evaluation: {self.shots}",
            f"  Mutation probability: {self.mutation_probability}",
            f"  Mode: {self.mode}",
        ]
        if self.dynamic_mutation:
            lines.append(f"  Dynamic mutation: enabled (decay={self.mutation_decay})")
        return "\n".join(lines)


@dataclass
class TrainingState:
    """State of GADD training that can be serialized and resumed.

    This class encapsulates the complete state of a genetic algorithm training session,
    enabling checkpointing and resumption of long-running optimization processes.
    """

    population: List[str] = field(default_factory=list)
    iteration: int = 0
    best_scores: List[float] = field(default_factory=list)
    best_sequences: List[str] = field(default_factory=list)
    mutation_probability: float = 0.75
    iteration_data: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Serializes training state to a dictionary for checkpointing.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingState":
        """Deserializes a dictionary to a :class:`.TrainingState` object.

        Args:
            data: The serialized input dictionary.

        Returns:
            The deserialized :class:`.TrainingState` instance.
        """
        return cls(**data)


@dataclass
class TrainingResult:
    """Results from GADD training.

    This class encapsulates all outputs and metrics from a completed genetic algorithm
    training session, including the best strategy found, performance data, and
    comparison against standard dynamical decoupling sequences.
    """

    best_sequence: DDStrategy
    best_score: float
    iteration_data: List[Dict[str, Any]]
    benchmark_scores: Dict[str, float]
    final_population: List[str]
    config: TrainingConfig
    training_time: float
    benchmark_history: Optional[Dict[str, List[float]]] = (
        None  # Only if tracked each iteration
    )

    def __post_init__(self):
        """Extract benchmark history from iteration data if not already provided."""
        if self.iteration_data and not self.benchmark_history:
            # Extract benchmark scores from iteration data
            benchmarks = {}
            for iteration in self.iteration_data:
                if "benchmark_scores" in iteration:
                    for name, score in iteration["benchmark_scores"].items():
                        if name not in benchmarks:
                            benchmarks[name] = []
                        benchmarks[name].append(score)
            self.benchmark_history = benchmarks

    def to_dict(self) -> Dict[str, Any]:
        """Serializes training result to a dictionary for export or analysis.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        result_dict = asdict(self)
        result_dict["best_sequence"] = self.best_sequence.to_dict()
        result_dict["config"] = self.config.to_dict()
        return result_dict

    def __str__(self) -> str:
        """Return human-readable string representation."""
        lines = [
            "GADD Training Results:",
            f"  Best score: {self.best_score:.4f}",
            f"  Training time: {self.training_time:.1f}s",
            f"  Iterations completed: {len(self.iteration_data)}",
        ]

        if self.iteration_data:
            first_score = self.iteration_data[0]["best_score"]
            improvement = self.best_score - first_score
            lines.append(f"  Score improvement: +{improvement:.4f}")

        if self.comparison_data:
            lines.append(
                f"  Compared against {len(self.comparison_data)} standard sequences"
            )
            best_standard = max(self.comparison_data.values())
            advantage = self.best_score - best_standard
            if advantage > 0:
                lines.append(f"  Advantage over best standard: +{advantage:.4f}")

        lines.append(f"\nBest sequence found:")
        lines.append(f"  {self.best_sequence}")

        return "\n".join(lines)


class GADD:
    """Genetic Algorithm for Dynamical Decoupling optimization.

    This class implements the core GADD algorithm for empirically optimizing dynamical
    decoupling sequences on quantum processors using genetic algorithms. It evolves
    populations of DD strategies to find the best-performing strategy as evaluated
    by the specified utility function.
    """

    def __init__(
        self,
        backend: Optional[Backend] = None,
        utility_function: Optional[UtilityFunction] = None,
        coloring: Optional[Union[Dict, ColorAssignment]] = None,
        seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
        config: Optional[TrainingConfig] = None,
    ):
        """Initialize the GADD optimizer with backend and configuration parameters.

        Args:
            backend: Quantum backend for circuit execution and device properties.
            utility_function: Function to evaluate circuit performance.
            coloring: Qubit coloring for multi-color DD strategies.
            seed: Random seed for reproducible results.
            config: Training configuration parameters.
        """
        self._backend = backend
        self._utility_function = utility_function
        self._seed = seed
        self._rng = default_rng(seed=seed)
        self._population = None
        self.config = config or TrainingConfig()

        # Set up coloring
        if coloring is None and backend is not None:
            # Default coloring from backend
            self._coloring = ColorAssignment(backend=backend)
        elif isinstance(coloring, dict):
            # Convert dict to ColorAssignment
            color_to_qubits = {}
            for qubit, color in coloring.items():
                if color not in color_to_qubits:
                    color_to_qubits[color] = []
                color_to_qubits[color].append(qubit)
            self._coloring = ColorAssignment.from_manual_assignment(color_to_qubits)
        elif isinstance(coloring, ColorAssignment):
            self._coloring = coloring
        else:
            self._coloring = None

        # Training state for resume capability
        self._training_state = None

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self._rng = default_rng(seed=seed)

    @property
    def backend(self):
        return self._backend

    @property
    def utility_function(self):
        return self._utility_function

    @property
    def coloring(self):
        return self._coloring

    @coloring.setter
    def coloring(self, coloring):
        if isinstance(coloring, dict):
            # Convert dict to ColorAssignment
            color_to_qubits = {}
            for qubit, color in coloring.items():
                if color not in color_to_qubits:
                    color_to_qubits[color] = []
                color_to_qubits[color].append(qubit)
            self._coloring = ColorAssignment.from_manual_assignment(color_to_qubits)
        elif isinstance(coloring, ColorAssignment):
            self._coloring = coloring
        elif coloring is None:
            self._coloring = None
        else:
            raise TypeError(
                "Coloring must be a dictionary, ColorAssignment instance, or None"
            )

    def apply_strategy(
        self,
        strategy: DDStrategy,
        target_circuit: QuantumCircuit,
        backend: Optional[Backend] = None,
        staggered: bool = False,
    ) -> QuantumCircuit:
        """
        Apply a DD strategy to a target circuit.

        This is a convenience method that handles the circuit padding
        with the appropriate coloring for the backend.

        Args:
            strategy: DD strategy to apply.
            target_circuit: Circuit to apply DD to.
            backend: Backend for coloring (uses self.backend if None).
            staggered: Whether to apply CR-aware staggering for crosstalk suppression.

        Returns:
            Circuit with DD sequences applied.
        """
        # Use provided backend or fall back to instance backend
        backend = backend or self._backend

        # Get coloring for the circuit
        if self._coloring is not None:
            coloring_dict = self._coloring.to_dict()
        elif backend:
            color_assignment = ColorAssignment(backend=backend)
            coloring_dict = color_assignment.to_dict()
        else:
            # Fallback: all qubits same color
            coloring_dict = {i: 0 for i in range(target_circuit.num_qubits)}

        # Apply the DD strategy
        return _apply_dd_strategy(
            target_circuit,
            strategy,
            coloring_dict,
            backend=backend,
            staggered=staggered,
        )

    def train(
        self,
        sampler: Sampler,
        training_circuit: QuantumCircuit,
        utility_function: Optional[Union[Callable, UtilityFunction]] = None,
        mode: Optional[str] = None,
        save_iterations: bool = True,
        benchmark_strategies: Optional[Union[List[str], List[DDStrategy]]] = None,
        evaluate_benchmarks_each_iteration: bool = False,
        resume_from_state: Optional[TrainingState] = None,
        save_path: Optional[str] = None,
    ) -> Tuple[DDStrategy, TrainingResult]:
        """Train DD sequences using genetic algorithm optimization.

        This method executes the core GADD algorithm, evolving a population of DD
        strategies over multiple generations to optimize performance on the training
        circuit. The process includes population initialization, fitness evaluation,
        selection, crossover, mutation, and optional benchmarking against standard
        DD sequences.

        Args:
            sampler: :class:`qiskit_ibm_runtime.SamplerV2` for circuit execution.
            training_circuit: Quantum circuit to optimize DD sequences for.
            utility_function: Function to evaluate circuit performance, either a
                callable(circuit, result) -> float or :class:`.UtilityFunction`.
            mode: Population initialization mode (``random`` or ``uniform``).
            save_iterations: Whether to save iteration data for analysis.
            benchmark_strategies: DD strategies to compare against, either standard
                sequence names or :class:`.DDStrategy` objects.
            evaluate_benchmarks_each_iteration: Whether to evaluate benchmarks at
                each iteration or only at the end.
            resume_from_state: Previous :class:`.TrainingState` to resume from.
            save_path: Directory path to save training checkpoints.

        Returns:
            Tuple of the best :class:`.DDStrategy` and :class:`.TrainingResult`.
        """
        start_time = time.time()

        # Normalize utility function
        if utility_function is None:
            # Default to success probability for all-zero state
            utility_function = SuccessProbability("0" * training_circuit.num_qubits)

        # If it's a UtilityFunction instance, extract the compute method
        if hasattr(utility_function, "compute"):
            utility_func = lambda circuit, result: utility_function.compute(
                result.quasi_dists[0]
                if hasattr(result, "quasi_dists")
                else result.get_counts()
            )
        else:
            utility_func = utility_function

        # Set mode from config if not provided
        if mode is None:
            mode = self.config.mode

        if mode not in ["random", "uniform"]:
            raise ValueError("Mode must be one of 'random' or 'uniform'.")

        # Initialize or resume training state
        if resume_from_state is not None:
            self._training_state = resume_from_state
            print(f"Resuming training from iteration {self._training_state.iteration}")
        else:
            # Fresh training - initialize population
            self._training_state = TrainingState()
            self._training_state.population = self._initialize_population(mode)
            self._training_state.mutation_probability = self.config.mutation_probability

        # Prepare benchmark strategies
        benchmark_strategy_map = {}  # name -> (DDStrategy, staggered)
        if benchmark_strategies:
            standard_seqs = StandardSequences()
            for i, benchmark in enumerate(benchmark_strategies):
                if isinstance(benchmark, str):
                    # Convert string name to DDStrategy
                    name = benchmark
                    sequence = standard_seqs.get(benchmark)
                    strategy = DDStrategy.from_single_sequence(sequence)
                    staggered = standard_seqs.is_staggered(benchmark)
                    benchmark_strategy_map[name] = (strategy, staggered)
                elif isinstance(benchmark, DDStrategy):
                    # Use DDStrategy directly with generic name
                    name = f"custom_strategy_{i}"
                    benchmark_strategy_map[name] = (benchmark, False)
                else:
                    raise TypeError(
                        f"benchmark_strategies must contain strings or DDStrategy objects, got {type(benchmark)}"
                    )

        benchmark_history = {} if evaluate_benchmarks_each_iteration else None

        # Training loop
        for iteration in range(
            self._training_state.iteration, self.config.n_iterations
        ):
            print(f"GA Iteration {iteration + 1}/{self.config.n_iterations}")

            # Evaluate current population
            scores = self._evaluate_population(
                self._training_state.population, sampler, training_circuit, utility_func
            )

            # Evaluate benchmark strategies if provided
            iteration_benchmark_scores = {}
            if evaluate_benchmarks_each_iteration and benchmark_strategy_map:
                print("  Evaluating benchmark strategies...")
                for name, (strategy, staggered) in benchmark_strategy_map.items():
                    padded_circuit = self.apply_strategy(
                        strategy, training_circuit, staggered=staggered
                    )
                    job = sampler.run(padded_circuit, shots=self.config.shots)
                    result = job.result()

                    # Extract result
                    if hasattr(result, "quasi_dists"):
                        circuit_result = type(
                            "Result",
                            (),
                            {
                                "quasi_dists": [result.quasi_dists[0]],
                                "metadata": (
                                    result.metadata[0]
                                    if hasattr(result, "metadata")
                                    else {}
                                ),
                            },
                        )
                    else:
                        circuit_result = result

                    score = utility_func(padded_circuit, circuit_result)
                    iteration_benchmark_scores[name] = score

                    # Track history
                    if name not in benchmark_history:
                        benchmark_history[name] = []
                    benchmark_history[name].append(score)

                    print(f"    {name}: {score:.4f}")

            # Track best performance
            best_score = max(scores.values())
            best_sequence = max(scores, key=scores.get)

            self._training_state.best_scores.append(best_score)
            self._training_state.best_sequences.append(best_sequence)

            # Save iteration data
            if save_iterations:
                iteration_info = {
                    "iteration": iteration,
                    "best_score": best_score,
                    "mean_score": np.mean(list(scores.values())),
                    "std_score": np.std(list(scores.values())),
                    "mutation_probability": self._training_state.mutation_probability,
                    "population_diversity": self._calculate_diversity(
                        self._training_state.population
                    ),
                }

                # Only add benchmark scores if we're tracking them each iteration
                if evaluate_benchmarks_each_iteration and iteration_benchmark_scores:
                    iteration_info["benchmark_scores"] = iteration_benchmark_scores

            self._training_state.iteration_data.append(iteration_info)

            # Dynamic mutation adjustment
            if self.config.dynamic_mutation and iteration > 0:
                self._adjust_mutation_probability(iteration_info)

            # Generate next population (except for last iteration)
            if iteration < self.config.n_iterations - 1:
                # Generate offspring from current population
                new_population = self._generate_offspring(
                    self._training_state.population, scores
                )

                # Evaluate the combined population (parents + offspring)
                all_scores = self._evaluate_population(
                    new_population, sampler, training_circuit, utility_func
                )

                # Select top K for next generation
                sorted_combined = sorted(
                    new_population, key=lambda x: all_scores[x], reverse=True
                )
                self._training_state.population = sorted_combined[
                    : self.config.pop_size
                ]

            # Update iteration counter
            self._training_state.iteration = iteration + 1

            # Save checkpoint if path provided
            if save_path:
                self._save_checkpoint(save_path)

            print(
                f"Best score: {best_score:.4f}, Mean: {np.mean(list(scores.values())):.4f}"
            )

        # Get final best sequence
        final_scores = self._evaluate_population(
            self._training_state.population, sampler, training_circuit, utility_func
        )
        best_sequence = max(final_scores, key=final_scores.get)
        best_score = final_scores[best_sequence]

        # Calculate training time
        training_time = time.time() - start_time

        # Evaluate benchmarks at the end if not done during training
        final_benchmark_scores = {}
        if benchmark_strategy_map:
            if not evaluate_benchmarks_each_iteration:
                print("\nEvaluating benchmark strategies...")
                for name, (strategy, staggered) in benchmark_strategy_map.items():
                    padded_circuit = self.apply_strategy(
                        strategy, training_circuit, staggered=staggered
                    )
                    job = sampler.run(padded_circuit, shots=self.config.shots)
                    result = job.result()

                    if hasattr(result, "quasi_dists"):
                        circuit_result = type(
                            "Result",
                            (),
                            {
                                "quasi_dists": [result.quasi_dists[0]],
                                "metadata": (
                                    result.metadata[0]
                                    if hasattr(result, "metadata")
                                    else {}
                                ),
                            },
                        )
                    else:
                        circuit_result = result

                    score = utility_func(padded_circuit, circuit_result)
                    final_benchmark_scores[name] = score
                    print(f"  {name}: {score:.4f}")
            else:
                # Get final scores from history
                for name, scores in benchmark_history.items():
                    if scores:
                        final_benchmark_scores[name] = scores[-1]

        # Create result
        result = TrainingResult(
            best_sequence=self._create_strategy_from_string(best_sequence),
            best_score=best_score,
            iteration_data=self._training_state.iteration_data,
            benchmark_scores=final_benchmark_scores,
            final_population=self._training_state.population.copy(),
            config=self.config,
            training_time=training_time,
        )

        return self._create_strategy_from_string(best_sequence), result

    def _initialize_population(self, mode: str = "random") -> List[str]:
        """Initialize population of DD sequences."""
        if mode == "uniform":
            return self._initialize_uniform_population()
        else:
            return self._initialize_random_population()

    def _initialize_uniform_population(self) -> List[str]:
        """Initialize uniform population as described in the paper."""
        population = []
        group_size = self.config.decoupling_group.size

        # Calculate repetitions per element
        reps_per_element = self.config.pop_size // group_size
        if reps_per_element == 0:
            reps_per_element = 1

        # Generate base patterns with cyclic shifts
        base_patterns = []
        for shift in range(min(self.config.sequence_length, self.config.pop_size)):
            pattern = []
            for pos in range(self.config.sequence_length - 1):
                element = (pos + shift) % group_size
                pattern.append(element)
            base_patterns.append(pattern)

        # Generate variants for each base pattern
        for pattern in base_patterns:
            if len(population) >= self.config.pop_size:
                break

            sequence = pattern.copy()

            # Calculate last element to make sequence multiply to identity
            last_element = complete_sequence_to_identity(
                sequence, self.config.decoupling_group
            )
            sequence.append(last_element)

            # Create strategy for all colors
            strategy = self._encode_strategy(sequence)
            population.append(strategy)

        # Fill remaining slots with random sequences
        while len(population) < self.config.pop_size:
            sequence = self._generate_random_sequence()
            strategy = self._encode_strategy(sequence)
            if strategy not in population:
                population.append(strategy)

        return population

    def _initialize_random_population(self) -> List[str]:
        """Initialize random population."""
        population = []
        while len(population) < self.config.pop_size:
            sequence = self._generate_random_sequence()
            strategy = self._encode_strategy(sequence)
            if strategy not in population:
                population.append(strategy)
        return population

    def _generate_random_sequence(self) -> List[int]:
        """Generate a random DD sequence that multiplies to identity."""
        sequence = []
        for _ in range(self.config.sequence_length - 1):
            sequence.append(self._rng.integers(0, self.config.decoupling_group.size))

        # Calculate last element to ensure multiplication to identity
        last_element = complete_sequence_to_identity(
            sequence, self.config.decoupling_group
        )
        sequence.append(last_element)

        return sequence

    def _encode_strategy(self, sequence: List[int]) -> str:
        """Encode a sequence as a strategy string."""
        # For multi-color strategies, repeat sequence for each color
        strategy_parts = []
        for _ in range(self.config.num_colors):
            strategy_parts.append("".join(str(x) for x in sequence))
        return "".join(strategy_parts)

    def _decode_sequence(self, strategy: str) -> List[str]:
        """Decode strategy string back to gate sequence."""
        # Extract first color sequence (they should all be the same for now)
        seq_len = self.config.sequence_length
        first_sequence = strategy[:seq_len]

        gates = []
        for char in first_sequence:
            gates.append(self.config.decoupling_group.element_name(int(char)))

        return gates

    def _create_strategy_from_string(self, strategy: str) -> DDStrategy:
        """Create DDStrategy object from encoded string."""
        gates = self._decode_sequence(strategy)
        dd_sequence = DDSequence(gates)

        # Create strategy with same sequence for all colors
        sequences = []
        for _ in range(self.config.num_colors):
            sequences.append(dd_sequence.copy())

        return DDStrategy(sequences)

    def _evaluate_population(
        self,
        population: List[str],
        sampler: Sampler,
        circuit: QuantumCircuit,
        utility_function: Callable,
    ) -> Dict[str, float]:
        """Evaluate fitness of all sequences in population."""
        scores = {}

        # Batch circuits for efficiency
        circuits_to_run = []
        strategy_map = {}

        for i, strategy_str in enumerate(population):
            # Create strategy object
            dd_strategy = self._create_strategy_from_string(strategy_str)

            # Apply DD to circuit
            padded_circuit = self.apply_strategy(dd_strategy, circuit)
            circuits_to_run.append(padded_circuit)
            strategy_map[i] = strategy_str

        # Run all circuits in a single job for efficiency
        print(f"  Evaluating {len(circuits_to_run)} circuits...", end="\r")
        job = sampler.run(circuits_to_run, shots=self.config.shots)
        results = job.result()

        # Calculate utilities
        for i, strategy_str in strategy_map.items():
            # Extract result for this circuit
            if hasattr(results, "quasi_dists"):
                circuit_result = type(
                    "Result",
                    (),
                    {
                        "quasi_dists": [results.quasi_dists[i]],
                        "metadata": (
                            results.metadata[i] if hasattr(results, "metadata") else {}
                        ),
                    },
                )
            else:
                circuit_result = results[i]

            # Calculate utility
            scores[strategy_str] = utility_function(circuits_to_run[i], circuit_result)

        print(
            f"  Evaluated {len(population)} strategies. Best: {max(scores.values()):.4f}"
        )
        return scores

    def _generate_offspring(
        self, population: List[str], scores: Dict[str, float]
    ) -> List[str]:
        """Generate offspring population following the paper's GA approach.

        1. Keep original K parents
        2. Generate 2K offspring through reproduction
        3. Return combined 3K population for evaluation and selection
        """
        K = len(population)

        # Sort population and select parents
        sorted_pop = sorted(population, key=lambda x: scores[x], reverse=True)
        n_parents = max(2, int(self.config.parent_fraction * K))
        parents = sorted_pop[:n_parents]

        # Generate 2K offspring
        offspring = []
        for _ in range(2 * K):
            # Select two parents for reproduction
            p1, p2 = self._select_parents(parents, scores)
            child = self._crossover(p1, p2)

            # Apply mutation with probability
            if self._rng.random() < self._training_state.mutation_probability:
                child = self._mutate(child)

            offspring.append(child)

        # Return combined population (K parents + 2K offspring = 3K total)
        return population + offspring

    def _select_parents(
        self, parents: List[str], scores: Dict[str, float]
    ) -> Tuple[str, str]:
        """Select two parents using fitness-proportional selection."""
        # Convert scores to probabilities
        parent_scores = [scores[p] for p in parents]
        min_score = min(parent_scores)

        # Shift scores to be positive and add small epsilon
        adjusted_scores = [s - min_score + 0.001 for s in parent_scores]
        total_score = sum(adjusted_scores)
        probabilities = [s / total_score for s in adjusted_scores]

        # Select two parents
        indices = self._rng.choice(len(parents), size=2, replace=False, p=probabilities)
        return parents[indices[0]], parents[indices[1]]

    def _crossover(self, parent1: str, parent2: str) -> str:
        """Perform crossover between two strategy strings."""
        seq_len = self.config.sequence_length

        # Work with first color sequence
        seq1 = parent1[:seq_len]
        seq2 = parent2[:seq_len]

        # Random crossover point
        point = self._rng.integers(1, seq_len)

        # Create child sequence
        child_seq = seq1[:point] + seq2[point:-1]  # Exclude last element

        # Calculate last element to maintain group constraint
        child_indices = [int(c) for c in child_seq]
        last_element = complete_sequence_to_identity(
            child_indices, self.config.decoupling_group
        )
        child_seq += str(last_element)

        # Replicate for all colors
        return child_seq * self.config.num_colors

    def _mutate(self, strategy: str) -> str:
        """Mutate a strategy string."""
        seq_len = self.config.sequence_length
        seq = list(strategy[:seq_len])

        # Mutate random position (except last)
        if len(seq) > 1:
            idx = self._rng.integers(0, len(seq) - 1)
            seq[idx] = str(self._rng.integers(0, self.config.decoupling_group.size))

            # Recalculate last element
            seq_indices = [int(c) for c in seq[:-1]]
            last_element = complete_sequence_to_identity(
                seq_indices, self.config.decoupling_group
            )
            seq[-1] = str(last_element)

        # Replicate for all colors
        return "".join(seq) * self.config.num_colors

    def _adjust_mutation_probability(self, iteration_info: Dict[str, Any]):
        """Dynamically adjust mutation probability based on population diversity."""
        diversity = iteration_info.get("population_diversity", 0.5)

        if diversity < 0.1:  # Low diversity - increase mutation
            self._training_state.mutation_probability = min(
                0.9,
                self._training_state.mutation_probability + self.config.mutation_decay,
            )
        elif diversity > 0.8:  # High diversity - decrease mutation
            self._training_state.mutation_probability = max(
                0.1,
                self._training_state.mutation_probability - self.config.mutation_decay,
            )

    def _calculate_diversity(self, population: List[str]) -> float:
        """Calculate population diversity as fraction of unique individuals."""
        return len(set(population)) / len(population)

    def _evaluate_standard_sequences(
        self,
        sequences: List[str],
        sampler: Sampler,
        circuit: QuantumCircuit,
        utility_function: Callable,
    ) -> Dict[str, float]:
        """Evaluate standard DD sequences for comparison."""
        results = {}
        std_sequences = StandardSequences()

        for seq_name in sequences:
            print(f"Evaluating standard sequence: {seq_name}")
            try:
                sequence = std_sequences.get(seq_name.lower())
                strategy = DDStrategy.from_single_sequence(
                    sequence, n_colors=self.config.num_colors
                )

                # Check if this should be staggered
                staggered = std_sequences.is_staggered(seq_name.lower())

                padded_circuit = self.apply_strategy(
                    strategy, circuit, staggered=staggered
                )

                job = sampler.run(padded_circuit, shots=self.config.shots)
                result = job.result()
                results[seq_name] = utility_function(padded_circuit, result)
            except Exception as e:
                print(f"Warning: Could not evaluate {seq_name}: {e}")
                results[seq_name] = 0.0

        return results

    def _save_checkpoint(self, save_path: str):
        """Save training checkpoint."""
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        checkpoint = {
            "state": self._training_state.to_dict(),
            "config": self.config.to_dict(),
            "timestamp": datetime.now().isoformat(),
        }

        filename = os.path.join(
            save_path, f"checkpoint_iter_{self._training_state.iteration}.json"
        )
        with open(filename, "w") as f:
            json.dump(checkpoint, f, indent=2)

        print(f"Checkpoint saved to {filename}")

    def load_training_state(self, checkpoint_path: str) -> TrainingState:
        """Load training state from checkpoint file."""
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)

        return TrainingState.from_dict(checkpoint["state"])

    def plot_training_progress(
        self, results: TrainingResult, save_path: Optional[str] = None
    ):
        """Plot training progression and comparison data."""
        # Determine number of subplots needed
        n_plots = 2 if results.comparison_data else 1
        if results.benchmark_history:
            n_plots = 3

        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        # Plot training progression
        ax1 = axes[0]
        iterations = [d["iteration"] for d in results.iteration_data]
        best_scores = [d["best_score"] for d in results.iteration_data]
        mean_scores = [d["mean_score"] for d in results.iteration_data]

        ax1.plot(iterations, best_scores, "-o", label="Best Score", linewidth=2)
        ax1.plot(iterations, mean_scores, "-s", label="Mean Score", alpha=0.7)

        # Add benchmark lines if available
        if results.benchmark_history:
            for name, scores in results.benchmark_history.items():
                ax1.plot(
                    iterations[: len(scores)],
                    scores,
                    "--",
                    label=f"{name} (benchmark)",
                    alpha=0.6,
                )

        ax1.fill_between(
            iterations,
            [d["mean_score"] - d["std_score"] for d in results.iteration_data],
            [d["mean_score"] + d["std_score"] for d in results.iteration_data],
            alpha=0.3,
            label="±1 STD",
        )
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Utility Score")
        ax1.set_title("Training Progression")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot comparison data
        if results.comparison_data and n_plots >= 2:
            ax2 = axes[1]
            names = list(results.comparison_data.keys())
            values = list(results.comparison_data.values())

            # Add GADD result for comparison
            names.append("GADD")
            values.append(results.best_score)

            # Create bar plot
            bars = ax2.bar(names, values)

            # Highlight GADD bar
            bars[-1].set_color("green")
            bars[-1].set_alpha(0.8)

            ax2.set_xlabel("DD Sequence")
            ax2.set_ylabel("Utility Score")
            ax2.set_title("Sequence Comparison")
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

            # Add value labels on bars
            for bar, val in zip(bars, values):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                )

        # Plot benchmark history separately if we have 3 plots
        if results.benchmark_history and n_plots == 3:
            ax3 = axes[2]
            for name, scores in results.benchmark_history.items():
                ax3.plot(iterations[: len(scores)], scores, "-o", label=name)

            ax3.set_xlabel("Iteration")
            ax3.set_ylabel("Utility Score")
            ax3.set_title("Benchmark Performance")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def evaluate(
        self,
        strategy: DDStrategy,
        target_circuit: QuantumCircuit,
        sampler: Sampler,
        utility_function: Optional[Callable[[QuantumCircuit, Any], float]] = None,
        staggered: bool = False,
    ) -> Dict[str, Any]:
        """Run a specific DD strategy on a target circuit and evaluate its utility.

        Args:
            strategy: DD strategy to apply.
            target_circuit: Target quantum circuit.
            sampler: Qiskit sampler for execution.
            utility_function: Optional utility function to evaluate performance.
            staggered: Whether to apply CR-aware staggering.

        Returns:
            Dictionary with execution results.
        """
        # Apply DD strategy to circuit
        padded_circuit = self.apply_strategy(
            strategy, target_circuit, staggered=staggered
        )

        # Execute circuit
        job = sampler.run(padded_circuit, shots=self.config.shots)
        result = job.result()

        # Get counts (handling different result formats)
        if hasattr(result, "quasi_dists"):
            counts = result.quasi_dists[0]
        elif hasattr(result, "get_counts"):
            counts = result.get_counts()
        else:
            raise ValueError("Unable to extract counts from result")

        # Calculate utility if function provided
        utility_value = None
        if utility_function is not None:
            utility_value = utility_function(padded_circuit, result)

        return {
            "counts": counts,
            "utility": utility_value,
            "padded_circuit": padded_circuit,
            "staggered": staggered,
            "result": result,  # Include full result for flexibility
        }
