from typing import List, Tuple, Optional, Callable, Dict, Any, Union

import rustworkx as rx

import numpy as np
from numpy.random import Generator, default_rng
from numpy.random.bit_generator import BitGenerator, SeedSequence

from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit_ibm_runtime import Sampler
import matplotlib.pyplot as plt
from dataclasses import dataclass
from .sequences import DDStrategy, DDSequence, StandardSequences
from .library import *
from .utility_functions import UtilityFunction


@dataclass
class TrainingConfig:
    """Configuration parameters for GADD training."""

    population_size: int = 16
    sequence_length: int = 8
    parent_fraction: float = 0.25
    n_iterations: int = 20
    mutation_probability: float = 0.75
    optimization_level: int = 1
    shots: int = 4000


@dataclass
class TrainingResult:
    """Results from GADD training."""

    best_sequence: DDStrategy
    iteration_data: List[Dict[str, float]]
    comparison_data: Dict[str, float]


class GADD:
    """Genetic Algorithm for Dynamical Decoupling optimization."""

    def __init__(
        self,
        backend: Optional[Backend] = None,
        utility_function: Optional[UtilityFunction] = None,
        coloring: Optional[Dict] = None,
        seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
    ):
        """Initialize the GADD class."""
        self._backend = backend
        self._utility_function = utility_function
        self._seed = seed
        self._rng = default_rng(seed=seed)
        self.config = TrainingConfig()

        if not coloring:
            # output is {qubit: color} key-value pairs
            self._coloring = rx.graph_greedy_color(backend.coupling_map.graph.to_undirected())

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
        if not isinstance(coloring, dict):
            raise TypeError(
                "Coloring must be either `greedy` or a dictionary keyed to colors with a list of indices for each."
            )
        self._coloring = coloring

    def train(
        self,
        sampler: Sampler,
        training_circuit: QuantumCircuit,
        utility_function: Callable[[QuantumCircuit, List[float]], float],
        mode: str = "random",
        save_iterations: bool = True,
        comparison_seqs: List[str] = None,
    ) -> Tuple[DDStrategy, TrainingResult]:
        """Train DD sequences using genetic algorithm."""

        if mode not in ["random", "uniform"]:
            raise TypeError("Mode must be one of 'random' or 'uniform'.")

        # Initialize population
        population = self._initialize_population()
        iteration_data = []

        for iteration in range(self.config.n_iterations):
            # Evaluate population
            scores = self._evaluate_population(
                population, sampler, training_circuit, utility_function
            )

            # Save iteration data
            if save_iterations:
                iteration_data.append(
                    {
                        "generation": iteration,
                        "best_score": max(scores.values()),
                        "population": scores,
                    }
                )

            # Generate next population
            population = self._next_generation(population, scores)

        # Get best sequence
        best_sequence = max(population, key=lambda x: scores[x])

        # Run comparison sequences if requested
        comparison_data = {}
        if comparison_seqs:
            comparison_data = self._evaluate_standard_sequences(
                comparison_seqs, sampler, training_circuit, utility_function
            )

        return (
            DDStrategy(best_sequence),
            TrainingResult(DDStrategy(best_sequence), iteration_data, comparison_data),
        )

    def plot(self, sequence: DDSequence, results: TrainingResult):
        """Plot training progression and comparison data."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot training progression
        generations = [d["generation"] for d in results.iteration_data]
        scores = [d["best_score"] for d in results.iteration_data]
        ax1.plot(generations, scores, "-o")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Best Score")
        ax1.set_title("Training Progression")

        # Plot comparison data
        if results.comparison_data:
            names = list(results.comparison_data.keys())
            values = list(results.comparison_data.values())
            ax2.bar(names, values)
            ax2.set_xlabel("Sequence Type")
            ax2.set_ylabel("Score")
            ax2.set_title("Sequence Comparison")
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    def run(
        self, strategy: DDStrategy, target_circuit: QuantumCircuit, sampler: Sampler
    ) -> Dict[str, float]:
        """Run a specific sequence on a target circuit."""
        # TODO: add color map
        padded_circuit = pad_sequence(target_circuit, sequence)
        result = sampler.run(padded_circuit).result()
        return self._process_results(result)

    def _initialize_population(pop_size, seq_length, num_colors, group_size=8, mode="random"):
        """
        Initialize population of DD sequences with specified characteristics

        Args:
            pop_size (int): Size of the population to generate
            seq_length (int): Length of each DD sequence
            num_colors (int): Number of distinct sequences per strategy
            group_size (int): Size of the group (default 8 for {I,Ib,X,Xb,Y,Yb,Z,Zb})
            mode (str): Mode for generating initial population
                - "uniform": Each group element appears equally often in each position
                - "random": Random population meeting group constraints

        Returns:
            list[str]: List of DD strategies encoded as strings

        Note:
            Each strategy consists of num_colors sequences of length seq_length
            The sequences must multiply to identity under group multiplication
        """
        if mode == "uniform":
            # Calculate how many times each element should appear at each position
            reps_per_element = pop_size // group_size
            if reps_per_element == 0:
                raise ValueError(
                    "Population size must be at least as large as group size for uniform mode"
                )

            population = []

            # Generate base patterns that are cyclic shifts of group elements
            base_patterns = []
            for shift in range(seq_length):
                pattern = ""
                for pos in range(seq_length - 1):  # Leave last element to enforce constraint
                    element = (pos + shift) % group_size
                    pattern += str(element)
                base_patterns.append(pattern)

            # For each base pattern, generate variants by permuting elements
            for pattern in base_patterns:
                for _ in range(reps_per_element):
                    # Convert pattern to list for manipulation
                    sequence = list(pattern)

                    # Calculate last element to make sequence multiply to identity
                    prefix_product = 0
                    for element in sequence:
                        prefix_product = multiply(prefix_product, int(element))
                    last_element = invert(prefix_product)
                    sequence.append(str(last_element))

                    # Create full strategy by repeating for each color
                    strategy = "".join(sequence) * num_colors
                    population.append(strategy)

            # If we need more sequences to reach pop_size, add random valid sequences
            while len(population) < pop_size:
                random_sequence = list(str(np.random.randint(0, group_size)) * (seq_length - 1))
                last_element = invert(multiply_list([int(x) for x in random_sequence]))
                random_sequence.append(str(last_element))
                strategy = "".join(random_sequence) * num_colors
                population.append(strategy)

        else:  # Random mode
            population = []
            while len(population) < pop_size:
                # Generate random sequences for each color that multiply to identity
                strategy_parts = []
                for _ in range(num_colors):
                    sequence = list(str(np.random.randint(0, group_size)) * (seq_length - 1))
                    last_element = invert(multiply_list([int(x) for x in sequence]))
                    sequence.append(str(last_element))
                    strategy_parts.append("".join(sequence))
                strategy = "".join(strategy_parts)

                # Only add if unique
                if strategy not in population:
                    population.append(strategy)

        return population

    # def _initialize_population(self) -> List[DDSequence]:
    #     """Initialize population of DD sequences."""
    #     population = []
    #     for _ in range(self.config.population_size):
    #         sequence = self._generate_random_sequence()
    #         population.append(sequence)
    #     return population

    def _evaluate_population(
        self,
        population: List[DDSequence],
        sampler: Sampler,
        circuit: QuantumCircuit,
        utility_function: Callable,
    ) -> Dict[DDSequence, float]:
        """Evaluate fitness of all sequences in population."""
        scores = {}
        for sequence in population:
            padded_circuit = pad_sequence(circuit, sequence)
            result = sampler.run(padded_circuit).result()
            scores[sequence] = utility_function(padded_circuit, result)
        return scores

    def _next_generation(
        self, population: List[DDSequence], scores: Dict[DDSequence, float]
    ) -> List[DDSequence]:
        """Generate next generation through selection, crossover and mutation."""
        # Sort population by fitness
        sorted_pop = sorted(population, key=lambda x: scores[x], reverse=True)

        # Select parents
        n_parents = int(self.config.parent_fraction * len(population))
        parents = sorted_pop[:n_parents]

        # Generate offspring through crossover and mutation
        offspring = []
        while len(offspring) < len(population) - len(parents):
            p1, p2 = np.random.choice(parents, 2, replace=False)
            child = self._crossover(p1, p2)
            if np.random.random() < self.config.mutation_probability:
                child = self._mutate(child)
            offspring.append(child)

        return parents + offspring

    def _generate_random_sequence(self) -> DDSequence:
        """Generate a random DD sequence."""
        sequence = []
        for _ in range(self.config.sequence_length):
            gate = np.random.choice(["I", "X", "Y", "Z"])
            sequence.append(gate)
        return DDSequence(sequence)

    def _crossover(self, seq1: DDSequence, seq2: DDSequence) -> DDSequence:
        """Perform crossover between two sequences."""
        point = np.random.randint(1, len(seq1.gates))
        child = seq1.gates[:point] + seq2.gates[point:]
        return DDSequence(child)

    def _mutate(self, sequence: DDSequence) -> DDSequence:
        """Mutate a sequence."""
        idx = np.random.randint(len(sequence.gates))
        gates = sequence.gates.copy()
        gates[idx] = np.random.choice(["I", "X", "Y", "Z"])
        return DDSequence(gates)

    def _evaluate_standard_sequences(
        self,
        sequences: List[str],
        sampler: Sampler,
        circuit: QuantumCircuit,
        utility_function: Callable,
    ) -> Dict[str, float]:
        """Evaluate standard DD sequences."""
        results = {}
        std_sequences = StandardSequences()

        for seq_name in sequences:
            sequence = std_sequences.get(seq_name)
            padded_circuit = pad_sequence(circuit, sequence)
            result = sampler.run(padded_circuit).result()
            results[seq_name] = utility_function(padded_circuit, result)

        return results

    def _process_results(self, result: Any) -> Dict[str, float]:
        """Process measurement results."""
        # TODO Implementation depends on specific result format
        return {"success_probability": result.get_counts().get("0", 0) / result.shots}
