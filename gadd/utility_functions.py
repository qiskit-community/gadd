"""
Utility function implementations.

This module implements the general class for utility functions as well as common
utility functions described in the paper.
"""

from abc import ABC, abstractmethod
from typing import Dict, Callable, Union, Optional
from qiskit.result import QuasiDistribution, Counts

# Type alias for various count dictionary formats
CountsType = Union[QuasiDistribution, Counts, Dict[Union[str, int], float]]


def normalize_counts(counts: CountsType) -> Dict[str, float]:
    """Normalize measurement counts into a standardized format.

    Converts various count dictionary formats into a consistent representation
    with bitstring keys and normalized probability values. Handles both integer
    and string keys, ensuring proper normalization and format standardization.

    Args:
        counts: Measurement counts in QuasiDistribution, Counts, or dict format.

    Returns:
        Dictionary mapping bitstrings to normalized probabilities.

    Raises:
        ValueError: If counts format is invalid, empty, or total is non-positive.
    """
    if not counts:
        raise ValueError("Empty counts provided")

    # Convert to regular dictionary if needed
    if isinstance(counts, (QuasiDistribution, Counts)):
        counts = dict(counts)

    # Determine number of bits needed if we have integer keys
    n_bits = 1  # Default minimum
    if any(isinstance(k, int) for k in counts.keys()):
        int_keys = [k for k in counts.keys() if isinstance(k, int)]
        if int_keys:
            n_bits = max(len(bin(k)[2:]) for k in int_keys)
    else:
        # If we have string keys, use their length
        n_bits = len(next(iter(counts.keys())))

    # Convert keys to strings if they're integers and normalize values
    processed_counts = {}
    total = 0.0

    for key, value in counts.items():
        # Convert integer keys to binary strings
        if isinstance(key, int):
            str_key = format(key, f"0{n_bits}b")
        else:
            str_key = str(key)

        # Store value, accumulate total
        processed_counts[str_key] = float(value)
        total += float(value)

    # Normalize
    if total <= 0:
        raise ValueError("Total counts/probabilities must be positive")

    return {k: v / total for k, v in processed_counts.items()}


class UtilityFunction(ABC):
    """Abstract base class for DD utility functions used in GADD optimization.

    Utility functions quantify the performance of dynamical decoupling sequences
    by analyzing the measurement outcomes from quantum circuits. They serve as
    the fitness function for the genetic algorithm, guiding the search toward
    sequences that maximize computational fidelity or other performance metrics.

    The utility function takes measurement counts in various formats and returns
    a scalar score between 0 and 1, where higher values indicate better
    performance. The specific computation depends on the target application
    and desired error suppression goals.

    Subclasses must implement the abstract methods to define how measurement
    outcomes are processed and scored for their specific use case.
    """

    @abstractmethod
    def compute(self, counts: CountsType) -> float:
        """Compute utility value from measurement counts."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the utility function."""
        pass

    def verify_state(self, state: str, counts: Dict[str, float]) -> None:
        """
        Verify that a quantum state has the correct format relative to counts.

        Args:
            state: Target state to verify
            counts: Normalized count dictionary

        Raises:
            ValueError: If state format doesn't match counts format
        """
        if not counts:
            raise ValueError("Empty counts provided")

        # Get expected state length from counts
        expected_length = len(next(iter(counts.keys())))
        if len(state) != expected_length:
            raise ValueError(
                f"Target state length ({len(state)}) doesn't match "
                f"measured states length ({expected_length})"
            )

        # Verify state is binary string
        if not set(state).issubset({"0", "1"}):
            raise ValueError("Target state must be a binary string")


class SuccessProbability(UtilityFunction):
    """Utility function measuring success probability for a target quantum state.

    This utility function evaluates DD sequence performance by calculating the
    probability of measuring a specific target state. It is particularly useful
    for oracular algorithms like Bernstein-Vazirani where the ideal outcome is
    a single known bitstring, and higher success probability indicates better
    error suppression.

    The function normalizes measurement counts and returns the probability of
    observing the target state. This provides a direct measure of computational
    success that can guide the genetic algorithm toward sequences that preserve
    the desired quantum computation.

    Args:
        target_state: Expected quantum state as binary string or integer.
            For integer inputs, the value is converted to binary representation.

    Example:
        >>> utility = SuccessProbability("101")
        >>> score = utility.compute(measurement_counts)
        >>> # Returns probability of measuring |101⟩ state
    """

    def __init__(self, target_state: Union[str, int]):
        if isinstance(target_state, int):
            # Convert to binary string - width will be determined from counts
            self._target = bin(target_state)[2:]  # Remove '0b' prefix
        else:
            self._target = str(target_state)
            if not set(self._target).issubset({"0", "1"}):
                raise ValueError("Target state must be a binary string or integer")

    def compute(self, counts: CountsType) -> float:
        """
        Compute success probability from measurement counts.

        Args:
            counts: Measurement outcomes in various formats

        Returns:
            Success probability for measuring target state
        """
        normalized = normalize_counts(counts)

        # If target was from integer, might need to pad
        target = self._target.zfill(len(next(iter(normalized.keys()))))

        # Verify target format
        self.verify_state(target, normalized)

        return normalized.get(target, 0.0)

    def get_name(self) -> str:
        return f"Success Probability (|{self._target}⟩)"

    def __str__(self) -> str:
        """Return human-readable string representation."""
        return f"Success probability for |{self._target}⟩"


class OneNormDistance(UtilityFunction):
    """Utility function based on 1-norm distance to an ideal probability distribution.

    This utility function measures DD sequence performance by comparing the observed
    measurement distribution to an ideal target distribution. It computes the
    1-norm (total variation) distance and returns 1 minus half the distance,
    normalized to [0,1] where 1 indicates perfect agreement.

    This metric is particularly useful for quantum state preparation tasks where
    the ideal outcome is a known probability distribution over measurement outcomes,
    such as GHZ states or other entangled states with well-defined measurement
    statistics.

    The 1-norm distance is calculated as: Σ|p_ideal(k) - p_observed(k)| over all
    possible measurement outcomes k, providing a comprehensive measure of
    distributional fidelity.

    Args:
        ideal_distribution: Target probability distribution mapping states to
            their ideal probabilities. States can be specified as binary strings
            or integers.

    Example:
        >>> ideal = {"000": 0.5, "111": 0.5}  # GHZ state
        >>> utility = OneNormDistance(ideal)
        >>> score = utility.compute(measurement_counts)
    """

    def __init__(self, ideal_distribution: Dict[Union[str, int], float]):
        self.ideal_distribution = normalize_counts(ideal_distribution)

    def compute(self, counts: CountsType) -> float:
        """
        Compute 1-norm distance between measured and ideal distributions.

        Args:
            counts: Measurement outcomes in various formats

        Returns:
            1 - (1/2 * ∑|p(k) - p̂(k)|), normalized to [0,1] where 1 is perfect
        """
        normalized = normalize_counts(counts)

        # Verify dimensions match
        if len(next(iter(normalized.keys()))) != len(
            next(iter(self.ideal_distribution.keys()))
        ):
            raise ValueError("Measured and ideal state dimensions don't match")

        # Calculate absolute differences
        total_diff = 0.0
        for state in set(self.ideal_distribution.keys()) | set(normalized.keys()):
            ideal_prob = self.ideal_distribution.get(state, 0.0)
            measured_prob = normalized.get(state, 0.0)
            total_diff += abs(ideal_prob - measured_prob)

        return 1 - (total_diff / 2)

    def get_name(self) -> str:
        return "1-Norm Distance"

    def __str__(self) -> str:
        """Return human-readable string representation."""
        ideal_states = [
            state for state, prob in self.ideal_distribution.items() if prob > 0.01
        ]
        if len(ideal_states) <= 3:
            states_str = ", ".join(f"|{state}⟩" for state in ideal_states)
        else:
            states_str = f"{len(ideal_states)} target states"
        return f"1-norm fidelity to {states_str}"


class GHZUtility(OneNormDistance):
    """Specialized utility function for Greenberger-Horne-Zeilinger state preparation.

    This utility function is optimized for evaluating DD sequence performance on
    GHZ state preparation circuits. GHZ states are maximally entangled states of
    the form (|00...0⟩ + |11...1⟩)/√2, which should ideally produce equal
    probability of measuring all-zero or all-one bitstrings.

    The function automatically constructs the ideal GHZ distribution for the
    specified number of qubits and uses the 1-norm distance metric to evaluate
    how well the observed measurements match this target distribution. This
    provides a specialized utility function for one of the key benchmarking
    tasks described in the GADD paper.

    Args:
        n_qubits: Number of qubits in the GHZ state preparation circuit.

    Raises:
        ValueError: If n_qubits is not positive.

    Example:
        >>> utility = GHZUtility(3)  # For 3-qubit GHZ state
        >>> # Expects 50% probability each for |000⟩ and |111⟩
    """

    def __init__(self, n_qubits: int):
        if n_qubits <= 0:
            raise ValueError("Number of qubits must be positive")

        # GHZ state should have equal superposition of all-0 and all-1 states
        ideal_distribution = {"0" * n_qubits: 0.5, "1" * n_qubits: 0.5}
        super().__init__(ideal_distribution)

    def get_name(self) -> str:
        return "GHZ State Fidelity"

    def __str__(self) -> str:
        """Return human-readable string representation."""
        n_qubits = len(next(iter(self.ideal_distribution.keys())))
        return f"GHZ state fidelity ({n_qubits} qubits)"


class CustomUtility(UtilityFunction):
    """Wrapper for user-defined custom utility functions.

    This class allows users to provide their own utility functions for specialized
    applications beyond the standard use cases. The custom function receives
    normalized measurement counts as a dictionary mapping bitstrings to probabilities
    and should return a scalar utility value.

    This flexibility enables GADD optimization for novel quantum algorithms or
    specialized error suppression goals not covered by the built-in utility
    functions. The custom function should be deterministic and return values
    in a consistent range to ensure proper genetic algorithm convergence.

    Args:
        function: Custom function taking normalized counts dict and returning float.
        name: Optional descriptive name for the utility function.

    Example:
        >>> def my_utility(counts):
        ...     return counts.get("000", 0) + 0.5 * counts.get("001", 0)
        >>> utility = CustomUtility(my_utility, "Custom Weighting")
    """

    def __init__(
        self,
        function: Callable[[Dict[str, float]], float],
        name: Optional[str] = "Custom Utility",
    ):
        self.function = function
        self._name = name

    def compute(self, counts: CountsType) -> float:
        # Normalize counts before passing to custom function
        normalized = normalize_counts(counts)
        return self.function(normalized)

    def get_name(self) -> str:
        return self._name

    def __str__(self) -> str:
        """Return human-readable string representation."""
        return f"Custom Utility {self._name}"


class UtilityFactory:
    """Factory class providing convenient creation methods for common utility functions.

    This factory simplifies the instantiation of utility functions for common
    use cases in dynamical decoupling optimization. It provides static methods
    that encapsulate the parameter setup for standard utility functions, reducing
    boilerplate code and potential configuration errors.

    The factory methods correspond to the main experimental scenarios described
    in the GADD paper: success probability for oracular algorithms, 1-norm
    distance for state preparation tasks, GHZ state fidelity for entanglement
    benchmarks, and custom functions for specialized applications.

    Example:
        >>> # Create utility for Bernstein-Vazirani with target "101"
        >>> utility = UtilityFactory.success_probability("101")
        >>>
        >>> # Create utility for 4-qubit GHZ state
        >>> ghz_utility = UtilityFactory.ghz_state(4)
    """

    @staticmethod
    def success_probability(target_state: Union[str, int]) -> UtilityFunction:
        """Create success probability utility function."""
        return SuccessProbability(target_state)

    @staticmethod
    def one_norm(ideal_distribution: Dict[Union[str, int], float]) -> UtilityFunction:
        """Create 1-norm distance utility function."""
        return OneNormDistance(ideal_distribution)

    @staticmethod
    def ghz_state(n_qubits: int) -> UtilityFunction:
        """Create GHZ state utility function."""
        return GHZUtility(n_qubits)

    @staticmethod
    def custom(
        function: Callable[[Dict[str, float]], float],
        name: str = "Custom Utility Function",
    ) -> UtilityFunction:
        """Create custom utility function."""
        return CustomUtility(function, name)
