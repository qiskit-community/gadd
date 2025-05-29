"""GADD: Genetic Algorithm for Dynamical Decoupling optimization."""

from .gadd import GADD, TrainingConfig, TrainingState, TrainingResult
from .strategies import DDSequence, DDStrategy, StandardSequences
from .utility_functions import (
    UtilityFunction,
    SuccessProbability,
    OneNormDistance,
    GHZUtility,
    UtilityFactory,
)
from .experiments import (
    create_bernstein_vazirani_circuit,
    create_ghz_circuit,
    run_bv_experiment,
    run_ghz_experiment,
)

__version__ = "0.1.0"

__all__ = [
    "GADD",
    "TrainingConfig",
    "TrainingState",
    "TrainingResult",
    "DDSequence",
    "DDStrategy",
    "StandardSequences",
    "UtilityFunction",
    "SuccessProbability",
    "OneNormDistance",
    "GHZUtility",
    "UtilityFactory",
    "create_bernstein_vazirani_circuit",
    "create_ghz_circuit",
    "run_bv_experiment",
    "run_ghz_experiment",
]
