"""Simple, configurable Monte Carlo option-pricing engine."""

from .inputs import SimulationInputs
from .autocallable_engine import AutocallableEngine
from .manager import (
    MultiRunSimulationResult,
    SimulationManager, 
    SimulationResult
)
from .engine import AUTOCALLABLE_MODELS, MODELS, MonteCarloEngine
from .payoffs import PAYOFFS
from .sampling import SAMPLERS
from .discretizations import DISCRETIZATIONS

__all__ = [
    "DISCRETIZATIONS",
    "AutocallableEngine",
    "AUTOCALLABLE_MODELS",
    "MODELS",
    "MonteCarloEngine",
    "MultiRunSimulationResult",
    "PAYOFFS",
    "SAMPLERS",
    "SimulationInputs",
    "SimulationManager",
    "SimulationResult",
]
