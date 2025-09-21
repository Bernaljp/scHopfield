# scHopfield: A modular package for single-cell trajectory analysis using Hopfield-like dynamics

__version__ = "1.0.0"
__author__ = "scHopfield Team"

# Import main interface classes
from .core.base_models import (
    BaseAnalyzer,
    BaseEnergyCalculator,
    BaseOptimizer,
    BaseSimulator,
    ConfigMixin,
    ValidationMixin,
)
from .core.data_processing import DataProcessor
from .utils.utilities import (
    sigmoid,
    fit_sigmoid,
    to_numpy,
    soften,
    rezet,
    ordinal,
)

__all__ = [
    # Base classes
    "BaseAnalyzer",
    "BaseEnergyCalculator",
    "BaseOptimizer",
    "BaseSimulator",
    "ConfigMixin",
    "ValidationMixin",
    # Core classes
    "DataProcessor",
    # Utility functions
    "sigmoid",
    "fit_sigmoid",
    "to_numpy",
    "soften",
    "rezet",
    "ordinal",
]