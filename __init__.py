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
from .utils.analysis_utilities import (
    change_spines,
    extract_cluster_colors,
    prepare_scaffold_matrix,
    get_correlation_table,
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
    # Analysis utilities
    "change_spines",
    "extract_cluster_colors",
    "prepare_scaffold_matrix",
    "get_correlation_table",
]