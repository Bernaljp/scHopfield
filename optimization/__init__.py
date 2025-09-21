"""
Optimization module for scHopfield package.

This module contains classes and functions for optimizing energy landscapes,
interaction matrices, and trajectories.
"""

# Import optimization classes
from .scaffold_optimizer import ScaffoldOptimizer, MaskedLinearLayer, CustomDataset, InteractionMatrixOptimizer
from .energy_optimizer import EnergyOptimizer, TrajectoryOptimizer

__all__ = [
    "ScaffoldOptimizer",
    "MaskedLinearLayer",
    "CustomDataset",
    "InteractionMatrixOptimizer",
    "EnergyOptimizer",
    "TrajectoryOptimizer",
]