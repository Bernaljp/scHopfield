"""Inference module for network parameter estimation."""

from .interactions import fit_interactions
from .optimizer import ScaffoldOptimizer, MaskedLinearLayer
from .datasets import CustomDataset
from .scaffold import build_scaffold, scaffold_from_edges

__all__ = [
    'fit_interactions',
    'ScaffoldOptimizer',
    'MaskedLinearLayer',
    'CustomDataset',
    'build_scaffold',
    'scaffold_from_edges',
]
