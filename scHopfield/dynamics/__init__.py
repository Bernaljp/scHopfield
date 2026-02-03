"""Dynamics module for ODE solving and simulation."""

from .solver import ODESolver, create_solver
from .simulation import simulate_trajectory, simulate_perturbation

# CellOracle-style GRN signal propagation simulation
from .perturbation import (
    simulate_perturbation as simulate_shift,  # CellOracle naming
    calculate_perturbation_effect_scores,
    calculate_cell_transition_scores,
    get_top_affected_genes,
    compare_perturbations
)

__all__ = [
    # ODE-based simulation
    'ODESolver',
    'create_solver',
    'simulate_trajectory',
    'simulate_perturbation',
    # CellOracle-style GRN propagation
    'simulate_shift',
    'calculate_perturbation_effect_scores',
    'calculate_cell_transition_scores',
    'get_top_affected_genes',
    'compare_perturbations'
]
