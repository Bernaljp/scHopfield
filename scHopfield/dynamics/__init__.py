"""Dynamics module for ODE solving and simulation."""

from .solver import ODESolver, create_solver
from .simulation import (
    simulate_trajectory,
    simulate_perturbation_ode  # ODE-based single-cell perturbation
)

# CellOracle-style GRN signal propagation simulation
from .perturbation import (
    simulate_perturbation,  # Main perturbation function (all cells)
    calculate_perturbation_effect_scores,
    calculate_cell_transition_scores,
    get_top_affected_genes,
    compare_perturbations
)

# Alias for backward compatibility
simulate_shift = simulate_perturbation

__all__ = [
    # ODE-based simulation
    'ODESolver',
    'create_solver',
    'simulate_trajectory',
    'simulate_perturbation_ode',
    # CellOracle-style GRN propagation (main functions)
    'simulate_perturbation',
    'simulate_shift',  # alias
    'calculate_perturbation_effect_scores',
    'calculate_cell_transition_scores',
    'get_top_affected_genes',
    'compare_perturbations'
]
