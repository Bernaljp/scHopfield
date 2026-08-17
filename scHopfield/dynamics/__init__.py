"""Dynamics module for ODE solving and simulation."""

from .solver import ODESolver, create_solver
from .simulation import (
    simulate_trajectory,
    simulate_perturbation_ode,  # ODE-based single-cell perturbation
    simulate_shift_ode,
    calculate_trajectory_flow
)

# CellOracle-style GRN signal propagation simulation
from .perturbation import (
    simulate_perturbation,  # Main perturbation function (all cells)
    calculate_perturbation_effect_scores,
    calculate_cell_transition_scores,
    get_top_affected_genes,
    compare_perturbations,
    run_ko_screen,
    score_ko_panel,
    run_pairwise_ko_screen,
    compute_epistasis,
    run_dose_response,
    dose_levels_from_fractions,
)

# Deprecated alias, kept for backward compatibility. `simulate_perturbation` is
# the canonical name; `simulate_shift` will be removed in a future release.
import functools as _functools
import warnings as _warnings


@_functools.wraps(simulate_perturbation)
def simulate_shift(*args, **kwargs):
    _warnings.warn(
        "sch.dyn.simulate_shift is deprecated and will be removed in a future "
        "release; use sch.dyn.simulate_perturbation instead (identical behavior).",
        DeprecationWarning,
        stacklevel=2,
    )
    return simulate_perturbation(*args, **kwargs)

__all__ = [
    # ODE-based simulation
    'ODESolver',
    'create_solver',
    'simulate_trajectory',
    'simulate_perturbation_ode',
    'simulate_shift_ode',
    'calculate_trajectory_flow',
    # CellOracle-style GRN propagation (main functions)
    'simulate_perturbation',
    'simulate_shift',  # alias
    'calculate_perturbation_effect_scores',
    'calculate_cell_transition_scores',
    'get_top_affected_genes',
    'compare_perturbations',
    # KO screen helpers
    'run_ko_screen',
    'score_ko_panel',
    'run_pairwise_ko_screen',
    'compute_epistasis',
    'run_dose_response',
    'dose_levels_from_fractions',
]
