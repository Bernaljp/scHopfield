"""Dynamics module for ODE solving and simulation."""

from .solver import ODESolver, create_solver
from .simulation import simulate_trajectory, simulate_perturbation_ode, simulate_shift_ode

# CellOracle-style GRN signal propagation simulation
from .perturbation import (
    simulate_perturbation,
    calculate_perturbation_effect_scores,
    run_ko_screen,
    score_ko_panel,
    run_pairwise_ko_screen,
    compute_epistasis,
    run_dose_response,
    dose_levels_from_fractions,
    knockout_displacement_flow,
    perturbation_cascade,
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
    'simulate_perturbation',
    'simulate_shift',  # alias
    'calculate_perturbation_effect_scores',
    'run_ko_screen',
    'score_ko_panel',
    'run_pairwise_ko_screen',
    'compute_epistasis',
    'run_dose_response',
    'dose_levels_from_fractions',
    # Propagated (ODE) perturbation readouts
    'knockout_displacement_flow',
    'perturbation_cascade',
]
