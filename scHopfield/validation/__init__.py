"""scHopfield validation suite.

Synthetic gene circuits with known ground-truth interaction matrices, used to
verify that scHopfield can recover the underlying GRN from analytic-derivative
(dx/dt) data alone. Each circuit module exposes:

    simulate(...)   -> (AnnData, ground_truth_W, ground_truth_params)
    W()             -> the literal interaction matrix the circuit equations imply

Circuits live in `scHopfield.validation.circuits`. Shared infrastructure
(generic ODE/SDE simulator, fitting wrapper, evaluation metrics) lives at the
top level of this package.

Two of the circuits are ports of published models. Each cites its source
publication in its own module docstring.
"""

from . import circuits
from .simulate import simulate_circuit
from .fit_validation import fit_circuit
from .metrics import (
    edge_sign_accuracy,
    edge_signed_correlation,
    spectral_overlap,
    symmetry_index,
    frobenius_distance,
    summarize_recovery,
)

__all__ = [
    "circuits",
    "simulate_circuit",
    "fit_circuit",
    "edge_sign_accuracy",
    "edge_signed_correlation",
    "spectral_overlap",
    "symmetry_index",
    "frobenius_distance",
    "summarize_recovery",
]
