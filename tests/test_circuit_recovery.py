"""Regression test pinning the synthetic-circuit recovery that Figure 2 reports.

These two circuits are Figure 2. They need no downloaded data and they carry their own
ground truth, so ``fit_circuit`` returns both the inferred and the true interaction
matrix and the comparison needs nothing external. That makes them the one place where
the published numbers can be pinned by a test rather than described by a docstring.

Marked ``slow`` because each fit trains a model for about a minute. Excluded from the
default ``pytest`` run by ``addopts``; run with ``pytest -m slow``.

What is pinned, and why the tolerances are what they are
--------------------------------------------------------
**Determinism.** At ``seed=0`` two fits of the same circuit are bitwise identical,
measured as ``max|run1 - run2| = 0.000e+00``, so the assertion is exact equality with no
tolerance at all. This is the assertion that would catch an unseeded RNG or a
nondeterministic reduction creeping back in.

**Accuracy against ground truth.** Across seeds 0, 1 and 2 the worst elementwise
deviation from the true matrix is 4.42e-3 for the toggle, whose largest weight is 5.00,
and 1.21e-2 for the oscillator, whose largest weight is 10.00. Both sit under
``0.0025 * |W_true|.max()``, which is 1.25e-2 and 2.5e-2 respectively, leaving two to
three times headroom. A single scale-relative rule therefore covers both circuits, and a
real regression in the optimizer moves the error by far more than that margin.

**Sign accuracy.** Every nonzero edge of both circuits gets the right sign, so this is
pinned at exactly 1.0 rather than at a threshold.
"""
import numpy as np
import pytest

from scHopfield.validation import fit_circuit, simulate_circuit
from scHopfield.validation.circuits import OscillatorCircuit, ToggleCircuit
from scHopfield.validation.metrics import edge_sign_accuracy

# Relative tolerance on the recovered weights, as a fraction of the circuit's largest
# true weight. See the module docstring for the measurements behind it.
REL_ATOL = 0.0025

CIRCUITS = {
    # The bistable toggle switch of Figure 2a-e.
    "toggle": ToggleCircuit(a=5.0, b=4.0),
    # The three-gene repressilator of Figure 2f-j.
    "oscillator": OscillatorCircuit(alpha=10.0, n=4),
}


def _fit(circuit, seed=0):
    adata = simulate_circuit(circuit, seed=seed)
    return fit_circuit(adata, scaffold_mode="full", seed=seed)


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_circuit_recovery_matches_ground_truth(name):
    res = _fit(CIRCUITS[name])
    W_inf = np.asarray(res["W_inferred"])
    W_true = np.asarray(res["W_true"])

    assert W_inf.shape == W_true.shape
    atol = REL_ATOL * np.abs(W_true).max()
    worst = np.abs(W_inf - W_true).max()
    assert worst <= atol, f"{name}: worst deviation {worst:.3e} exceeds {atol:.3e}"
    assert edge_sign_accuracy(W_inf, W_true) == 1.0


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_circuit_fit_is_bitwise_reproducible(name):
    a = _fit(CIRCUITS[name])["W_inferred"]
    b = _fit(CIRCUITS[name])["W_inferred"]
    assert np.array_equal(np.asarray(a), np.asarray(b)), (
        f"{name}: two fits at seed 0 differ by "
        f"{np.abs(np.asarray(a) - np.asarray(b)).max():.3e}; the fit is not deterministic"
    )
