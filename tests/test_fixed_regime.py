"""A two-component gene keeps the Hill component of its observed state away from that state.

The component a cell uses is assigned by the nearest-threshold rule at the observed expression.
The rule is a cut halfway between the two thresholds, so reading it from a moved state (an
integration step, a clamp, a finite difference) would jump the cell from one Hill onto the other
the moment its expression crosses the cut. These tests pin that every evaluation away from the
observed state holds the observed component fixed.
"""
import numpy as np
import pytest

from scHopfield._utils.math import sigmoid, sigmoid_regime, d_sigmoid_regime, hill_regime
from scHopfield.dynamics.solver import ODESolver

K1, N1, K2, N2 = 0.5, 4.0, 2.0, 3.0
BOUNDARY = 0.5 * (K1 + K2)


def test_regime_argument_overrides_the_value_rule():
    x = np.array([0.3, BOUNDARY - 0.01, BOUNDARY + 0.01, 3.0])
    held_low = sigmoid_regime(x, K1, N1, K2, N2, regime=np.zeros(4, dtype=np.int8))
    held_high = sigmoid_regime(x, K1, N1, K2, N2, regime=np.ones(4, dtype=np.int8))
    assert np.allclose(held_low, sigmoid(x, K1, N1))
    assert np.allclose(held_high, sigmoid(x, K2, N2))


def test_value_rule_jumps_at_the_midpoint_and_the_fixed_regime_does_not():
    below, above = BOUNDARY - 1e-6, BOUNDARY + 1e-6
    by_value = sigmoid_regime(np.array([below, above]), K1, N1, K2, N2)
    assert by_value[0] - by_value[1] > 0.5              # the cut drops the activation
    reg = hill_regime(np.array([below]), K1, K2)
    held = sigmoid_regime(np.array([below, above]), K1, N1, K2, N2,
                          regime=np.repeat(reg, 2))
    assert abs(held[1] - held[0]) < 1e-4                 # continuous within one component


def test_fixed_regime_derivative_matches_finite_difference_across_the_midpoint():
    x0 = np.array([BOUNDARY - 5e-3])
    reg = hill_regime(x0, K1, K2)
    h = 1e-2                                             # the step crosses the midpoint
    fd = (sigmoid_regime(x0 + h, K1, N1, K2, N2, regime=reg)
          - sigmoid_regime(x0 - h, K1, N1, K2, N2, regime=reg)) / (2 * h)
    analytic = d_sigmoid_regime(x0, K1, N1, K2, N2, regime=reg)
    assert np.allclose(fd, analytic, rtol=5e-3)


def _two_gene_solver():
    # Gene 2 is driven by its bias from below its midpoint to a steady level above it, and it is the
    # only regulator (it activates gene 1), so its crossing is what a value-read regime would feel.
    # Nothing represses, so neither gene reaches zero.
    W = np.array([[0.0, 3.0], [0.0, 0.0]])
    return ODESolver(W, np.array([0.1, 2.0]), np.array([1.0, 1.0]),
                     np.array([K1, K1]), np.array([N1, N1]),
                     np.array([K2, K2]), np.array([N2, N2]), x_max=np.array([10.0, 10.0]))


@pytest.mark.parametrize("method", ["euler", "odeint", "RK45"])
def test_solver_holds_the_passed_regime_along_the_trajectory(method):
    solver = _two_gene_solver()
    x0 = np.array([1.0, BOUNDARY - 0.25])
    reg = hill_regime(x0, np.array([K1, K1]), np.array([K2, K2]))
    t = np.linspace(0, 2.0, 41)
    traj = solver.solve(x0, t, method=method, regime=reg)
    # Not vacuous: the regulator starts below its midpoint and is driven across it.
    assert traj[0, 1] < BOUNDARY < traj[-1, 1]
    # Re-evaluate the derivative along the returned path in the fixed regime; it must be the one
    # the solver integrates, which a value-read regime does not reproduce once a gene crosses.
    crossed = False
    for x in traj[:-1]:
        xc = np.maximum(x, 0)
        expected = solver.W @ sigmoid_regime(xc, solver.threshold, solver.exponent,
                                             solver.threshold2, solver.exponent2, regime=reg) \
            - solver.gamma * xc + solver.I
        at_lower = x <= solver.x_min                      # the solver's non-negativity projection
        expected[at_lower] = np.maximum(expected[at_lower], 0)
        got = solver.dynamics(x, 0.0, reg)
        assert np.allclose(got, expected)
        if x[1] > BOUNDARY:
            crossed = True
            assert not np.allclose(got, solver.dynamics(x, 0.0))
    assert crossed


def test_batch_and_single_dynamics_agree_under_a_fixed_regime():
    solver = _two_gene_solver()
    X = np.array([[BOUNDARY + 0.02, 0.4], [0.2, BOUNDARY - 0.02]])
    R = hill_regime(X, np.array([K1, K1])[None, :], np.array([K2, K2])[None, :])
    X_moved = X + np.array([[-0.1, 0.0], [0.0, 0.1]])  # both cells cross a midpoint
    batch = solver.dynamics_batch(X_moved, 0.0, regime=R)
    single = np.vstack([solver.dynamics(X_moved[i], 0.0, R[i]) for i in range(2)])
    assert np.allclose(batch, single)
    assert not np.allclose(batch, solver.dynamics_batch(X_moved, 0.0))
