"""Regression tests for the Hill activation and its derivative.

The Hill derivative phi'(x) = n * phi(1-phi)/x must include the factor n. A
previous version of `d_sigmoid` omitted it (correct only for n=1); this test
pins the correct behavior against finite differences.
"""
import numpy as np
import pytest

from scHopfield._utils.math import sigmoid, d_sigmoid


@pytest.mark.parametrize("n", [1.0, 2.0, 2.72, 4.0])
@pytest.mark.parametrize("k", [0.5, 1.0, 2.0])
def test_d_sigmoid_matches_finite_difference(n, k):
    x = np.array([0.2, 0.5, 0.9, 1.3, 2.5, 4.0])
    h = 1e-6
    fd = (sigmoid(x + h, k, n) - sigmoid(x - h, k, n)) / (2 * h)
    analytic = d_sigmoid(x, k, n)
    assert np.allclose(analytic, fd, rtol=1e-3, atol=1e-6)


def test_d_sigmoid_includes_hill_exponent():
    # For n != 1 the derivative must scale with n; the buggy version (missing n)
    # would be off by exactly the factor n.
    x = np.array([0.5, 1.0, 2.0])
    k, n = 1.0, 4.0
    analytic = d_sigmoid(x, k, n)
    without_n = sigmoid(x, k, n) * (1 - sigmoid(x, k, n)) / x
    assert np.allclose(analytic, n * without_n, rtol=1e-9)


def test_d_sigmoid_zero_is_finite():
    out = d_sigmoid(np.array([0.0, 1.0]), 1.0, 4.0)
    assert np.all(np.isfinite(out))
    assert out[0] == 0.0  # phi(0) = 0 -> derivative numerator is 0
