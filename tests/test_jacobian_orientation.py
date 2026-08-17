"""Regression test for the Jacobian broadcast orientation.

The model velocity is v_i(x) = sum_j W[i,j] phi_j(x_j) - gamma_i x_i + I_i
(see tools/velocity.py). Its Jacobian is therefore

    J_ij = dv_i/dx_j = W[i,j] * phi'_j(x_j) - gamma_i * delta_ij,

i.e. the Hill derivative scales the COLUMN index j (the regulator). In array
terms that is ``W * phi_prime.reshape(1, -1)``. A previous version used
``phi_prime.reshape(-1, 1)`` (scaling rows -> phi'_i); that yields a matrix that
is a similarity transform D_phi^{-1} J D_phi of the true Jacobian, so it has the
SAME eigenvalues (stability analysis is unaffected) but WRONG individual elements
and a wrong antisymmetric (rotational/vorticity) part. This test pins the correct
orientation against a finite-difference Jacobian of the model velocity.
"""
import numpy as np
import pytest

from scHopfield._utils.math import sigmoid, d_sigmoid


def _model_velocity(x, W, gamma, k, n):
    return W @ sigmoid(x, k, n) - gamma * x


def _analytic_jacobian(x, W, gamma, k, n):
    # column-scaled: J_ij = W_ij * phi'_j - gamma_i delta_ij
    phip = d_sigmoid(x, k, n)
    return W * phip.reshape(1, -1) - np.diag(gamma)


def test_jacobian_matches_finite_difference():
    rng = np.random.default_rng(1)
    N = 8
    W = rng.normal(size=(N, N))
    gamma = rng.uniform(0.2, 1.5, size=N)
    k = rng.uniform(0.3, 1.2, size=N)
    n = rng.uniform(1.5, 4.0, size=N)
    x0 = rng.uniform(0.2, 2.0, size=N)

    h = 1e-6
    Jfd = np.zeros((N, N))
    for j in range(N):
        xp = x0.copy(); xp[j] += h
        xm = x0.copy(); xm[j] -= h
        Jfd[:, j] = (_model_velocity(xp, W, gamma, k, n)
                     - _model_velocity(xm, W, gamma, k, n)) / (2 * h)

    J = _analytic_jacobian(x0, W, gamma, k, n)
    assert np.allclose(J, Jfd, rtol=1e-3, atol=1e-5)

    # The row-scaled (buggy) orientation must NOT match the true Jacobian elements,
    # even though its eigenvalues coincide.
    phip = d_sigmoid(x0, k, n)
    J_row = W * phip.reshape(-1, 1) - np.diag(gamma)
    assert not np.allclose(J_row, Jfd, atol=1e-3)
    assert np.allclose(np.sort_complex(np.linalg.eigvals(J_row)),
                       np.sort_complex(np.linalg.eigvals(Jfd)), atol=1e-6)
