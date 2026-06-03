"""Two-gene mutual inhibition with positive autoregulation (toggle switch).

Source: Dissertation §3.4.1. Hopfield-reformulated, so the ground-truth
interaction matrix W is known exactly.

Original (Cherry-Adler-style) form
----------------------------------
.. math::

    \\frac{dx_1}{dt} = \\frac{a_1 x_1^n}{k^n + x_1^n} + \\frac{b_1 k^n}{k^n + x_2^n} - \\gamma_1 x_1,
    \\qquad
    \\frac{dx_2}{dt} = \\frac{a_2 x_2^n}{k^n + x_2^n} + \\frac{b_2 k^n}{k^n + x_1^n} - \\gamma_2 x_2.

Hopfield form (after the identity ``k^n / (k^n + x^n) = 1 - x^n / (k^n + x^n)``)
-------------------------------------------------------------------------------
.. math::

    \\frac{dx_1}{dt} = a_1 \\sigma(x_1) - b_1 \\sigma(x_2) - \\gamma_1 x_1 + b_1,
    \\qquad
    \\frac{dx_2}{dt} = a_2 \\sigma(x_2) - b_2 \\sigma(x_1) - \\gamma_2 x_2 + b_2,

with :math:`\\sigma(x) = x^n / (k^n + x^n)`. This matches the scHopfield form
``dx/dt = W sigma(x) + I - gamma x`` exactly, so the ground-truth interaction
matrix is

.. math::

    W = \\begin{pmatrix} a_1 & -b_1 \\\\ -b_2 & a_2 \\end{pmatrix}, \\quad
    I = (b_1, b_2), \\quad \\gamma = (\\gamma_1, \\gamma_2).

Negative off-diagonal entries encode mutual repression; positive diagonal entries
encode positive autoregulation.

Default parameters reproduce Figure 3.5 of the dissertation
(:math:`a_1 = a_2 = 5`, :math:`k = 1`, :math:`n = 4`,
:math:`\\gamma_1 = \\gamma_2 = 3`). Vary :math:`b` to traverse the pitchfork
bifurcation from a single equilibrium at the origin (monostable) to two stable
equilibria on the diagonal (bistable).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple
import numpy as np


@dataclass
class ToggleCircuit:
    """Two-gene mutual inhibition + positive autoregulation circuit.

    Parameters
    ----------
    a : float
        Positive autoregulation strength (gene activates itself). Equal for both genes.
    b : float
        Mutual inhibition strength. Critical bifurcation parameter:
        for the default ``a=5, k=1, n=4, gamma=3``, the system is monostable
        for ``b < ~2`` and bistable for ``b > ~2``.
    k : float
        Hill threshold (half-maximal activation).
    n : int
        Hill coefficient (cooperativity).
    gamma : float
        Linear degradation rate, equal for both genes.
    """

    a: float = 5.0
    b: float = 4.0
    k: float = 1.0
    n: int = 4
    gamma: float = 3.0
    gene_names: Tuple[str, str] = field(default=("x1", "x2"))

    @property
    def n_genes(self) -> int:
        return 2

    def sigma(self, x: np.ndarray) -> np.ndarray:
        """Per-gene Hill activation, applied elementwise."""
        xn = np.power(np.maximum(x, 0.0), self.n)
        return xn / (self.k**self.n + xn)

    def W(self) -> np.ndarray:
        """Ground-truth interaction matrix in Hopfield form.

        Returns
        -------
        W : np.ndarray of shape (2, 2)
            ``W[i, j]`` is the influence of TF ``j`` on gene ``i``.
            Diagonal = positive autoregulation (+a). Off-diagonal = mutual
            repression (-b).
        """
        return np.array(
            [[self.a, -self.b],
             [-self.b, self.a]],
            dtype=np.float64,
        )

    def I_vec(self) -> np.ndarray:
        """Ground-truth bias / basal transcription vector."""
        return np.array([self.b, self.b], dtype=np.float64)

    def gamma_vec(self) -> np.ndarray:
        """Ground-truth degradation rate vector."""
        return np.array([self.gamma, self.gamma], dtype=np.float64)

    def rhs(self, x: np.ndarray) -> np.ndarray:
        """Right-hand side of the ODE: dx/dt = W sigma(x) + I - gamma x."""
        return self.W() @ self.sigma(x) + self.I_vec() - self.gamma_vec() * x

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Analytic Jacobian at state ``x``. Useful for stability analysis.

        d/dx_j [W sigma(x) + I - gamma x]_i
            = W_{i,j} * sigma'(x_j)            for i != j
            = W_{i,i} * sigma'(x_i) - gamma_i  for i == j
        with sigma'(x) = n * x^(n-1) * k^n / (k^n + x^n)^2.
        """
        xn = np.power(np.maximum(x, 0.0), self.n)
        kn = self.k**self.n
        sigma_prime = self.n * np.power(np.maximum(x, 1e-12), self.n - 1) * kn / (kn + xn) ** 2
        W = self.W()
        J = W * sigma_prime[np.newaxis, :]   # broadcast over rows
        J[np.diag_indices_from(J)] -= self.gamma_vec()
        return J

    def equilibria(self, n_starts: int = 200, x_max: float = 10.0,
                   tol: float = 1e-6, seed: int = 0) -> np.ndarray:
        """Find fixed points by integrating from many random initial conditions
        until the system settles, then deduplicating with tolerance ``tol``.

        Returns an array of shape (n_unique_equilibria, 2).
        """
        from scipy.integrate import solve_ivp
        rng = np.random.default_rng(seed)
        starts = rng.uniform(0, x_max, size=(n_starts, 2))
        finals = []
        for x0 in starts:
            sol = solve_ivp(
                lambda t, x: self.rhs(x),
                t_span=(0, 100.0), y0=x0,
                t_eval=[100.0], method="LSODA",
                rtol=1e-8, atol=1e-10,
            )
            if sol.success:
                finals.append(sol.y[:, -1])
        finals = np.array(finals)

        # Deduplicate
        unique = []
        for p in finals:
            if not any(np.linalg.norm(p - q) < tol * 100 for q in unique):
                unique.append(p)
        return np.array(unique)

    def is_bistable(self) -> bool:
        """Quick check: does the circuit have more than one stable equilibrium?"""
        eqs = self.equilibria()
        n_stable = 0
        for x in eqs:
            J = self.jacobian(x)
            if np.all(np.real(np.linalg.eigvals(J)) < 0):
                n_stable += 1
        return n_stable >= 2

    def sample_initial_conditions(self, n: int = 50, x_max: float = 6.0,
                                   seed: int = 0) -> np.ndarray:
        """Sample IC's uniformly in [0, x_max]^2, used as starting points for
        long trajectories during data generation."""
        rng = np.random.default_rng(seed)
        return rng.uniform(0, x_max, size=(n, 2))

    def __repr__(self) -> str:
        return (f"ToggleCircuit(a={self.a}, b={self.b}, k={self.k}, "
                f"n={self.n}, gamma={self.gamma})")
