"""Three-gene oscillator circuits.

Two variants are exported:

* :class:`OscillatorCircuit` -- Elowitz-Leibler repressilator (Hopfield form).
  Robust limit cycle for ``alpha=10, n=4`` and similar parameters. This is the
  one used for the validation suite because it oscillates sustainably from
  generic initial conditions without parameter fine-tuning.

* :class:`DissertationOscillatorCircuit` -- 3-gene cyclic repression with an
  external inducer S, as written in Dissertation §3.4.2. Mathematically
  consistent in Hopfield form but the limit cycle is fragile and depends on
  parameter fine-tuning. Kept for reference; reproduces the bifurcation
  structure of the original (non-Hopfield) Wang-style 3-gene model.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Elowitz-Leibler repressilator (Hopfield form)  --  primary oscillator
# ---------------------------------------------------------------------------

@dataclass
class OscillatorCircuit:
    """Elowitz-Leibler repressilator, three genes in a cyclic repression loop.

    Original (Elowitz & Leibler, 2000) form (after :math:`k = 1`):
    .. math::

        \\dot{p}_i = -p_i + \\frac{\\alpha}{1 + p_{i-1}^n}.

    Hopfield reformulation using ``1 / (1 + x^n) = 1 - x^n / (1 + x^n)``:
    .. math::

        \\dot{p}_i = -\\alpha\\,\\varphi(p_{i-1}) - p_i + \\alpha,
        \\qquad \\varphi(x) = \\frac{x^n}{1 + x^n}.

    This matches the scHopfield form ``dx/dt = W sigma(x) + I - gamma x``
    exactly, with

    .. math::

        W = -\\alpha \\begin{pmatrix} 0 & 0 & 1 \\\\ 1 & 0 & 0 \\\\ 0 & 1 & 0 \\end{pmatrix},
        \\quad I = (\\alpha, \\alpha, \\alpha), \\quad \\gamma = (1, 1, 1).

    Defaults (``alpha=10, n=4, k=1``) produce a robust limit cycle from
    generic initial conditions (range ~[0.4, 7.7], period ~ 12 time units).

    Parameters
    ----------
    alpha : float
        Maximal transcription rate. Larger alpha -> larger oscillation
        amplitude and shorter period.
    k : float
        Hill threshold (set to 1 for the canonical form).
    n : int
        Hill coefficient. Must be >= 2 for sustained oscillations; n=4 is
        well inside the limit-cycle regime.
    gamma : float
        Linear degradation rate (set to 1 for the canonical form).
    """

    alpha: float = 10.0
    k: float = 1.0
    n: int = 4
    gamma: float = 1.0
    gene_names: Tuple[str, str, str] = field(default=("x", "y", "z"))

    @property
    def n_genes(self) -> int:
        return 3

    def sigma(self, x: np.ndarray) -> np.ndarray:
        """Hill activation. Vectorized, safe for negative inputs (even n)."""
        xn = np.power(x, self.n)             # even n keeps this nonneg
        return xn / (self.k**self.n + xn)

    def W(self) -> np.ndarray:
        """Ground-truth interaction matrix.

        Cyclic repression: gene ``i`` is repressed by gene ``i-1`` (mod 3).
        Equivalently, ``W[i, i-1] = -alpha`` and all other entries are zero.
        """
        sign_W = np.array(
            [[0, 0, 1],   # x repressed by z (column 2)
             [1, 0, 0],   # y repressed by x (column 0)
             [0, 1, 0]],  # z repressed by y (column 1)
            dtype=np.float64,
        )
        return -self.alpha * sign_W

    def I_vec(self) -> np.ndarray:
        return np.array([self.alpha, self.alpha, self.alpha], dtype=np.float64)

    def gamma_vec(self) -> np.ndarray:
        return np.array([self.gamma, self.gamma, self.gamma], dtype=np.float64)

    def rhs(self, x: np.ndarray) -> np.ndarray:
        return self.W() @ self.sigma(x) + self.I_vec() - self.gamma_vec() * x

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        xn = np.power(x, self.n)
        kn = self.k**self.n
        sigma_prime = self.n * np.power(np.maximum(x, 1e-12), self.n - 1) * kn / (kn + xn) ** 2
        W = self.W()
        J = W * sigma_prime[np.newaxis, :]
        J[np.diag_indices_from(J)] -= self.gamma_vec()
        return J

    def has_limit_cycle(self, t_end: float = 200.0, threshold: float = 0.5,
                         seed: int = 0) -> bool:
        """Heuristic: integrate from a generic IC and check whether the tail
        of the trajectory shows large per-gene std (sign of sustained
        oscillation, not convergence to a fixed point)."""
        from scipy.integrate import solve_ivp
        rng = np.random.default_rng(seed)
        x0 = rng.uniform(0.2, 2.0, size=3)
        sol = solve_ivp(
            lambda t, x: self.rhs(x),
            t_span=(0, t_end), y0=x0,
            t_eval=np.linspace(0, t_end, 2000), method="LSODA",
            rtol=1e-8, atol=1e-10,
        )
        if not sol.success:
            return False
        tail = sol.y[:, int(0.6 * sol.y.shape[1]):]
        return float(tail.std(axis=1).mean()) > threshold

    def sample_initial_conditions(self, n: int = 50, x_max: float = 5.0,
                                   seed: int = 0) -> np.ndarray:
        """Sample diverse ICs along the limit cycle. For the repressilator we
        want ICs spread across the basin, so we use a wide uniform box."""
        rng = np.random.default_rng(seed)
        return rng.uniform(0.1, x_max, size=(n, 3))

    def __repr__(self) -> str:
        return (f"OscillatorCircuit(alpha={self.alpha}, k={self.k}, "
                f"n={self.n}, gamma={self.gamma})")


# ---------------------------------------------------------------------------
# Dissertation §3.4.2 oscillator with external signal S  --  reference
# ---------------------------------------------------------------------------

@dataclass
class DissertationOscillatorCircuit:
    """3-gene cyclic-repression oscillator from Dissertation §3.4.2.

    .. math::

        \\dot{x} = -\\varphi(z) - x + S, \\\\
        \\dot{y} = -\\varphi(x) - \\varphi(z) - y + S, \\\\
        \\dot{z} = -\\varphi(x) - \\varphi(y) - z.

    External signal ``S`` is the bifurcation parameter, producing Hopf and
    Saddle-Node bifurcations. The dissertation reports "decaying oscillations"
    for the standard parameters and "sustained oscillations" only after
    fine-tuning. Kept for reference and to show the framework can express
    this regulatory architecture; the more robust limit cycle for validation
    is the Elowitz :class:`OscillatorCircuit` above.
    """

    w: float = 1.07
    S: float = 0.5
    k: float = 1.0
    n: int = 4
    gamma: float = 0.4
    gene_names: Tuple[str, str, str] = field(default=("x", "y", "z"))

    @property
    def n_genes(self) -> int:
        return 3

    def sigma(self, x: np.ndarray) -> np.ndarray:
        xn = np.power(x, self.n)
        return xn / (self.k**self.n + xn)

    def W(self) -> np.ndarray:
        sign_W = np.array(
            [[0, 0, 1],
             [1, 0, 1],
             [1, 1, 0]],
            dtype=np.float64,
        )
        return -self.w * sign_W

    def I_vec(self) -> np.ndarray:
        return np.array([self.S, self.S, 0.0], dtype=np.float64)

    def gamma_vec(self) -> np.ndarray:
        return np.array([self.gamma, self.gamma, self.gamma], dtype=np.float64)

    def rhs(self, x: np.ndarray) -> np.ndarray:
        return self.W() @ self.sigma(x) + self.I_vec() - self.gamma_vec() * x

    def sample_initial_conditions(self, n: int = 50, x_max: float = 3.0,
                                   seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.uniform(0, x_max, size=(n, 3))

    def __repr__(self) -> str:
        return (f"DissertationOscillatorCircuit(w={self.w}, S={self.S}, "
                f"k={self.k}, n={self.n}, gamma={self.gamma})")
