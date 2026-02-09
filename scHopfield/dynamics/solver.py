"""ODE solver for gene regulatory network dynamics."""

import numpy as np
from scipy.integrate import odeint, solve_ivp
from typing import Optional, Callable, Tuple
from anndata import AnnData

from .._utils.math import sigmoid
from .._utils.io import get_genes_used


class ODESolver:
    """
    ODE solver for Hopfield network dynamics.

    Solves: dx/dt = W * sigmoid(x) - gamma * x + I

    With constraints to ensure expression values remain non-negative
    and bounded to prevent divergence.
    """

    def __init__(
        self,
        W: np.ndarray,
        I: np.ndarray,
        gamma: np.ndarray,
        threshold: np.ndarray,
        exponent: np.ndarray,
        x_min: float = 0.0,
        x_max: Optional[np.ndarray] = None
    ):
        """
        Initialize ODE solver.

        Parameters
        ----------
        W : np.ndarray
            Interaction matrix
        I : np.ndarray
            External input
        gamma : np.ndarray
            Degradation rates
        threshold : np.ndarray
            Sigmoid threshold parameters
        exponent : np.ndarray
            Sigmoid exponent parameters
        x_min : float, optional (default: 0.0)
            Minimum expression value (non-negative constraint)
        x_max : np.ndarray, optional
            Maximum expression values per gene. If None, no upper bound.
        """
        self.W = W
        self.I = I
        self.gamma = gamma
        self.threshold = threshold
        self.exponent = exponent
        self.x_min = x_min
        self.x_max = x_max

    def dynamics(self, x: np.ndarray, t: float) -> np.ndarray:
        """Compute dx/dt with soft boundary enforcement."""
        # Clip x to valid range before computing dynamics
        x_clipped = np.maximum(x, self.x_min)
        if self.x_max is not None:
            x_clipped = np.minimum(x_clipped, self.x_max)

        sig = sigmoid(x_clipped, self.threshold, self.exponent)
        dxdt = self.W @ sig - self.gamma * x_clipped + self.I

        # Soft boundary: if x is at lower bound, don't let it go more negative
        at_lower = x <= self.x_min
        dxdt[at_lower] = np.maximum(dxdt[at_lower], 0)

        # If x is at upper bound, don't let it go more positive
        if self.x_max is not None:
            at_upper = x >= self.x_max
            dxdt[at_upper] = np.minimum(dxdt[at_upper], 0)

        return dxdt

    def dynamics_ivp(self, t: float, x: np.ndarray) -> np.ndarray:
        """Compute dx/dt for solve_ivp (arguments reversed)."""
        return self.dynamics(x, t)

    def solve(
        self,
        x0: np.ndarray,
        t_span: np.ndarray,
        method: str = 'euler',
        clip_each_step: bool = True
    ) -> np.ndarray:
        """
        Solve ODE from initial condition x0.

        Parameters
        ----------
        x0 : np.ndarray
            Initial condition (must be non-negative)
        t_span : np.ndarray
            Time points
        method : str, optional (default: 'euler')
            Integration method:
            - 'euler': Simple Euler method with clipping (stable, recommended)
            - 'odeint': scipy.integrate.odeint (may diverge)
            - 'RK45': scipy.integrate.solve_ivp with RK45
        clip_each_step : bool, optional (default: True)
            Whether to clip values at each step (prevents divergence)

        Returns
        -------
        np.ndarray
            Solution trajectory (len(t_span) × n_genes)
        """
        # Ensure initial condition is valid
        x0 = np.maximum(x0, self.x_min)
        if self.x_max is not None:
            x0 = np.minimum(x0, self.x_max)

        if method == 'euler':
            return self._solve_euler(x0, t_span, clip_each_step)
        elif method == 'odeint':
            trajectory = odeint(self.dynamics, x0, t_span)
            if clip_each_step:
                trajectory = np.maximum(trajectory, self.x_min)
                if self.x_max is not None:
                    trajectory = np.minimum(trajectory, self.x_max)
            return trajectory
        elif method in ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']:
            return self._solve_ivp(x0, t_span, method, clip_each_step)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'euler', 'odeint', or scipy method names.")

    def _solve_euler(
        self,
        x0: np.ndarray,
        t_span: np.ndarray,
        clip_each_step: bool = True
    ) -> np.ndarray:
        """
        Solve ODE using Euler method with clipping at each step.

        This is more stable for stiff systems and ensures non-negativity.
        """
        n_steps = len(t_span)
        n_genes = len(x0)
        trajectory = np.zeros((n_steps, n_genes))
        trajectory[0] = x0.copy()

        x = x0.copy()
        for i in range(1, n_steps):
            dt = t_span[i] - t_span[i-1]

            # Compute derivative
            dxdt = self.dynamics(x, t_span[i-1])

            # Euler step
            x = x + dt * dxdt

            # Clip to valid range
            if clip_each_step:
                x = np.maximum(x, self.x_min)
                if self.x_max is not None:
                    x = np.minimum(x, self.x_max)

            trajectory[i] = x

        return trajectory

    def _solve_ivp(
        self,
        x0: np.ndarray,
        t_span: np.ndarray,
        method: str,
        clip_each_step: bool
    ) -> np.ndarray:
        """Solve using scipy solve_ivp."""
        sol = solve_ivp(
            self.dynamics_ivp,
            (t_span[0], t_span[-1]),
            x0,
            method=method,
            t_eval=t_span,
            dense_output=False
        )

        trajectory = sol.y.T  # Transpose to (n_times, n_genes)

        if clip_each_step:
            trajectory = np.maximum(trajectory, self.x_min)
            if self.x_max is not None:
                trajectory = np.minimum(trajectory, self.x_max)

        return trajectory


def create_solver(
    adata: AnnData,
    cluster: str,
    degradation_key: str = 'gamma',
    spliced_key: Optional[str] = None,
    x_max_percentile: float = 99.0
) -> ODESolver:
    """
    Create ODE solver for a specific cluster.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with fitted interactions
    cluster : str
        Cluster name
    degradation_key : str, optional
        Key for degradation rates
    spliced_key : str, optional
        Key for expression data to compute bounds. If None, uses scHopfield default.
    x_max_percentile : float, optional (default: 99.0)
        Percentile of expression values to use as upper bound.
        Set to None to disable upper bound.

    Returns
    -------
    ODESolver
        Configured ODE solver with bounds
    """
    from .._utils.io import get_matrix, to_numpy

    genes = get_genes_used(adata)

    W = adata.varp[f'W_{cluster}']
    I = adata.var[f'I_{cluster}'].values[genes]

    gamma_key = f'gamma_{cluster}'
    gamma = adata.var[gamma_key].values[genes] if gamma_key in adata.var else adata.var[degradation_key].values[genes]

    threshold = adata.var['sigmoid_threshold'].values[genes]
    exponent = adata.var['sigmoid_exponent'].values[genes]

    # Compute upper bounds from data
    x_max = None
    if x_max_percentile is not None:
        if spliced_key is None:
            spliced_key = adata.uns.get('scHopfield', {}).get('spliced_key', 'Ms')
        X = to_numpy(get_matrix(adata, spliced_key, genes=genes))
        # Use percentile to avoid outliers setting unreasonable bounds
        x_max = np.percentile(X, x_max_percentile, axis=0)
        # Add some margin (2x the max observed)
        x_max = x_max * 2.0

    return ODESolver(W, I, gamma, threshold, exponent, x_min=0.0, x_max=x_max)
