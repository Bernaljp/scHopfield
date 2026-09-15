"""ODE solver for gene regulatory network dynamics."""

import numpy as np
from scipy.integrate import odeint, solve_ivp
from typing import Optional
from anndata import AnnData

from .._utils.math import sigmoid, sigmoid_regime
from .._utils.io import get_genes_used, get_hill_params


class ODESolver:
    """
    ODE solver for Hopfield network dynamics.

    Solves: dx/dt = W * sigmoid(x) - gamma * x + I

    With constraints to ensure expression values remain non-negative
    and bounded to prevent divergence. Supports fixing certain genes
    at constant values (e.g., for knockout/overexpression simulations).
    """

    def __init__(
        self,
        W: np.ndarray,
        bias_vector: np.ndarray,
        gamma: np.ndarray,
        threshold: np.ndarray,
        exponent: np.ndarray,
        threshold2: Optional[np.ndarray] = None,
        exponent2: Optional[np.ndarray] = None,
        x_min: float = 0.0,
        x_max: Optional[np.ndarray] = None,
        fixed_indices: Optional[np.ndarray] = None,
        fixed_values: Optional[np.ndarray] = None,
        mix: Optional[np.ndarray] = None,
        active_min: Optional[np.ndarray] = None
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
        fixed_indices : np.ndarray, optional
            Indices of genes to keep fixed (e.g., perturbed genes)
        fixed_values : np.ndarray, optional
            Values to fix the genes at (must match length of fixed_indices)
        """
        self.W = W
        self.I = bias_vector
        self.gamma = gamma
        self.threshold = threshold
        self.exponent = exponent
        # Second Hill component, or None for a single-Hill fit. Which component a cell uses is
        # passed per call as ``regime`` (its observed-state assignment), so a cell keeps its mode
        # along a trajectory rather than switching when its expression crosses the midpoint.
        self.threshold2 = threshold2
        self.exponent2 = exponent2
        # Mixture weight of component 1 when the object assigns components by maximum posterior;
        # None keeps the nearest-threshold rule. Used only where no fixed regime is passed.
        self.mix = mix
        self.active_min = active_min
        self.x_min = x_min
        self.x_max = x_max
        self.fixed_indices = fixed_indices
        self.fixed_values = fixed_values

    def set_fixed_genes(
        self,
        fixed_indices: Optional[np.ndarray],
        fixed_values: Optional[np.ndarray]
    ) -> None:
        """
        Set genes to be held fixed during simulation.

        Parameters
        ----------
        fixed_indices : np.ndarray
            Indices of genes to keep fixed
        fixed_values : np.ndarray
            Values to fix the genes at
        """
        self.fixed_indices = fixed_indices
        self.fixed_values = fixed_values

    def _clip(self, x: np.ndarray) -> np.ndarray:
        x = np.maximum(x, self.x_min)
        if self.x_max is not None:
            x = np.minimum(x, self.x_max)
        return x

    def _clip_trajectory(self, traj: np.ndarray) -> np.ndarray:
        traj = np.maximum(traj, self.x_min)
        if self.x_max is not None:
            traj = np.minimum(traj, self.x_max)
        return traj

    def _enforce_fixed(self, x: np.ndarray) -> None:
        """Overwrite fixed-gene positions (in-place, 1-D)."""
        if self.fixed_indices is not None and len(self.fixed_indices) > 0:
            x[self.fixed_indices] = self.fixed_values

    def _enforce_fixed_trajectory(self, traj: np.ndarray) -> None:
        """Overwrite fixed-gene columns (in-place, 2-D n_times×n_genes)."""
        if self.fixed_indices is not None and len(self.fixed_indices) > 0:
            traj[:, self.fixed_indices] = self.fixed_values

    def dynamics(self, x: np.ndarray, t: float, regime: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute dx/dt with soft boundary enforcement, in each gene's fixed ``regime`` if given."""
        # Clip x to valid range before computing dynamics
        x_clipped = self._clip(x.copy())

        sig = sigmoid_regime(x_clipped, self.threshold, self.exponent,
                             self.threshold2, self.exponent2, regime=regime, a=self.mix, tau=self.active_min)
        dxdt = self.W @ sig - self.gamma * x_clipped + self.I

        # Soft boundary: if x is at lower bound, don't let it go more negative
        at_lower = x <= self.x_min
        dxdt[at_lower] = np.maximum(dxdt[at_lower], 0)

        # If x is at upper bound, don't let it go more positive
        if self.x_max is not None:
            at_upper = x >= self.x_max
            dxdt[at_upper] = np.minimum(dxdt[at_upper], 0)

        # Fixed genes have zero derivative (they don't change)
        if self.fixed_indices is not None and len(self.fixed_indices) > 0:
            dxdt[self.fixed_indices] = 0.0

        return dxdt

    def dynamics_ivp(self, t: float, x: np.ndarray, regime: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute dx/dt for solve_ivp (arguments reversed)."""
        return self.dynamics(x, t, regime)

    def dynamics_batch(self, X: np.ndarray, t: float, regime: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute dx/dt for a batch of states (n_cells, n_genes).

        Vectorized equivalent of dynamics(). Uses sig @ W.T instead of W @ sig
        to handle the 2-D case correctly. ``regime`` is the per-cell component assignment,
        shape (n_cells, n_genes), held fixed instead of read from ``X``.
        """
        X_clipped = np.maximum(X, self.x_min)
        if self.x_max is not None:
            X_clipped = np.minimum(X_clipped, self.x_max)

        sig = sigmoid_regime(X_clipped, self.threshold, self.exponent,
                             self.threshold2, self.exponent2, regime=regime, a=self.mix, tau=self.active_min)  # (n_cells, n_genes)
        dxdt = sig @ self.W.T - self.gamma * X_clipped + self.I  # (n_cells, n_genes)

        at_lower = X <= self.x_min
        dxdt[at_lower] = np.maximum(dxdt[at_lower], 0)

        if self.x_max is not None:
            at_upper = X >= self.x_max
            dxdt[at_upper] = np.minimum(dxdt[at_upper], 0)

        if self.fixed_indices is not None and len(self.fixed_indices) > 0:
            dxdt[:, self.fixed_indices] = 0.0

        return dxdt

    def solve(
        self,
        x0: np.ndarray,
        t_span: np.ndarray,
        method: str = 'euler',
        clip_each_step: bool = True,
        regime: Optional[np.ndarray] = None
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
            - 'RK45' and the other scipy names: scipy.integrate.solve_ivp. With
              ``clip_each_step`` these run segment by segment between output times and
              project the state after each segment, which is the adaptive-step counterpart
              of the clipped Euler path.
        clip_each_step : bool, optional (default: True)
            Project the state onto the admissible box at each step, and re-impose the fixed
            genes. This enforces the non-negative range; it is not a stability property of
            the scheme.
        regime : np.ndarray, optional
            Component assignment per gene (1 selects component 2), held fixed for the whole
            trajectory. Pass the cell's observed-state assignment; if omitted, the
            nearest-threshold rule is read from the current state at every evaluation.

        Returns
        -------
        np.ndarray
            Solution trajectory (len(t_span) × n_genes)
        """
        # Ensure initial condition is valid
        x0 = self._clip(x0)
        self._enforce_fixed(x0)

        if method == 'euler':
            return self._solve_euler(x0, t_span, clip_each_step, regime)
        elif method == 'odeint':
            trajectory = odeint(self.dynamics, x0, t_span, args=(regime,))
            if clip_each_step:
                trajectory = self._clip_trajectory(trajectory)
            self._enforce_fixed_trajectory(trajectory)
            return trajectory
        elif method in ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']:
            return self._solve_ivp(x0, t_span, method, clip_each_step, regime)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'euler', 'odeint', or scipy method names.")

    def _solve_euler(
        self,
        x0: np.ndarray,
        t_span: np.ndarray,
        clip_each_step: bool = True,
        regime: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Solve ODE using Euler method with clipping at each step.

        This is more stable for stiff systems and ensures non-negativity.
        Fixed genes (if any) are held constant throughout the simulation.
        """
        n_steps = len(t_span)
        n_genes = len(x0)
        trajectory = np.zeros((n_steps, n_genes), dtype=np.float32)
        trajectory[0] = x0.copy()

        x = x0.copy()
        for i in range(1, n_steps):
            dt = t_span[i] - t_span[i-1]

            # Compute derivative
            dxdt = self.dynamics(x, t_span[i-1], regime)

            # Euler step
            x = x + dt * dxdt

            # Clip to valid range and enforce fixed genes
            if clip_each_step:
                x = self._clip(x)
            self._enforce_fixed(x)

            trajectory[i] = x

        return trajectory

    def _solve_ivp(
        self,
        x0: np.ndarray,
        t_span: np.ndarray,
        method: str,
        clip_each_step: bool,
        regime: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Solve using scipy solve_ivp, optionally projecting between output steps.

        With ``clip_each_step`` the integration is run segment by segment between consecutive
        entries of ``t_span``, and the state is projected onto the admissible box and the fixed
        genes re-imposed at the end of each segment, so the state that seeds the next segment is
        admissible. That is what makes this the adaptive-step counterpart of the clipped Euler
        path: the constraint acts on the state being integrated, not only on the samples that
        come back.

        Without it, a single solve runs over the whole interval and only the returned samples
        are clipped, which reports an in-range trajectory even when the solve left the range.
        """
        if not clip_each_step:
            sol = solve_ivp(self.dynamics_ivp, (t_span[0], t_span[-1]), x0,
                            method=method, t_eval=t_span, dense_output=False, args=(regime,))
            trajectory = sol.y.T
            self._enforce_fixed_trajectory(trajectory)
            return trajectory

        n_steps = len(t_span)
        trajectory = np.zeros((n_steps, len(x0)), dtype=np.float32)
        trajectory[0] = x0
        x = np.asarray(x0, dtype=float).copy()
        for i in range(1, n_steps):
            sol = solve_ivp(self.dynamics_ivp, (float(t_span[i - 1]), float(t_span[i])), x,
                            method=method, t_eval=[float(t_span[i])], dense_output=False,
                            args=(regime,))
            if not sol.success or sol.y.shape[1] == 0:
                # A failed segment is reported rather than silently carried forward as the
                # previous state, which would look like a converged trajectory.
                raise RuntimeError(
                    f"{method} failed on segment {i} of {n_steps - 1} "
                    f"(t={t_span[i - 1]:g} to {t_span[i]:g}): {sol.message}"
                )
            x = self._clip(sol.y[:, -1])
            self._enforce_fixed(x)
            trajectory[i] = x

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
    from ._utils import _get_W_matrix, _compute_x_bounds

    genes = get_genes_used(adata)

    W = _get_W_matrix(adata, cluster, use_cluster_specific=True)
    bias_vector = adata.var[f'I_{cluster}'].values[genes]

    gamma_key = f'gamma_{cluster}'
    gamma = adata.var[gamma_key].values[genes] if gamma_key in adata.var else adata.var[degradation_key].values[genes]

    threshold, exponent, threshold2, exponent2, mix = get_hill_params(adata, genes, with_mix=True)
    from .._utils.io import regime_rule
    active_min = None
    if regime_rule(adata) != 'posterior':
        mix = None
    elif 'sigmoid_active_min' in adata.var.columns:
        active_min = adata.var['sigmoid_active_min'].values[genes].astype(float)

    # Compute upper bounds from data
    if x_max_percentile is not None:
        if spliced_key is None:
            spliced_key = adata.uns.get('scHopfield', {}).get('spliced_key', 'Ms')
        X = to_numpy(get_matrix(adata, spliced_key, genes=genes))
        _, x_max = _compute_x_bounds(X, x_max_percentile, multiplier=2.0)
    else:
        x_max = None

    return ODESolver(W, bias_vector, gamma, threshold, exponent,
                     threshold2, exponent2, x_min=0.0, x_max=x_max, mix=mix,
                     active_min=active_min)
