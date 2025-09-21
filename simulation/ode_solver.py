"""
ODE solver functionality for scHopfield package.
Contains classes for solving ordinary differential equations in the Hopfield system.
"""

import numpy as np
import scipy.integrate as solve
try:
    import sdeint
    HAS_SDEINT = True
except ImportError:
    HAS_SDEINT = False
from typing import Optional, Callable, Union

try:
    from ..core.base_models import BaseSimulator
    from ..utils.utilities import sigmoid
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.base_models import BaseSimulator
    from utils.utilities import sigmoid


class ODESolver(BaseSimulator):
    """
    ODE solver for the Hopfield-like dynamical system.

    This class provides methods for solving ordinary and stochastic differential
    equations that govern the dynamics of the gene regulatory network.
    """

    def __init__(self, W: np.ndarray, gamma: np.ndarray, I: np.ndarray,
                 k: np.ndarray, n: np.ndarray, solver: Optional[Callable] = None,
                 clipping: bool = False, method: Optional[str] = None):
        """
        Initialize the ODE solver.

        Args:
            W: Interaction matrix
            gamma: Degradation rates
            I: Bias vector
            k: Sigmoid thresholds
            n: Sigmoid exponents
            solver: Custom solver function
            clipping: Whether to clip negative values
            method: Integration method for scipy
        """
        if method is not None:
            assert method in ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'], 'Invalid method'
            self.method = method
        else:
            self.method = 'RK45'

        self.W = np.array(W)
        self.gamma = np.array(gamma)
        self.I = np.array(I)
        self.k = np.array(k)
        self.n = n
        self.clip = clipping
        self.solver = solve.solve_ivp if solver is None else solver

    @staticmethod
    def sigmoid(x: np.ndarray, k: np.ndarray, n: np.ndarray) -> np.ndarray:
        """
        Compute sigmoid activation function.

        Args:
            x: Input values
            k: Threshold parameters
            n: Exponent parameters

        Returns:
            Sigmoid-transformed values
        """
        # Avoid division by zero and ensure numerical stability
        x_safe = np.clip(x, 1e-5, np.inf)
        return x_safe**n / (x_safe**n + k**n)

    def ode_system(self) -> Callable:
        """
        Create the ODE system function.

        Returns:
            Function that computes derivatives for the ODE system
        """
        def f(t: float, y: np.ndarray) -> np.ndarray:
            s = self.sigmoid(y, self.k, self.n)
            dydt = self.W @ s - self.gamma * y + self.I
            return np.where(np.logical_and(y <= 0, dydt < 0), 0, dydt) if self.clip else dydt
        return f

    def simulate_system(self, x0: np.ndarray, tf: float, n_steps: int, noise: float = 0) -> np.ndarray:
        """
        Simulate the dynamical system.

        Args:
            x0: Initial conditions
            tf: Final time
            n_steps: Number of time steps
            noise: Noise level for SDE simulation

        Returns:
            Time series of system states
        """
        t = np.linspace(0, tf, n_steps)
        f = self.ode_system()

        if noise == 0:
            sol = self.solver(f, [0, tf], x0, t_eval=t, method=self.method)
            return sol.y
        else:
            if not HAS_SDEINT:
                raise ImportError("sdeint package required for stochastic simulation")

            def G(x: np.ndarray, t: float) -> np.ndarray:
                return noise * np.eye(len(x0))

            sol = sdeint.itoint(f, G, x0, t)
            return sol.T

    def simulate(self, initial_state: np.ndarray, time_points: np.ndarray, **kwargs) -> np.ndarray:
        """
        Simulate the system over specified time points.

        Args:
            initial_state: Initial system state
            time_points: Array of time points for simulation
            **kwargs: Additional simulation parameters

        Returns:
            System trajectory over time
        """
        tf = time_points[-1]
        n_steps = len(time_points)
        noise = kwargs.get('noise', 0)

        return self.simulate_system(initial_state, tf, n_steps, noise)


class StochasticSimulator(BaseSimulator):
    """
    Stochastic simulator for the Hopfield-like system with noise.

    This class extends the basic ODE solver to include various types of noise
    and stochastic effects in the simulation.
    """

    def __init__(self, ode_solver: ODESolver):
        """
        Initialize the stochastic simulator.

        Args:
            ode_solver: Base ODE solver instance
        """
        self.ode_solver = ode_solver

    def simulate(self, initial_state: np.ndarray, time_points: np.ndarray,
                noise_type: str = 'additive', noise_strength: float = 0.1, **kwargs) -> np.ndarray:
        """
        Simulate the system with stochastic effects.

        Args:
            initial_state: Initial system state
            time_points: Array of time points for simulation
            noise_type: Type of noise ('additive', 'multiplicative')
            noise_strength: Strength of noise
            **kwargs: Additional simulation parameters

        Returns:
            Stochastic system trajectory
        """
        if not HAS_SDEINT:
            # Fallback to deterministic simulation with post-hoc noise
            trajectory = self.ode_solver.simulate(initial_state, time_points, **kwargs)
            if noise_type == 'additive':
                noise = np.random.normal(0, noise_strength, trajectory.shape)
                trajectory += noise
            elif noise_type == 'multiplicative':
                noise = np.random.normal(1, noise_strength, trajectory.shape)
                trajectory *= noise
            return trajectory
        else:
            # Use proper SDE integration
            return self.ode_solver.simulate(initial_state, time_points, noise=noise_strength, **kwargs)

    def ensemble_simulation(self, initial_state: np.ndarray, time_points: np.ndarray,
                           n_realizations: int = 100, **kwargs) -> np.ndarray:
        """
        Perform ensemble simulation with multiple noise realizations.

        Args:
            initial_state: Initial system state
            time_points: Array of time points for simulation
            n_realizations: Number of stochastic realizations
            **kwargs: Additional simulation parameters

        Returns:
            Array of shape (n_realizations, n_timepoints, n_genes) containing all trajectories
        """
        trajectories = []
        for _ in range(n_realizations):
            trajectory = self.simulate(initial_state, time_points, **kwargs)
            trajectories.append(trajectory.T)  # Transpose to get (n_timepoints, n_genes)

        return np.array(trajectories)

    def compute_statistics(self, trajectories: np.ndarray) -> dict:
        """
        Compute statistics from ensemble trajectories.

        Args:
            trajectories: Array of trajectories from ensemble simulation

        Returns:
            Dictionary containing mean, std, and confidence intervals
        """
        mean_trajectory = np.mean(trajectories, axis=0)
        std_trajectory = np.std(trajectories, axis=0)

        # Compute confidence intervals (95%)
        percentile_2_5 = np.percentile(trajectories, 2.5, axis=0)
        percentile_97_5 = np.percentile(trajectories, 97.5, axis=0)

        return {
            'mean': mean_trajectory,
            'std': std_trajectory,
            'ci_lower': percentile_2_5,
            'ci_upper': percentile_97_5,
            'all_trajectories': trajectories
        }


class DynamicsSimulator(BaseSimulator):
    """
    High-level simulator for studying system dynamics and behavior.

    This class provides convenient methods for common simulation tasks
    such as finding attractors, studying stability, and phase portraits.
    """

    def __init__(self, landscape_analyzer):
        """
        Initialize the dynamics simulator.

        Args:
            landscape_analyzer: Reference to the main LandscapeAnalyzer instance
        """
        self.analyzer = landscape_analyzer

    def simulate(self, initial_state: np.ndarray, time_points: np.ndarray,
                cluster: str = 'all', **kwargs) -> np.ndarray:
        """
        Simulate system dynamics for a specific cluster.

        Args:
            initial_state: Initial system state
            time_points: Array of time points for simulation
            cluster: Cluster to use for simulation parameters
            **kwargs: Additional simulation parameters

        Returns:
            System trajectory over time
        """
        # Get parameters for the specified cluster
        W = self.analyzer.W[cluster]
        gamma = (self.analyzer.adata.var[self.analyzer.gamma_key][self.analyzer.genes].values
                if not self.analyzer.refit_gamma else self.analyzer.gamma[cluster])
        I = self.analyzer.I[cluster]
        k = self.analyzer.threshold
        n = self.analyzer.exponent

        # Create ODE solver
        solver = ODESolver(W, gamma, I, k, n, **kwargs)

        # Simulate
        return solver.simulate(initial_state, time_points, **kwargs)

    def find_fixed_points(self, cluster: str = 'all', n_random_starts: int = 100,
                         tolerance: float = 1e-6) -> list:
        """
        Find fixed points of the dynamical system.

        Args:
            cluster: Cluster to analyze
            n_random_starts: Number of random starting points
            tolerance: Convergence tolerance

        Returns:
            List of found fixed points
        """
        # Get typical expression range for random starting points
        expression_data = self.analyzer.get_matrix(self.analyzer.spliced_matrix_key, genes=self.analyzer.genes)
        min_expr = np.min(expression_data, axis=0)
        max_expr = np.max(expression_data, axis=0)

        fixed_points = []

        for _ in range(n_random_starts):
            # Generate random starting point
            x0 = np.random.uniform(min_expr, max_expr)

            # Simulate for long time to approach steady state
            time_points = np.linspace(0, 100, 1000)
            trajectory = self.simulate(x0, time_points, cluster=cluster)

            # Check if final state is a fixed point
            final_state = trajectory[:, -1]

            # Simulate a bit more to check stability
            final_time_points = np.linspace(0, 10, 100)
            final_trajectory = self.simulate(final_state, final_time_points, cluster=cluster)

            # Check if state remained stable
            if np.max(np.std(final_trajectory, axis=1)) < tolerance:
                # Check if this fixed point is already found
                is_new = True
                for fp in fixed_points:
                    if np.linalg.norm(final_state - fp) < tolerance:
                        is_new = False
                        break

                if is_new:
                    fixed_points.append(final_state)

        return fixed_points

    def compute_flow_field(self, cluster: str = 'all', grid_resolution: int = 20,
                          gene_indices: tuple = (0, 1)) -> dict:
        """
        Compute flow field for visualization of system dynamics.

        Args:
            cluster: Cluster to analyze
            grid_resolution: Resolution of the grid
            gene_indices: Indices of genes to use for 2D projection

        Returns:
            Dictionary containing grid coordinates and flow vectors
        """
        # Get expression range for the selected genes
        expression_data = self.analyzer.get_matrix(self.analyzer.spliced_matrix_key, genes=self.analyzer.genes)
        gene1_range = [np.min(expression_data[:, gene_indices[0]]), np.max(expression_data[:, gene_indices[0]])]
        gene2_range = [np.min(expression_data[:, gene_indices[1]]), np.max(expression_data[:, gene_indices[1]])]

        # Create grid
        x = np.linspace(gene1_range[0], gene1_range[1], grid_resolution)
        y = np.linspace(gene2_range[0], gene2_range[1], grid_resolution)
        X, Y = np.meshgrid(x, y)

        # Initialize flow field arrays
        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        # Get system parameters
        W = self.analyzer.W[cluster]
        gamma = (self.analyzer.adata.var[self.analyzer.gamma_key][self.analyzer.genes].values
                if not self.analyzer.refit_gamma else self.analyzer.gamma[cluster])
        I = self.analyzer.I[cluster]

        # Compute flow at each grid point
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                # Create state vector (use mean values for other genes)
                state = np.mean(expression_data, axis=0)
                state[gene_indices[0]] = X[i, j]
                state[gene_indices[1]] = Y[i, j]

                # Compute derivatives
                sig = sigmoid(state, self.analyzer.threshold, self.analyzer.exponent)
                dydt = W @ sig - gamma * state + I

                # Store flow components for selected genes
                U[i, j] = dydt[gene_indices[0]]
                V[i, j] = dydt[gene_indices[1]]

        return {
            'X': X,
            'Y': Y,
            'U': U,
            'V': V,
            'gene_indices': gene_indices
        }