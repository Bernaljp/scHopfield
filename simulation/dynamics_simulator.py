"""
Advanced dynamics simulation functionality for scHopfield package.
Contains specialized simulators for complex dynamical behavior analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy.integrate import odeint
from scipy.optimize import minimize

from ..core.base_models import BaseSimulator
from ..utils.utilities import to_numpy, sigmoid
from .ode_solver import ODESolver


class AttractorAnalyzer(BaseSimulator):
    """
    Analyzer for finding and characterizing attractors in the dynamical system.

    This class provides methods for identifying fixed points, limit cycles,
    and other attracting sets in the phase space.
    """

    def __init__(self, landscape_analyzer):
        """
        Initialize the attractor analyzer.

        Args:
            landscape_analyzer: Reference to the main LandscapeAnalyzer instance
        """
        self.analyzer = landscape_analyzer

    def simulate(self, initial_state: np.ndarray, time_points: np.ndarray, **kwargs) -> np.ndarray:
        """
        Simulate the system dynamics.

        Args:
            initial_state: Initial system state
            time_points: Array of time points for simulation
            **kwargs: Additional simulation parameters

        Returns:
            System trajectory over time
        """
        cluster = kwargs.get('cluster', 'all')
        dynamics_sim = DynamicsSimulator(self.analyzer)
        return dynamics_sim.simulate(initial_state, time_points, cluster=cluster)

    def find_attractors(self, cluster: str = 'all', n_initial_conditions: int = 50,
                       simulation_time: float = 100.0, tolerance: float = 1e-4) -> Dict[str, List]:
        """
        Find attractors by simulating from multiple initial conditions.

        Args:
            cluster: Cluster to analyze
            n_initial_conditions: Number of random initial conditions to try
            simulation_time: Duration of simulation to reach attractors
            tolerance: Tolerance for identifying unique attractors

        Returns:
            Dictionary containing different types of attractors
        """
        # Get expression range for generating initial conditions
        expression_data = self.analyzer.get_matrix(self.analyzer.spliced_matrix_key, genes=self.analyzer.genes)
        min_expr = np.min(expression_data, axis=0)
        max_expr = np.max(expression_data, axis=0)

        fixed_points = []
        limit_cycles = []
        other_attractors = []

        time_points = np.linspace(0, simulation_time, 1000)

        for _ in range(n_initial_conditions):
            # Generate random initial condition
            x0 = np.random.uniform(min_expr, max_expr)

            # Simulate to approach attractor
            trajectory = self.simulate(x0, time_points, cluster=cluster)

            # Analyze the final portion of the trajectory
            final_portion = trajectory[:, -200:]  # Last 200 time points
            attractor_type, attractor_data = self._classify_attractor(final_portion, tolerance)

            # Check if this attractor is already found
            is_new = self._is_new_attractor(attractor_data, fixed_points + limit_cycles + other_attractors, tolerance)

            if is_new:
                if attractor_type == 'fixed_point':
                    fixed_points.append(attractor_data)
                elif attractor_type == 'limit_cycle':
                    limit_cycles.append(attractor_data)
                else:
                    other_attractors.append(attractor_data)

        return {
            'fixed_points': fixed_points,
            'limit_cycles': limit_cycles,
            'other_attractors': other_attractors
        }

    def _classify_attractor(self, trajectory: np.ndarray, tolerance: float) -> Tuple[str, np.ndarray]:
        """
        Classify the type of attractor based on trajectory behavior.

        Args:
            trajectory: Final portion of trajectory
            tolerance: Tolerance for classification

        Returns:
            Tuple of (attractor_type, attractor_data)
        """
        # Check if it's a fixed point
        final_state = trajectory[:, -1]
        variation = np.std(trajectory, axis=1)

        if np.max(variation) < tolerance:
            return 'fixed_point', final_state

        # Check for periodicity (simple limit cycle detection)
        # This is a simplified implementation - more sophisticated methods exist
        autocorr = self._compute_autocorrelation(trajectory)
        if self._has_periodic_peaks(autocorr, tolerance):
            return 'limit_cycle', trajectory

        return 'chaotic_or_other', trajectory

    def _compute_autocorrelation(self, trajectory: np.ndarray) -> np.ndarray:
        """Compute autocorrelation of trajectory."""
        # Simplified autocorrelation computation
        n_points = trajectory.shape[1]
        autocorr = np.zeros(n_points // 2)

        for gene in range(trajectory.shape[0]):
            signal = trajectory[gene, :]
            signal = signal - np.mean(signal)  # Remove mean
            corr = np.correlate(signal, signal, mode='full')
            corr = corr[corr.size // 2:]
            autocorr += corr[:len(autocorr)]

        return autocorr / trajectory.shape[0]

    def _has_periodic_peaks(self, autocorr: np.ndarray, tolerance: float) -> bool:
        """Check if autocorrelation has periodic peaks indicating limit cycle."""
        # Find peaks in autocorrelation
        peaks = []
        for i in range(1, len(autocorr) - 1):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                if autocorr[i] > tolerance:
                    peaks.append(i)

        # Check if peaks are regularly spaced
        if len(peaks) > 2:
            spacings = np.diff(peaks)
            return np.std(spacings) < tolerance * np.mean(spacings)

        return False

    def _is_new_attractor(self, attractor_data: np.ndarray, existing_attractors: List,
                         tolerance: float) -> bool:
        """Check if an attractor is new or already discovered."""
        if len(existing_attractors) == 0:
            return True

        for existing in existing_attractors:
            if attractor_data.shape == existing.shape:
                if np.linalg.norm(attractor_data - existing) < tolerance:
                    return False
            else:
                # For different shaped attractors (e.g., fixed point vs trajectory)
                # Use a different comparison method
                if hasattr(existing, 'shape') and len(existing.shape) == 1:  # Fixed point
                    if len(attractor_data.shape) == 1:  # Also fixed point
                        if np.linalg.norm(attractor_data - existing) < tolerance:
                            return False

        return True

    def analyze_stability(self, fixed_point: np.ndarray, cluster: str = 'all') -> Dict[str, Any]:
        """
        Analyze the stability of a fixed point using linearization.

        Args:
            fixed_point: The fixed point to analyze
            cluster: Cluster to use for analysis

        Returns:
            Dictionary containing stability analysis results
        """
        # Compute Jacobian at the fixed point
        jacobian = self._compute_jacobian(fixed_point, cluster)

        # Compute eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(jacobian)

        # Classify stability
        real_parts = np.real(eigenvalues)
        stability = 'stable' if np.all(real_parts < 0) else 'unstable'
        if np.any(real_parts == 0):
            stability = 'marginally_stable'

        return {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'stability': stability,
            'jacobian': jacobian
        }

    def _compute_jacobian(self, state: np.ndarray, cluster: str) -> np.ndarray:
        """Compute Jacobian matrix at a given state."""
        W = self.analyzer.W[cluster]
        gamma = (self.analyzer.adata.var[self.analyzer.gamma_key][self.analyzer.genes].values
                if not self.analyzer.refit_gamma else self.analyzer.gamma[cluster])

        # Compute sigmoid derivative
        sig = sigmoid(state, self.analyzer.threshold, self.analyzer.exponent)
        sig_prime = (self.analyzer.exponent * sig * (1 - sig) /
                    np.maximum(state, 1e-8))  # Avoid division by zero

        # Jacobian: W * diag(sig_prime) - diag(gamma)
        jacobian = W * sig_prime[None, :] - np.diag(gamma)

        return jacobian


class EnergySimulator(BaseSimulator):
    """
    Simulator for energy-based analysis of system dynamics.

    This class provides methods for analyzing energy changes during simulation
    and understanding energy landscape properties.
    """

    def __init__(self, landscape_analyzer):
        """
        Initialize the energy simulator.

        Args:
            landscape_analyzer: Reference to the main LandscapeAnalyzer instance
        """
        self.analyzer = landscape_analyzer

    def simulate(self, initial_state: np.ndarray, time_points: np.ndarray,
                cluster: str = 'all', **kwargs) -> np.ndarray:
        """
        Simulate system dynamics.

        Args:
            initial_state: Initial system state
            time_points: Array of time points for simulation
            cluster: Cluster to use for simulation
            **kwargs: Additional simulation parameters

        Returns:
            System trajectory over time
        """
        dynamics_sim = DynamicsSimulator(self.analyzer)
        return dynamics_sim.simulate(initial_state, time_points, cluster=cluster, **kwargs)

    def simulate_with_energy(self, initial_state: np.ndarray, time_points: np.ndarray,
                           cluster: str = 'all', **kwargs) -> Dict[str, np.ndarray]:
        """
        Simulate system and compute energy along trajectory.

        Args:
            initial_state: Initial system state
            time_points: Array of time points for simulation
            cluster: Cluster to use for simulation
            **kwargs: Additional simulation parameters

        Returns:
            Dictionary containing trajectory and energy profiles
        """
        # Simulate trajectory
        trajectory = self.simulate(initial_state, time_points, cluster=cluster, **kwargs)

        # Compute energy at each time point
        n_timepoints = trajectory.shape[1]
        total_energy = np.zeros(n_timepoints)
        interaction_energy = np.zeros(n_timepoints)
        degradation_energy = np.zeros(n_timepoints)
        bias_energy = np.zeros(n_timepoints)

        for i in range(n_timepoints):
            state = trajectory[:, i]
            energies = self.analyzer.get_energies(state.reshape(1, -1))

            if energies is not None:
                E, E_int, E_deg, E_bias = energies
                total_energy[i] = E[cluster][0]
                interaction_energy[i] = E_int[cluster][0]
                degradation_energy[i] = E_deg[cluster][0]
                bias_energy[i] = E_bias[cluster][0]

        return {
            'trajectory': trajectory,
            'time_points': time_points,
            'total_energy': total_energy,
            'interaction_energy': interaction_energy,
            'degradation_energy': degradation_energy,
            'bias_energy': bias_energy
        }

    def find_energy_minima(self, cluster: str = 'all', n_random_starts: int = 50) -> List[Dict]:
        """
        Find energy minima by optimizing the energy function.

        Args:
            cluster: Cluster to analyze
            n_random_starts: Number of random starting points

        Returns:
            List of energy minima with their properties
        """
        # Get expression range for random starting points
        expression_data = self.analyzer.get_matrix(self.analyzer.spliced_matrix_key, genes=self.analyzer.genes)
        min_expr = np.min(expression_data, axis=0)
        max_expr = np.max(expression_data, axis=0)

        minima = []

        def energy_function(state):
            """Energy function to minimize."""
            energies = self.analyzer.get_energies(state.reshape(1, -1))
            if energies is not None:
                E, _, _, _ = energies
                return E[cluster][0]
            return float('inf')

        for _ in range(n_random_starts):
            # Generate random starting point
            x0 = np.random.uniform(min_expr, max_expr)

            # Minimize energy
            result = minimize(energy_function, x0, method='L-BFGS-B')

            if result.success:
                minimum = {
                    'state': result.x,
                    'energy': result.fun,
                    'success': result.success
                }

                # Check if this minimum is new
                is_new = True
                for existing in minima:
                    if np.linalg.norm(result.x - existing['state']) < 1e-4:
                        is_new = False
                        break

                if is_new:
                    minima.append(minimum)

        return minima

    def compute_energy_barrier(self, state1: np.ndarray, state2: np.ndarray,
                              cluster: str = 'all', n_points: int = 50) -> Dict[str, Any]:
        """
        Compute energy barrier between two states.

        Args:
            state1: First state
            state2: Second state
            cluster: Cluster to analyze
            n_points: Number of points along the path

        Returns:
            Dictionary containing barrier height and path information
        """
        # Create linear path between states
        t = np.linspace(0, 1, n_points)
        path = np.array([state1 + t_i * (state2 - state1) for t_i in t])

        # Compute energy along path
        energies = []
        for state in path:
            energy_result = self.analyzer.get_energies(state.reshape(1, -1))
            if energy_result is not None:
                E, _, _, _ = energy_result
                energies.append(E[cluster][0])
            else:
                energies.append(float('inf'))

        energies = np.array(energies)

        # Find barrier height
        barrier_height = np.max(energies) - min(energies[0], energies[-1])
        barrier_position = np.argmax(energies)

        return {
            'barrier_height': barrier_height,
            'barrier_position': barrier_position,
            'path': path,
            'energies': energies,
            'barrier_state': path[barrier_position]
        }