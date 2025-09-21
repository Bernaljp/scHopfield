"""
Energy-based optimization functionality for scHopfield package.
Contains optimizers that work with energy landscapes and minimization.
"""

import numpy as np
import torch
from scipy.optimize import minimize
from typing import Callable, Optional, Union, Dict, Any

try:
    from ..core.base_models import BaseOptimizer
    from ..utils.utilities import to_numpy
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.base_models import BaseOptimizer
    from utils.utilities import to_numpy


class EnergyOptimizer(BaseOptimizer):
    """
    Optimizer for energy landscape minimization and trajectory optimization.

    This class provides methods for finding energy minima, optimizing trajectories,
    and performing gradient-based optimization on energy landscapes.
    """

    def __init__(self, energy_function: Callable, gradient_function: Optional[Callable] = None):
        """
        Initialize the energy optimizer.

        Args:
            energy_function: Function that computes energy given a state
            gradient_function: Optional function that computes energy gradients
        """
        self.energy_function = energy_function
        self.gradient_function = gradient_function

    def optimize(self, x0: np.ndarray, method: str = 'L-BFGS-B', **kwargs) -> Dict[str, Any]:
        """
        Optimize the energy function starting from initial state x0.

        Args:
            x0: Initial state for optimization
            method: Optimization method to use
            **kwargs: Additional arguments for the optimizer

        Returns:
            Dictionary containing optimization results
        """
        # Set up objective function
        def objective(x):
            return self.energy_function(x)

        # Set up gradient function if available
        jac = self.gradient_function if self.gradient_function is not None else None

        # Perform optimization
        result = minimize(objective, x0, method=method, jac=jac, **kwargs)

        return {
            'success': result.success,
            'final_state': result.x,
            'final_energy': result.fun,
            'iterations': result.nit,
            'message': result.message
        }

    def find_local_minima(self, x0_list: list, method: str = 'L-BFGS-B', **kwargs) -> list:
        """
        Find local minima starting from multiple initial conditions.

        Args:
            x0_list: List of initial states
            method: Optimization method to use
            **kwargs: Additional arguments for the optimizer

        Returns:
            List of optimization results for each initial condition
        """
        results = []
        for x0 in x0_list:
            result = self.optimize(x0, method=method, **kwargs)
            results.append(result)
        return results

    def gradient_descent(self, x0: np.ndarray, learning_rate: float = 0.01,
                        max_iterations: int = 1000, tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Perform gradient descent optimization.

        Args:
            x0: Initial state
            learning_rate: Step size for gradient descent
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance

        Returns:
            Dictionary containing optimization results
        """
        if self.gradient_function is None:
            raise ValueError("Gradient function required for gradient descent")

        x = x0.copy()
        trajectory = [x.copy()]
        energies = [self.energy_function(x)]

        for i in range(max_iterations):
            grad = self.gradient_function(x)
            x_new = x - learning_rate * grad

            energy_new = self.energy_function(x_new)
            trajectory.append(x_new.copy())
            energies.append(energy_new)

            # Check for convergence
            if np.linalg.norm(grad) < tolerance:
                break

            x = x_new

        return {
            'success': True,
            'final_state': x,
            'final_energy': energies[-1],
            'iterations': i + 1,
            'trajectory': np.array(trajectory),
            'energies': np.array(energies)
        }


class TrajectoryOptimizer(BaseOptimizer):
    """
    Optimizer for trajectory inference and path optimization.

    This class provides methods for optimizing trajectories between cell states,
    finding optimal paths through energy landscapes, and trajectory smoothing.
    """

    def __init__(self, landscape_analyzer):
        """
        Initialize the trajectory optimizer.

        Args:
            landscape_analyzer: Reference to the main LandscapeAnalyzer instance
        """
        self.analyzer = landscape_analyzer

    def optimize(self, start_state: np.ndarray, end_state: np.ndarray,
                n_points: int = 50, method: str = 'geodesic') -> Dict[str, Any]:
        """
        Optimize a trajectory between two cell states.

        Args:
            start_state: Starting cell state
            end_state: Target cell state
            n_points: Number of points along the trajectory
            method: Optimization method ('geodesic', 'energy_minimizing', 'linear')

        Returns:
            Dictionary containing optimized trajectory and metadata
        """
        if method == 'linear':
            return self._linear_trajectory(start_state, end_state, n_points)
        elif method == 'geodesic':
            return self._geodesic_trajectory(start_state, end_state, n_points)
        elif method == 'energy_minimizing':
            return self._energy_minimizing_trajectory(start_state, end_state, n_points)
        else:
            raise ValueError(f"Unknown trajectory optimization method: {method}")

    def _linear_trajectory(self, start: np.ndarray, end: np.ndarray, n_points: int) -> Dict[str, Any]:
        """Create a linear interpolation trajectory."""
        t = np.linspace(0, 1, n_points)
        trajectory = np.array([start + t_i * (end - start) for t_i in t])

        return {
            'trajectory': trajectory,
            'method': 'linear',
            'energy_profile': self._compute_energy_profile(trajectory)
        }

    def _geodesic_trajectory(self, start: np.ndarray, end: np.ndarray, n_points: int) -> Dict[str, Any]:
        """Create a geodesic trajectory (simplified implementation)."""
        # This is a simplified implementation - would need more sophisticated geodesic computation
        return self._linear_trajectory(start, end, n_points)

    def _energy_minimizing_trajectory(self, start: np.ndarray, end: np.ndarray, n_points: int) -> Dict[str, Any]:
        """Create an energy-minimizing trajectory."""
        # Initialize with linear trajectory
        initial_trajectory = self._linear_trajectory(start, end, n_points)['trajectory']

        # Optimize trajectory to minimize total energy
        def objective(trajectory_flat):
            trajectory = trajectory_flat.reshape(n_points, len(start))
            # Ensure endpoints remain fixed
            trajectory[0] = start
            trajectory[-1] = end
            return np.sum(self._compute_energy_profile(trajectory))

        # Flatten trajectory for optimization
        trajectory_flat = initial_trajectory.flatten()

        # Perform optimization
        result = minimize(objective, trajectory_flat, method='L-BFGS-B')

        optimized_trajectory = result.x.reshape(n_points, len(start))
        # Ensure endpoints are exactly preserved
        optimized_trajectory[0] = start
        optimized_trajectory[-1] = end

        return {
            'trajectory': optimized_trajectory,
            'method': 'energy_minimizing',
            'energy_profile': self._compute_energy_profile(optimized_trajectory),
            'optimization_success': result.success
        }

    def _compute_energy_profile(self, trajectory: np.ndarray) -> np.ndarray:
        """Compute energy profile along a trajectory."""
        energies = []
        for state in trajectory:
            # Use the analyzer's energy calculation methods
            # This is a simplified implementation
            energy = np.sum(state**2)  # Placeholder - would use actual energy calculation
            energies.append(energy)
        return np.array(energies)

    def smooth_trajectory(self, trajectory: np.ndarray, smoothing_factor: float = 0.1) -> np.ndarray:
        """
        Smooth a trajectory using regularization.

        Args:
            trajectory: Input trajectory to smooth
            smoothing_factor: Smoothing strength

        Returns:
            Smoothed trajectory
        """
        # Simple smoothing using weighted averages
        smoothed = trajectory.copy()
        for i in range(1, len(trajectory) - 1):
            smoothed[i] = (1 - smoothing_factor) * trajectory[i] + \
                         smoothing_factor * 0.5 * (trajectory[i-1] + trajectory[i+1])
        return smoothed

    def compute_trajectory_length(self, trajectory: np.ndarray) -> float:
        """
        Compute the total length of a trajectory.

        Args:
            trajectory: Input trajectory

        Returns:
            Total trajectory length
        """
        diffs = np.diff(trajectory, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        return np.sum(distances)

    def resample_trajectory(self, trajectory: np.ndarray, n_points: int) -> np.ndarray:
        """
        Resample a trajectory to have a specific number of points.

        Args:
            trajectory: Input trajectory
            n_points: Desired number of points

        Returns:
            Resampled trajectory
        """
        # Compute cumulative distances along trajectory
        diffs = np.diff(trajectory, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        cumulative_distances = np.concatenate([[0], np.cumsum(distances)])

        # Create uniform sampling points
        total_length = cumulative_distances[-1]
        sample_points = np.linspace(0, total_length, n_points)

        # Interpolate trajectory at sample points
        resampled = np.zeros((n_points, trajectory.shape[1]))
        for i in range(trajectory.shape[1]):
            resampled[:, i] = np.interp(sample_points, cumulative_distances, trajectory[:, i])

        return resampled