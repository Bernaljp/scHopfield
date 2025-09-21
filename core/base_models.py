"""
Base classes and interfaces for the scHopfield package.
Defines common interfaces and abstract base classes.
"""

from abc import ABC, abstractmethod
import numpy as np
import anndata
from typing import Union, List, Dict, Any


class BaseAnalyzer(ABC):
    """
    Abstract base class for all analyzer components in scHopfield.
    Defines common interface and shared functionality.
    """

    def __init__(self, adata: anndata.AnnData):
        """
        Initialize the analyzer with AnnData object.

        Parameters:
            adata: AnnData object containing the data
        """
        self.adata = adata
        self._validate_data()

    def _validate_data(self):
        """Validate the input AnnData object."""
        if not isinstance(self.adata, anndata.AnnData):
            raise TypeError("Input must be an AnnData object")

        if self.adata.n_obs == 0:
            raise ValueError("AnnData object cannot be empty")

    @abstractmethod
    def compute(self, **kwargs) -> Any:
        """
        Abstract method for computation. Must be implemented by subclasses.
        """
        pass


class BaseEnergyCalculator(BaseAnalyzer):
    """
    Base class for energy calculation components.
    """

    @abstractmethod
    def calculate_energy(self, x: np.ndarray, cluster: str = 'all') -> np.ndarray:
        """
        Calculate energy for given expression values.

        Parameters:
            x: Expression matrix or values
            cluster: Cluster identifier

        Returns:
            Calculated energy values
        """
        pass


class BaseOptimizer(ABC):
    """
    Base class for optimization components.
    """

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Fit the model to data.

        Parameters:
            x: Input data
            y: Target data
            **kwargs: Additional fitting parameters

        Returns:
            Fitting results
        """
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model.

        Parameters:
            x: Input data

        Returns:
            Predictions
        """
        pass


class BaseSimulator(ABC):
    """
    Base class for simulation components.
    """

    @abstractmethod
    def simulate(self, initial_conditions: np.ndarray,
                 time_points: np.ndarray, **kwargs) -> np.ndarray:
        """
        Simulate system dynamics.

        Parameters:
            initial_conditions: Initial state values
            time_points: Time points for simulation
            **kwargs: Additional simulation parameters

        Returns:
            Simulation results
        """
        pass


class ConfigMixin:
    """
    Mixin class for handling configuration and parameters.
    """

    def __init__(self):
        self._config = {}

    def set_config(self, **kwargs):
        """Set configuration parameters."""
        self._config.update(kwargs)

    def get_config(self, key: str, default=None):
        """Get configuration parameter."""
        return self._config.get(key, default)

    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration parameters."""
        return self._config.copy()


class ValidationMixin:
    """
    Mixin class for common validation operations.
    """

    @staticmethod
    def validate_array(arr: np.ndarray, name: str = "array",
                      min_dims: int = 1, max_dims: int = None):
        """
        Validate numpy array properties.

        Parameters:
            arr: Array to validate
            name: Name for error messages
            min_dims: Minimum number of dimensions
            max_dims: Maximum number of dimensions (None for no limit)
        """
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be a numpy array")

        if arr.ndim < min_dims:
            raise ValueError(f"{name} must have at least {min_dims} dimensions")

        if max_dims is not None and arr.ndim > max_dims:
            raise ValueError(f"{name} must have at most {max_dims} dimensions")

        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            raise ValueError(f"{name} contains invalid values (NaN or inf)")

    @staticmethod
    def validate_genes(genes: Union[None, List[str], List[bool], List[int]],
                      total_genes: int) -> np.ndarray:
        """
        Validate and convert gene specification to indices.

        Parameters:
            genes: Gene specification
            total_genes: Total number of genes available

        Returns:
            Array of gene indices
        """
        if genes is None:
            return np.arange(total_genes)

        if isinstance(genes, list):
            if len(genes) == 0:
                raise ValueError("Gene list cannot be empty")

            # Boolean mask
            if isinstance(genes[0], bool):
                if len(genes) != total_genes:
                    raise ValueError("Boolean gene mask must match total gene count")
                return np.where(genes)[0]

            # Integer indices
            elif isinstance(genes[0], (int, np.integer)):
                indices = np.array(genes)
                if np.any(indices < 0) or np.any(indices >= total_genes):
                    raise ValueError("Gene indices out of range")
                return indices

            # String names - need to be handled by caller with gene name mapping
            elif isinstance(genes[0], str):
                return genes  # Return as-is for caller to resolve

        raise TypeError("Genes must be None, list of strings, booleans, or integers")