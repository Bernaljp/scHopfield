"""ODE solver for gene regulatory network dynamics."""

import numpy as np
from scipy.integrate import odeint
from typing import Optional, Callable
from anndata import AnnData

from .._utils.math import sigmoid
from .._utils.io import get_genes_used


class ODESolver:
    """
    ODE solver for Hopfield network dynamics.
    
    Solves: dx/dt = W * sigmoid(x) - gamma * x + I
    """
    
    def __init__(
        self,
        W: np.ndarray,
        I: np.ndarray,
        gamma: np.ndarray,
        threshold: np.ndarray,
        exponent: np.ndarray
    ):
        self.W = W
        self.I = I
        self.gamma = gamma
        self.threshold = threshold
        self.exponent = exponent
    
    def dynamics(self, x: np.ndarray, t: float) -> np.ndarray:
        """Compute dx/dt."""
        sig = sigmoid(x, self.threshold, self.exponent)
        return self.W @ sig - self.gamma * x + self.I
    
    def solve(
        self,
        x0: np.ndarray,
        t_span: np.ndarray
    ) -> np.ndarray:
        """
        Solve ODE from initial condition x0.
        
        Parameters
        ----------
        x0 : np.ndarray
            Initial condition
        t_span : np.ndarray
            Time points
        
        Returns
        -------
        np.ndarray
            Solution trajectory (len(t_span) × n_genes)
        """
        return odeint(self.dynamics, x0, t_span)


def create_solver(
    adata: AnnData,
    cluster: str,
    degradation_key: str = 'gamma'
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
    
    Returns
    -------
    ODESolver
        Configured ODE solver
    """
    genes = get_genes_used(adata)
    
    W = adata.varp[f'W_{cluster}']
    I = adata.var[f'I_{cluster}'].values[genes]
    
    gamma_key = f'gamma_{cluster}'
    gamma = adata.var[gamma_key].values[genes] if gamma_key in adata.var else adata.var[degradation_key].values[genes]
    
    threshold = adata.var['sigmoid_threshold'].values[genes]
    exponent = adata.var['sigmoid_exponent'].values[genes]
    
    return ODESolver(W, I, gamma, threshold, exponent)
