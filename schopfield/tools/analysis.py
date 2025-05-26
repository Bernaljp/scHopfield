import numpy as np
from typing import Optional, Dict
import logging
from anndata import AnnData
from scipy.sparse import issparse
from schopfield.utils.math import compute_sigmoid, int_sig_act_inv
from schopfield.utils.data import get_matrix, to_numpy, write_energies

logger = logging.getLogger(__name__)

def compute_energies(landscape: 'Landscape', x: Optional[np.ndarray] = None) -> Dict[str, Dict[str, np.ndarray]]:
    """Calculate energy components for each cluster or a specific input.

    Computes interaction, degradation, and bias energies, storing results in adata.obs
    via write_energies if x is None, or returning a dictionary if x is provided.

    Args:
        landscape: Landscape object containing adata, W, I, gamma, and parameters.
        x: Optional input data (n_cells, n_genes) to compute energies for.

    Returns:
        Dict[str, Dict[str, np.ndarray]]: Dictionary with keys 'total', 'interaction',
            'degradation', 'bias', each mapping clusters to energy arrays.

    Raises:
        ValueError: If required parameters (W, I, gamma, threshold, exponent) are not initialized.

    Notes:
        Requires fitted parameters from schopfield.tools.fitting.fit_interactions and fit_sigmoids.
        Results are stored in adata.obs['Total_energy'], etc., if x is None.
    """
    logger.info("Computing energy components")
    
    # Validate parameters
    if not landscape.W or not landscape.I or not landscape.gamma:
        raise ValueError("Interaction parameters not initialized; run schopfield.tools.fitting.fit_interactions")
    if x is not None and (landscape.threshold is None or landscape.exponent is None):
        raise ValueError("Sigmoid parameters not initialized; run schopfield.tools.fitting.fit_sigmoids")
    
    # Initialize energy dictionaries
    energies = {'total': {}, 'interaction': {}, 'degradation': {}, 'bias': {}}
    
    # Compute energies for each cluster
    for cluster in landscape.W.keys():
        interaction = _interaction_energy(landscape, cluster, x)
        degradation = _degradation_energy(landscape, cluster, x)
        bias = _bias_energy(landscape, cluster, x)
        total = interaction + degradation + bias
        
        energies['interaction'][cluster] = interaction
        energies['degradation'][cluster] = degradation
        energies['bias'][cluster] = bias
        energies['total'][cluster] = total
    
    # Store in adata if x is None
    if x is None:
        write_energies(landscape, energies)
    
    return energies

def _interaction_energy(landscape: 'Landscape', cluster: str, x: Optional[np.ndarray] = None) -> np.ndarray:
    """Calculate interaction energy for a cluster.

    Args:
        landscape: Landscape object containing adata, W, and parameters.
        cluster: Cluster label or 'all'.
        x: Optional input data (n_cells, n_genes).

    Returns:
        np.ndarray: Interaction energy (n_cells,).
    """
    idx = (landscape.adata.obs[landscape.cluster_key] == cluster
           if cluster != 'all' and landscape.cluster_key is not None else slice(None))
    sig = (compute_sigmoid(x, landscape.threshold, landscape.exponent)
           if x is not None else get_matrix(landscape.adata, 'sigmoid', genes=landscape.genes)[idx])
    W = landscape.W[cluster]
    
    return -0.5 * np.sum((sig @ W.T) * sig, axis=1)

def _degradation_energy(landscape: 'Landscape', cluster: str, x: Optional[np.ndarray] = None) -> np.ndarray:
    """Calculate degradation energy for a cluster.

    Args:
        landscape: Landscape object containing adata, gamma, and parameters.
        cluster: Cluster label or 'all'.
        x: Optional input data (n_cells, n_genes).

    Returns:
        np.ndarray: Degradation energy (n_cells,).
    """
    idx = (landscape.adata.obs[landscape.cluster_key] == cluster
           if cluster != 'all' and landscape.cluster_key is not None else slice(None))
    g = (landscape.adata.var[landscape.gamma_key][landscape.genes].values
         if not landscape.refit_gamma else landscape.gamma[cluster])
    sig = (compute_sigmoid(x, landscape.threshold, landscape.exponent)
           if x is not None else get_matrix(landscape.adata, 'sigmoid', genes=landscape.genes)[idx])
    
    integral = int_sig_act_inv(sig, landscape.threshold, landscape.exponent)
    return np.sum(g[None, :] * integral, axis=1)

def _bias_energy(landscape: 'Landscape', cluster: str, x: Optional[np.ndarray] = None) -> np.ndarray:
    """Calculate bias energy for a cluster.

    Args:
        landscape: Landscape object containing adata, I, and parameters.
        cluster: Cluster label or 'all'.
        x: Optional input data (n_cells, n_genes).

    Returns:
        np.ndarray: Bias energy (n_cells,).
    """
    idx = (landscape.adata.obs[landscape.cluster_key] == cluster
           if cluster != 'all' and landscape.cluster_key is not None else slice(None))
    sig = (compute_sigmoid(x, landscape.threshold, landscape.exponent)
           if x is not None else get_matrix(landscape.adata, 'sigmoid', genes=landscape.genes)[idx])
    I = landscape.I[cluster]
    
    return -np.sum(I[None, :] * sig, axis=1)