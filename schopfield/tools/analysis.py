import numpy as np
from typing import Optional, Dict
import logging
from anndata import AnnData
from scipy.sparse import issparse
from schopfield.utils.math import compute_sigmoid, int_sig_act_inv
from schopfield.utils.data import to_numpy, write_energies, get_matrix

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
    decomposed_energy = interaction_energy_decomposed(landscape, cluster, x=x)
    return np.sum(decomposed_energy, axis=1)

def _degradation_energy(landscape: 'Landscape', cluster: str, x: Optional[np.ndarray] = None) -> np.ndarray:
    """Calculate degradation energy for a cluster.

    Args:
        landscape: Landscape object containing adata, gamma, and parameters.
        cluster: Cluster label or 'all'.
        x: Optional input data (n_cells, n_genes).

    Returns:
        np.ndarray: Degradation energy (n_cells,).
    """
    decomposed_energy = degradation_energy_decomposed(landscape, cluster, x=x)
    return np.sum(decomposed_energy, axis=1)

def _bias_energy(landscape: 'Landscape', cluster: str, x: Optional[np.ndarray] = None) -> np.ndarray:
    """Calculate bias energy for a cluster.

    Args:
        landscape: Landscape object containing adata, I, and parameters.
        cluster: Cluster label or 'all'.
        x: Optional input data (n_cells, n_genes).

    Returns:
        np.ndarray: Bias energy (n_cells,).
    """
    decomposed_energy = bias_energy_decomposed(landscape, cluster, x=x)
    return np.sum(decomposed_energy, axis=1)

def degradation_energy_decomposed(landscape: 'Landscape', cluster: str, x: Optional[np.ndarray] = None) -> np.ndarray:
    """Calculate per-gene decomposition of degradation energy for a cluster or all cells.

    Computes degradation energy contributions for each gene, using gamma and the integral
    of the inverse sigmoid activation.

    Args:
        landscape: Landscape object containing adata, gamma, and parameters.
        cluster: Cluster label or 'all' for all cells.
        x: Optional input data (n_cells, n_genes) to compute energy for.

    Returns:
        np.ndarray: Decomposed degradation energy (n_cells, n_genes).

    Raises:
        ValueError: If gamma, threshold, or exponent are not initialized.

    Notes:
        Requires fitted parameters from schopfield.tools.fitting.fit_sigmoids.
        Uses compute_sigmoid from utils/math.
    """
    logger.info(f"Computing degradation energy decomposition for cluster: {cluster}")

    # Validate parameters
    if landscape.gamma is None or (x is not None and (landscape.threshold is None or landscape.exponent is None)):
        raise ValueError("Parameters not initialized; run schopfield.tools.fitting.fit_sigmoids")

    idx = landscape.adata.obs[landscape.cluster_key] == cluster if cluster != 'all' else slice(None)
    g = (landscape.adata.var[landscape.gamma_key][landscape.genes].values
         if not landscape.refit_gamma else landscape.gamma[cluster])
    
    threshold = landscape.threshold
    exponent = landscape.exponent

    # Compute sigmoid activation
    sig = (compute_sigmoid(x, threshold[None,:], exponent[None,:]) if x is not None
           else get_matrix(landscape.adata, 'sigmoid', genes=landscape.genes)[idx])
    
    integral = int_sig_act_inv(sig, threshold[None,:], exponent[None,:]).squeeze()
    return g[None, :] * integral

def bias_energy_decomposed(landscape: 'Landscape', cluster: str, x: Optional[np.ndarray] = None) -> np.ndarray:
    """Calculate per-gene decomposition of bias energy for a cluster or all cells.

    Computes bias energy contributions for each gene, using the bias vector I and sigmoid activation.

    Args:
        landscape: Landscape object containing adata, I, and parameters.
        cluster: Cluster label or 'all' for all cells.
        x: Optional input data (n_cells, n_genes) to compute energy for.

    Returns:
        np.ndarray: Decomposed bias energy (n_cells, n_genes).

    Raises:
        ValueError: If I is not initialized or sigmoid parameters are missing when x is provided.

    Notes:
        Requires fitted parameters from schopfield.tools.fitting.fit_interactions.
        Uses compute_sigmoid from utils/math.
    """
    logger.info(f"Computing bias energy decomposition for cluster: {cluster}")
    # Validate parameters
    if landscape.I is None or (x is not None and (landscape.threshold is None or landscape.exponent is None)):
        raise ValueError("Parameters not initialized; run schopfield.tools.fitting.fit_interactions and fit_sigmoids")

    idx = landscape.adata.obs[landscape.cluster_key] == cluster if cluster != 'all' else slice(None)
    I = landscape.I[cluster]
    threshold = landscape.threshold
    exponent = landscape.exponent

    # Compute sigmoid activation
    sig = (compute_sigmoid(x, threshold[None,:], exponent[None,:]) if x is not None
           else get_matrix(landscape.adata, 'sigmoid', genes=landscape.genes)[idx])
    
    return -I[None, :] * sig

def interaction_energy_decomposed(landscape: 'Landscape', cluster: str, side: str = 'out', x: Optional[np.ndarray] = None) -> np.ndarray:
    """Calculate per-gene decomposition of interaction energy for a cluster or all cells.

    Computes interaction energy contributions for each gene, based on the interaction matrix W
    and sigmoid activation, for either incoming ('in') or outgoing ('out') interactions.

    Args:
        landscape: Landscape object containing adata, W, and parameters.
        cluster: Cluster label or 'all' for all cells.
        side: Specifies interaction direction ('in' or 'out'). Defaults to 'out'.
        x: Optional input data (n_cells, n_genes) to compute energy for.

    Returns:
        np.ndarray: Decomposed interaction energy (n_cells, n_genes).

    Raises:
        ValueError: If W is not initialized, side is invalid, or sigmoid parameters are missing when x is provided.

    Notes:
        Requires fitted parameters from schopfield.tools.fitting.fit_interactions.
        Uses compute_sigmoid from utils/math.
    """
    logger.info(f"Computing interaction energy decomposition for cluster: {cluster}, side: {side}")

    # Validate parameters
    if landscape.W is None or (x is not None and (landscape.threshold is None or landscape.exponent is None)):
        raise ValueError("Parameters not initialized; run schopfield.tools.fitting.fit_interactions and fit_sigmoids")
    if side not in ['in', 'out']:
        raise ValueError("Side must be 'in' or 'out'")

    idx = landscape.adata.obs[landscape.cluster_key] == cluster if cluster != 'all' else slice(None)
    W = landscape.W[cluster]
    threshold = landscape.threshold
    exponent = landscape.exponent

    # Compute sigmoid activation
    sig = (compute_sigmoid(x, threshold[None,:], exponent[None,:]) if x is not None
           else get_matrix(landscape.adata, 'sigmoid', genes=landscape.genes)[idx])
    
    return -0.5 * (sig @ W.T) * sig if side == 'out' else -0.5 * (sig @ W) * sig