import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from typing import Union, List, Optional, Dict
import logging

logger = logging.getLogger(__name__)

def to_numpy(matrix: Union[np.ndarray, sparse.spmatrix]) -> np.ndarray:
    """Convert a matrix to a dense NumPy array.

    Args:
        matrix: Input matrix (NumPy array or scipy.sparse matrix).

    Returns:
        np.ndarray: Dense NumPy array.
    """
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)

def get_matrix(adata: ad.AnnData, key: str, genes: Optional[np.ndarray] = None) -> np.ndarray:
    """Retrieve a specific matrix from the AnnData object.

    Args:
        adata: AnnData object containing the data.
        key: Key for the desired matrix in adata.layers or adata.X.
        genes: Indices of genes to subset the matrix (optional).

    Returns:
        np.ndarray: The requested matrix, optionally subset by genes.

    Raises:
        KeyError: If key is not found in adata.layers or adata.X.
    """
    logger.debug(f"Retrieving matrix with key '{key}'")
    if key in adata.layers:
        matrix = adata.layers[key]
    elif key == 'X':
        matrix = adata.X
    else:
        raise KeyError(f"Key '{key}' not found in adata.layers or adata.X")
    
    if genes is not None:
        matrix = matrix[:, genes]
    return to_numpy(matrix)

def write_property(adata: ad.AnnData, key: str, value: np.ndarray) -> None:
    """Write a value to the AnnData object under the specified key.

    The storage location (obs, var, layers, etc.) is determined by the shape of the value.

    Args:
        adata: AnnData object to store the value.
        key: Key under which to store the value.
        value: The value to store (NumPy array).

    Raises:
        ValueError: If value shape does not match expected dimensions for obs, var, layers, etc.
    """
    logger.debug(f"Writing property '{key}' to AnnData")
    shape = np.shape(value)
    
    # Scalar or 1D array
    if len(shape) <= 1:
        value = np.asarray(value).ravel()
        if value.shape[0] == adata.n_obs:
            adata.obs[key] = value
        elif value.shape[0] == adata.n_vars:
            adata.var[key] = value
        else:
            adata.uns[key] = value
    
    # 2D array
    elif len(shape) == 2:
        if shape[0] == adata.n_vars:
            if shape[1] == adata.n_vars:
                adata.varp[key] = value
            else:
                adata.varm[key] = value
        elif shape[0] == adata.n_obs:
            if shape[1] == adata.n_vars:
                adata.layers[key] = value
            elif shape[1] == adata.n_obs:
                adata.obsp[key] = value
            else:
                adata.obsm[key] = value
        else:
            adata.uns[key] = value
    
    # Higher-dimensional or other
    else:
        adata.uns[key] = value

def write_sigmoids(landscape: 'Landscape') -> None:
    """Write sigmoid activations to the AnnData object.

    Args:
        landscape: Landscape object containing adata, genes, threshold, and exponent.

    Stores:
        adata.layers['sigmoid']: Sigmoid activations for all cells and selected genes.

    Raises:
        ValueError: If sigmoid parameters are not initialized.
    """
    logger.info("Writing sigmoid activations to adata.layers")
    if landscape.threshold is None or landscape.exponent is None:
        raise ValueError("Sigmoid parameters not initialized; run schopfield.tools.fitting.fit_sigmoids")
    
    x = get_matrix(landscape.adata, landscape.spliced_matrix_key, genes=landscape.genes)
    sig = compute_sigmoid(x, landscape.threshold, landscape.exponent)
    
    sigmoids = np.zeros((landscape.adata.n_obs, landscape.adata.n_vars), dtype=sig.dtype)
    sigmoids[:, landscape.genes] = sig
    write_property(landscape.adata, 'sigmoid', sigmoids)

def write_energies(landscape: 'Landscape', energies: Dict[str, Dict[str, np.ndarray]]) -> None:
    """Write calculated energies to the AnnData object as observations.

    Args:
        landscape: Landscape object containing adata and cluster_key.
        energies: Dictionary of energy components per cluster, with keys 'total', 'interaction',
                 'degradation', 'bias' and values as arrays (n_cells,).

    Stores:
        adata.obs['Total_energy'], adata.obs['Interaction_energy'],
        adata.obs['Degradation_energy'], adata.obs['Bias_energy']
    """
    logger.info("Writing energies to adata.obs")
    
    # Initialize energy columns
    for key in ['Total_energy', 'Interaction_energy', 'Degradation_energy', 'Bias_energy']:
        landscape.adata.obs[key] = np.zeros(landscape.adata.n_obs, dtype=float)
    
    # Write energies for each cluster
    for cluster in [k for k in energies if k != 'all']:
        if landscape.cluster_key is None:
            continue
        cluster_indices = landscape.adata.obs[landscape.cluster_key] == cluster
        landscape.adata.obs.loc[cluster_indices, 'Total_energy'] = energies[cluster]['total']
        landscape.adata.obs.loc[cluster_indices, 'Interaction_energy'] = energies[cluster]['interaction']
        landscape.adata.obs.loc[cluster_indices, 'Degradation_energy'] = energies[cluster]['degradation']
        landscape.adata.obs.loc[cluster_indices, 'Bias_energy'] = energies[cluster]['bias']