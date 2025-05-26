import numpy as np
import anndata as ad
from scipy.sparse import issparse
from typing import Dict, Optional, Union
import logging
from .math import compute_sigmoid

logger = logging.getLogger(__name__)

def to_numpy(matrix: Union[np.ndarray, 'sparse.spmatrix']) -> np.ndarray:
    """Convert a matrix to a NumPy array."""
    logger.debug("Converting matrix to NumPy array")
    return matrix.toarray() if issparse(matrix) else np.asarray(matrix)

def get_matrix(adata: ad.AnnData, key: str, genes: Optional[np.ndarray] = None) -> np.ndarray:
    """Retrieve a specific matrix from the AnnData object."""
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

def write_property(adata: ad.AnnData, key: str, values: np.ndarray) -> None:
    """Write a property to adata.var."""
    if values.shape[0] != adata.n_vars:
        raise ValueError(f"Values shape {values.shape} does not match n_vars {adata.n_vars}")
    adata.var[key] = values
    logger.debug(f"Written property '{key}' to adata.var")

def write_sigmoids(landscape: 'Landscape') -> None:
    """Write sigmoid activations to the AnnData object."""
    logger.info("Writing sigmoid activations to adata.layers")
    if landscape.threshold is None or landscape.exponent is None:
        raise ValueError("Sigmoid parameters not initialized; run schopfield.tools.fitting.fit_sigmoids")
    
    x = get_matrix(landscape.adata, landscape.spliced_matrix_key, genes=landscape.genes)
    sig = compute_sigmoid(x, landscape.threshold, landscape.exponent)
    landscape.adata.layers['sigmoid'] = sig

def write_energies(landscape: 'Landscape', energies: Dict[str, Dict[str, np.ndarray]]) -> None:
    """Write energy components to adata.obs."""
    for e_type, clusters in energies.items():
        for cluster, values in clusters.items():
            key = f"{e_type.capitalize()}_energy"
            if cluster == "all":
                landscape.adata.obs[key] = values
            else:
                mask = landscape.adata.obs[landscape.cluster_key] == cluster
                landscape.adata.obs.loc[mask, key] = values[mask]
    logger.info("Written energies to adata.obs")