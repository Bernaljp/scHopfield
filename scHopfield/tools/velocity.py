"""Velocity computation and validation."""

import numpy as np
from typing import Optional, Union
from anndata import AnnData

from .._utils.io import get_matrix, to_numpy, get_genes_used


def compute_reconstructed_velocity(
    adata: AnnData,
    cluster: Optional[str] = None,
    spliced_key: str = 'Ms',
    degradation_key: str = 'gamma',
    cluster_key: str = 'cell_type',
    layer_key: Optional[str] = None,
    copy: bool = False
) -> Union[AnnData, np.ndarray]:
    """
    Compute reconstructed velocity from Hopfield model.

    The velocity is computed as: v = W @ sigmoid(X) - gamma * X + I

    Parameters
    ----------
    adata : AnnData
        Annotated data object with fitted interactions
    cluster : str, optional
        Cluster to compute velocity for. If None, computes for all cells
        using their respective cluster parameters
    spliced_key : str, optional (default: 'Ms')
        Key in adata.layers for spliced counts
    degradation_key : str, optional (default: 'gamma')
        Key in adata.var for degradation rates
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    layer_key : str, optional
        If provided, stores reconstructed velocity in adata.layers[layer_key]
        If cluster is specified, uses f'{layer_key}_{cluster}'
    copy : bool, optional (default: False)
        If True, return a copy instead of modifying in-place

    Returns
    -------
    AnnData or np.ndarray
        If layer_key is provided and copy=False: None (modifies adata in-place)
        If layer_key is provided and copy=True: modified copy of adata
        If layer_key is None: np.ndarray with reconstructed velocities
    """
    if layer_key is not None and copy:
        adata = adata.copy()

    genes = get_genes_used(adata)
    n_genes = len(genes)

    if cluster is not None:
        # Compute for specific cluster
        cluster_mask = (adata.obs[cluster_key] == cluster).values
        cluster_indices = np.where(cluster_mask)[0]
        n_cells = len(cluster_indices)

        # Get cluster-specific parameters
        W = adata.varp[f'W_{cluster}']
        I_vec = adata.var[f'I_{cluster}'].values[genes] if f'I_{cluster}' in adata.var.columns else np.zeros(n_genes)

        gamma_col = f'gamma_{cluster}'
        gamma_vec = adata.var[gamma_col].values[genes] if gamma_col in adata.var.columns else adata.var[degradation_key].values[genes]

        # Get expression data
        X = to_numpy(get_matrix(adata, spliced_key, genes=genes)[cluster_mask])
        sigmoid_vals = to_numpy(get_matrix(adata, 'sigmoid', genes=genes)[cluster_mask])

        # Compute velocity: W @ sigmoid(X) - gamma * X + I
        reconstructed_v = (W @ sigmoid_vals.T).T - gamma_vec * X + I_vec

        if layer_key is not None:
            # Store in layer
            key = f'{layer_key}_{cluster}'
            if key not in adata.layers:
                adata.layers[key] = np.zeros((adata.n_obs, adata.n_vars))
            adata.layers[key][cluster_indices[:, None], genes[None, :]] = reconstructed_v
            return adata if copy else None
        else:
            return reconstructed_v

    else:
        # Compute for all cells using their respective cluster parameters
        clusters = adata.obs[cluster_key].unique()
        reconstructed_v = np.zeros((adata.n_obs, n_genes))

        for clust in clusters:
            cluster_mask = (adata.obs[cluster_key] == clust).values
            cluster_indices = np.where(cluster_mask)[0]

            # Get cluster-specific parameters
            W = adata.varp[f'W_{clust}']
            I_vec = adata.var[f'I_{clust}'].values[genes] if f'I_{clust}' in adata.var.columns else np.zeros(n_genes)

            gamma_col = f'gamma_{clust}'
            gamma_vec = adata.var[gamma_col].values[genes] if gamma_col in adata.var.columns else adata.var[degradation_key].values[genes]

            # Get expression data
            X = to_numpy(get_matrix(adata, spliced_key, genes=genes)[cluster_mask])
            sigmoid_vals = to_numpy(get_matrix(adata, 'sigmoid', genes=genes)[cluster_mask])

            # Compute velocity
            reconstructed_v[cluster_indices] = (W @ sigmoid_vals.T).T - gamma_vec * X + I_vec

        if layer_key is not None:
            # Store in layer
            if layer_key not in adata.layers:
                adata.layers[layer_key] = np.zeros((adata.n_obs, adata.n_vars))
            adata.layers[layer_key][:, genes] = reconstructed_v
            return adata if copy else None
        else:
            return reconstructed_v


def validate_velocity(
    adata: AnnData,
    velocity_key: str = 'velocity',
    spliced_key: str = 'Ms',
    degradation_key: str = 'gamma',
    cluster_key: str = 'cell_type',
    return_mse: bool = True
) -> Union[float, dict]:
    """
    Validate reconstructed velocity against original velocity.

    Computes mean squared error between Hopfield model predictions
    and original RNA velocity estimates.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with fitted interactions and velocity
    velocity_key : str, optional (default: 'velocity')
        Key in adata.layers for original velocity
    spliced_key : str, optional (default: 'Ms')
        Key in adata.layers for spliced counts
    degradation_key : str, optional (default: 'gamma')
        Key in adata.var for degradation rates
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    return_mse : bool, optional (default: True)
        If True, returns overall MSE. If False, returns dict with
        per-cluster MSE values

    Returns
    -------
    float or dict
        Overall MSE (if return_mse=True) or dict mapping cluster names
        to their MSE values (if return_mse=False)
    """
    genes = get_genes_used(adata)
    clusters = adata.obs[cluster_key].unique()

    if not return_mse:
        mse_dict = {}

    total_squared_error = 0
    total_elements = 0

    for cluster in clusters:
        # Compute reconstructed velocity for this cluster
        reconstructed_v = compute_reconstructed_velocity(
            adata,
            cluster=cluster,
            spliced_key=spliced_key,
            degradation_key=degradation_key,
            cluster_key=cluster_key
        )

        # Get original velocity
        cluster_mask = (adata.obs[cluster_key] == cluster).values
        original_v = to_numpy(get_matrix(adata, velocity_key, genes=genes)[cluster_mask])

        # Compute squared error
        squared_error = (reconstructed_v - original_v) ** 2
        cluster_mse = np.mean(squared_error)

        if not return_mse:
            mse_dict[cluster] = cluster_mse

        total_squared_error += np.sum(squared_error)
        total_elements += squared_error.size

    if return_mse:
        return total_squared_error / total_elements
    else:
        return mse_dict
