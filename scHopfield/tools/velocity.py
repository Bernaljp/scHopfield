"""Velocity computation and validation."""

import numpy as np
from typing import Optional, Union
from anndata import AnnData
from scipy.sparse import issparse

from .._utils.io import get_matrix, to_numpy, get_genes_used
from .._utils.math import sigmoid


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


def compute_velocity(
    adata: AnnData,
    X: Optional[np.ndarray] = None,
    cluster: Optional[str] = None,
    cluster_key: str = 'cell_type',
    use_cluster_specific: bool = True,
    spliced_key: str = 'Ms',
) -> np.ndarray:
    """
    Compute Hopfield velocity at given expression state.

    v = W @ sigmoid(X) - gamma * X + I

    This is a unified function that replaces the various velocity computation
    functions that were previously in plotting/flow.py.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with fitted interactions
    X : np.ndarray, optional
        Expression matrix (n_cells, n_genes) to compute velocity at.
        If None, uses expression from adata.layers[spliced_key].
    cluster : str, optional
        Specific cluster to use parameters from. If None and use_cluster_specific=True,
        iterates over all clusters using their respective parameters.
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    use_cluster_specific : bool, optional (default: True)
        If True, use cluster-specific W/I/gamma. If False, use 'all' parameters.
    spliced_key : str, optional (default: 'Ms')
        Key in adata.layers for spliced counts (used if X is None)

    Returns
    -------
    np.ndarray
        Velocity matrix (n_cells, n_genes)
    """
    genes_mask = get_genes_used(adata)
    gene_names = adata.var.index[genes_mask]
    n_genes = len(gene_names)

    # Get sigmoid parameters
    threshold = adata.var.loc[gene_names, 'sigmoid_threshold'].values
    exponent = adata.var.loc[gene_names, 'sigmoid_exponent'].values

    # Handle X input
    if X is None:
        X_full = get_matrix(adata, spliced_key, genes=genes_mask)
        X = to_numpy(X_full)
        n_cells = X.shape[0]
        cell_mask = np.ones(n_cells, dtype=bool)
    else:
        n_cells = X.shape[0]
        cell_mask = np.ones(n_cells, dtype=bool)

    # Determine clusters to iterate over
    if cluster is not None:
        # Single cluster specified
        clusters = [cluster]
    elif use_cluster_specific:
        # All clusters
        clusters = adata.obs[cluster_key].unique().tolist()
    else:
        # Use 'all' parameters
        clusters = ['all']

    velocity = np.zeros((n_cells, n_genes))

    for clust in clusters:
        # Determine which cells to process
        if clust == 'all':
            clust_mask = np.ones(n_cells, dtype=bool)
        elif cluster is not None:
            # All cells use same parameters
            clust_mask = np.ones(n_cells, dtype=bool)
        else:
            # Get mask for this cluster
            if X is None or X.shape[0] == adata.n_obs:
                clust_mask = (adata.obs[cluster_key] == clust).values
            else:
                # X is already subset, this shouldn't happen in normal usage
                clust_mask = np.ones(n_cells, dtype=bool)

        if not np.any(clust_mask):
            continue

        # Get parameters for this cluster
        W_key = f'W_{clust}'
        I_key = f'I_{clust}'
        gamma_key = f'gamma_{clust}'

        if W_key not in adata.varp:
            raise ValueError(f"W matrix '{W_key}' not found in adata.varp")

        W = adata.varp[W_key]
        # Slice W if it's full size
        if W.shape[0] == adata.n_vars:
            W = W[np.ix_(genes_mask, genes_mask)]

        # Get I vector
        if I_key in adata.var.columns:
            I_vec = adata.var.loc[gene_names, I_key].values
        else:
            I_vec = np.zeros(n_genes)

        # Get gamma
        if gamma_key in adata.var.columns:
            gamma = adata.var.loc[gene_names, gamma_key].values
        elif 'gamma' in adata.var.columns:
            gamma = adata.var.loc[gene_names, 'gamma'].values
        else:
            gamma = np.ones(n_genes)

        # Compute velocity: v = W @ sigmoid(X) - gamma * X + I
        X_clust = X[clust_mask]
        sig_X = sigmoid(X_clust, threshold, exponent)
        v_clust = (sig_X @ W.T) - (gamma * X_clust) + I_vec

        # Store results
        velocity[clust_mask] = v_clust

    return velocity


def compute_velocity_delta(
    adata: AnnData,
    perturbed_key: str = 'simulated_count',
    original_key: str = 'Ms',
    cluster_key: str = 'cell_type',
    use_cluster_specific: bool = True,
) -> np.ndarray:
    """
    Compute velocity difference between perturbed and original states.

    Returns v(X_perturbed) - v(X_original) for each cell.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with perturbation results
    perturbed_key : str, optional (default: 'simulated_count')
        Key in adata.layers for perturbed expression
    original_key : str, optional (default: 'Ms')
        Key in adata.layers for original expression
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    use_cluster_specific : bool, optional (default: True)
        If True, use cluster-specific W/I/gamma. If False, use 'all' parameters.

    Returns
    -------
    np.ndarray
        Velocity delta matrix (n_cells, n_genes)
    """
    if perturbed_key not in adata.layers:
        raise ValueError(f"'{perturbed_key}' not found in adata.layers. Run simulation first.")

    genes_mask = get_genes_used(adata)

    # Get expression matrices
    X_orig = to_numpy(get_matrix(adata, original_key, genes=genes_mask))
    X_pert = to_numpy(get_matrix(adata, perturbed_key, genes=genes_mask))

    # Determine clusters
    if use_cluster_specific:
        clusters = adata.obs[cluster_key].unique().tolist()
    else:
        clusters = ['all']

    delta_velocity = np.zeros_like(X_orig)

    for cluster in clusters:
        if cluster == 'all':
            mask = np.ones(adata.n_obs, dtype=bool)
        else:
            mask = (adata.obs[cluster_key] == cluster).values

        if not np.any(mask):
            continue

        # Compute velocity at original and perturbed states
        v_orig = compute_velocity(adata, X=X_orig[mask], cluster=cluster)
        v_pert = compute_velocity(adata, X=X_pert[mask], cluster=cluster)

        delta_velocity[mask] = v_pert - v_orig

    return delta_velocity
