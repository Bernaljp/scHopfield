"""Energy landscape computation."""

import numpy as np
from typing import Optional, Union, Tuple, Dict
from anndata import AnnData

from .._utils.math import sigmoid, int_sig_act_inv
from .._utils.io import get_matrix, to_numpy, get_genes_used, get_cluster_key


def compute_energies(
    adata: AnnData,
    spliced_key: str = 'Ms',
    degradation_key: str = 'gamma',
    copy: bool = False
) -> Optional[AnnData]:
    """
    Calculate energy landscapes for all clusters.

    Computes total energy and its components (interaction, degradation, bias)
    for each cell based on the inferred gene regulatory network.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with fitted interactions
    spliced_key : str, optional (default: 'Ms')
        Key in adata.layers for spliced counts
    degradation_key : str, optional (default: 'gamma')
        Key in adata.var for degradation rates
    copy : bool, optional (default: False)
        If True, return a copy instead of modifying in-place

    Returns
    -------
    AnnData or None
        Returns adata if copy=True, otherwise None.
        Adds to adata.obs for each cluster:
        - 'energy_total_{cluster}'
        - 'energy_interaction_{cluster}'
        - 'energy_degradation_{cluster}'
        - 'energy_bias_{cluster}'

    Notes
    -----
    Energy formula:
    E = -0.5 * s^T W s + gamma * integral(sigmoid^-1) - I^T s
    """
    adata = adata.copy() if copy else adata

    cluster_key = get_cluster_key(adata)
    genes = get_genes_used(adata)

    # Find all clusters that have been fitted
    clusters = [k.replace('I_', '') for k in adata.var.columns if k.startswith('I_')]

    for cluster in clusters:
        # Compute energy components
        e_int = _interaction_energy(adata, cluster, spliced_key, degradation_key, x=None)
        e_deg = _degradation_energy(adata, cluster, spliced_key, degradation_key, x=None)
        e_bias = _bias_energy(adata, cluster, spliced_key, x=None)

        # Get cluster indices
        if cluster == 'all':
            idx = np.ones(adata.n_obs, dtype=bool)
        else:
            idx = adata.obs[cluster_key] == cluster

        # Initialize columns if not present
        if f'energy_total_{cluster}' not in adata.obs:
            adata.obs[f'energy_total_{cluster}'] = 0.0
            adata.obs[f'energy_interaction_{cluster}'] = 0.0
            adata.obs[f'energy_degradation_{cluster}'] = 0.0
            adata.obs[f'energy_bias_{cluster}'] = 0.0

        # Store energies
        adata.obs.loc[idx, f'energy_total_{cluster}'] = e_int + e_deg + e_bias
        adata.obs.loc[idx, f'energy_interaction_{cluster}'] = e_int
        adata.obs.loc[idx, f'energy_degradation_{cluster}'] = e_deg
        adata.obs.loc[idx, f'energy_bias_{cluster}'] = e_bias

    return adata if copy else None


def _interaction_energy(
    adata: AnnData,
    cluster: str,
    spliced_key: str,
    degradation_key: str,
    x: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate interaction energy: -0.5 * s^T W s

    Adapted from Landscape.interaction_energy.
    """
    cluster_key = get_cluster_key(adata)
    genes = get_genes_used(adata)

    # Get sigmoid activations
    if x is not None:
        threshold = adata.var['sigmoid_threshold'].values[genes]
        exponent = adata.var['sigmoid_exponent'].values[genes]
        sig = np.nan_to_num(sigmoid(x, threshold[None, :], exponent[None, :]))
    else:
        if cluster == 'all':
            idx = slice(None)
        else:
            idx = adata.obs[cluster_key] == cluster
        sig = get_matrix(adata, 'sigmoid', genes=genes)[idx]

    # Get interaction matrix
    W = adata.varp[f'W_{cluster}']

    # Calculate interaction energy
    interaction_energy = -0.5 * np.sum((sig @ W.T) * sig, axis=1)
    return interaction_energy


def _degradation_energy(
    adata: AnnData,
    cluster: str,
    spliced_key: str,
    degradation_key: str,
    x: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate degradation energy using integral of inverse sigmoid.

    Adapted from Landscape.degradation_energy.
    """
    cluster_key = get_cluster_key(adata)
    genes = get_genes_used(adata)

    # Get degradation rates
    gamma_col = f'gamma_{cluster}'
    if gamma_col in adata.var:
        g = adata.var[gamma_col].values[genes]
    else:
        g = adata.var[degradation_key].values[genes]

    # Get sigmoid activations
    if x is not None:
        threshold = adata.var['sigmoid_threshold'].values[genes]
        exponent = adata.var['sigmoid_exponent'].values[genes]
        sig = np.nan_to_num(sigmoid(x, threshold[None, :], exponent[None, :]))
    else:
        if cluster == 'all':
            idx = slice(None)
        else:
            idx = adata.obs[cluster_key] == cluster
        sig = get_matrix(adata, 'sigmoid', genes=genes)[idx]

    # Get sigmoid parameters
    threshold = adata.var['sigmoid_threshold'].values[genes]
    exponent = adata.var['sigmoid_exponent'].values[genes]

    # Compute integral
    integral = int_sig_act_inv(sig, threshold, exponent)
    degradation_energy = np.sum(g[None, :] * integral, axis=1)

    return degradation_energy


def _bias_energy(
    adata: AnnData,
    cluster: str,
    spliced_key: str,
    x: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate bias energy: -I^T s

    Adapted from Landscape.bias_energy.
    """
    cluster_key = get_cluster_key(adata)
    genes = get_genes_used(adata)

    # Get sigmoid activations
    if x is not None:
        threshold = adata.var['sigmoid_threshold'].values[genes]
        exponent = adata.var['sigmoid_exponent'].values[genes]
        sig = np.nan_to_num(sigmoid(x, threshold[None, :], exponent[None, :]))
    else:
        if cluster == 'all':
            idx = slice(None)
        else:
            idx = adata.obs[cluster_key] == cluster
        sig = get_matrix(adata, 'sigmoid', genes=genes)[idx]

    # Get bias vector
    I = adata.var[f'I_{cluster}'].values[genes]

    # Calculate bias energy
    bias_energy = -np.sum(I[None, :] * sig, axis=1)
    return bias_energy


def decompose_degradation_energy(
    adata: AnnData,
    cluster: str,
    spliced_key: str = 'Ms',
    degradation_key: str = 'gamma',
    x: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate gene-wise degradation energy.

    Adapted from Landscape.degradation_energy_decomposed.

    Returns
    -------
    np.ndarray
        Array of shape (n_cells, n_genes) with degradation energy per gene
    """
    cluster_key = get_cluster_key(adata)
    genes = get_genes_used(adata)

    # Get degradation rates
    gamma_col = f'gamma_{cluster}'
    if gamma_col in adata.var:
        g = adata.var[gamma_col].values[genes]
    else:
        g = adata.var[degradation_key].values[genes]

    # Get sigmoid
    if x is not None:
        threshold = adata.var['sigmoid_threshold'].values[genes]
        exponent = adata.var['sigmoid_exponent'].values[genes]
        sig = np.nan_to_num(sigmoid(x, threshold[None, :], exponent[None, :]))
    else:
        if cluster == 'all':
            idx = slice(None)
        else:
            idx = adata.obs[cluster_key] == cluster
        sig = get_matrix(adata, 'sigmoid', genes=genes)[idx]

    threshold = adata.var['sigmoid_threshold'].values[genes]
    exponent = adata.var['sigmoid_exponent'].values[genes]
    integral = int_sig_act_inv(sig, threshold, exponent)

    return g[None, :] * integral


def decompose_bias_energy(
    adata: AnnData,
    cluster: str,
    spliced_key: str = 'Ms',
    x: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate gene-wise bias energy.

    Adapted from Landscape.bias_energy_decomposed.

    Returns
    -------
    np.ndarray
        Array of shape (n_cells, n_genes) with bias energy per gene
    """
    cluster_key = get_cluster_key(adata)
    genes = get_genes_used(adata)

    # Get sigmoid
    if x is not None:
        threshold = adata.var['sigmoid_threshold'].values[genes]
        exponent = adata.var['sigmoid_exponent'].values[genes]
        sig = np.nan_to_num(sigmoid(x, threshold[None, :], exponent[None, :]))
    else:
        if cluster == 'all':
            idx = slice(None)
        else:
            idx = adata.obs[cluster_key] == cluster
        sig = get_matrix(adata, 'sigmoid', genes=genes)[idx]

    I = adata.var[f'I_{cluster}'].values[genes]
    return -I[None, :] * sig


def decompose_interaction_energy(
    adata: AnnData,
    cluster: str,
    side: str = 'in',
    spliced_key: str = 'Ms',
    x: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate gene-wise interaction energy.

    Adapted from Landscape.interaction_energy_decomposed.

    Parameters
    ----------
    side : str, optional (default: 'in')
        'in' for incoming interactions, 'out' for outgoing interactions

    Returns
    -------
    np.ndarray
        Array of shape (n_cells, n_genes) with interaction energy per gene
    """
    cluster_key = get_cluster_key(adata)
    genes = get_genes_used(adata)

    # Get sigmoid
    if x is not None:
        threshold = adata.var['sigmoid_threshold'].values[genes]
        exponent = adata.var['sigmoid_exponent'].values[genes]
        sig = np.nan_to_num(sigmoid(x, threshold[None, :], exponent[None, :]))
    else:
        if cluster == 'all':
            idx = slice(None)
        else:
            idx = adata.obs[cluster_key] == cluster
        sig = get_matrix(adata, 'sigmoid', genes=genes)[idx]

    W = adata.varp[f'W_{cluster}']

    if side == 'out':
        return -0.5 * (sig @ W.T) * sig
    else:  # 'in'
        return -0.5 * (sig @ W) * sig
