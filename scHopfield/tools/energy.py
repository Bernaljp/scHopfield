"""Energy landscape computation."""

import numpy as np
from typing import Optional, Union, Tuple, Dict
from anndata import AnnData

from .._utils.math import sigmoid, int_sig_act_inv
from .._utils.io import get_matrix, to_numpy, get_genes_used


def compute_energies(
    adata: AnnData,
    spliced_key: str = 'Ms',
    degradation_key: str = 'gamma',
    cluster_key: str = 'cell_type',
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
        Adds to adata.obs:
        - 'energy_total'
        - 'energy_interaction'
        - 'energy_degradation'
        - 'energy_bias'

        Each cell's energy is computed using cluster-specific parameters.

    Notes
    -----
    Energy formula:
    E = -0.5 * s^T W s + gamma * integral(sigmoid^-1) - I^T s
    """
    adata = adata.copy() if copy else adata

    genes = get_genes_used(adata)

    # Find all clusters that have been fitted
    clusters = adata.obs[cluster_key].unique()

    # Initialize columns once for all cells
    if 'energy_total' not in adata.obs:
        adata.obs['energy_total'] = 0.0
        adata.obs['energy_interaction'] = 0.0
        adata.obs['energy_degradation'] = 0.0
        adata.obs['energy_bias'] = 0.0

    for cluster in clusters:
        # Get cluster indices
        if cluster == 'all':
            idx = np.ones(adata.n_obs, dtype=bool)
        else:
            idx = (adata.obs[cluster_key] == cluster).values

        # Compute energy components for this cluster's cells using cluster-specific parameters
        e_int = _interaction_energy(adata, cluster, cluster_key)
        e_deg = _degradation_energy(adata, cluster, spliced_key, degradation_key, cluster_key)
        e_bias = _bias_energy(adata, cluster, spliced_key, cluster_key, x=None)

        # Store energies in shared columns
        adata.obs.loc[idx, 'energy_total'] = e_int + e_deg + e_bias
        adata.obs.loc[idx, 'energy_interaction'] = e_int
        adata.obs.loc[idx, 'energy_degradation'] = e_deg
        adata.obs.loc[idx, 'energy_bias'] = e_bias

    return adata if copy else None


def _interaction_energy(
    adata: AnnData,
    cluster: str,
    cluster_key: str,
    x: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate interaction energy component: -0.5 * s^T W s.

    Computes the energy contribution from gene-gene interactions using the
    cluster-specific interaction matrix W and sigmoid activations.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with fitted parameters
    cluster : str
        Cluster name to use for interaction matrix
    cluster_key : str
        Key in adata.obs for cluster labels
    x : np.ndarray, optional
        Optional expression data. If None, uses stored sigmoid values

    Returns
    -------
    np.ndarray
        Array of interaction energies for each cell
    """
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
            idx = (adata.obs[cluster_key] == cluster).values
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
    cluster_key: str = 'cell_type',
    x: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate degradation energy using integral of inverse sigmoid.

    Computes the energy contribution from mRNA degradation by integrating
    the inverse sigmoid function, weighted by degradation rates.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with fitted parameters
    cluster : str
        Cluster name to use for parameters
    spliced_key : str
        Key in adata.layers for spliced counts
    degradation_key : str
        Key in adata.var for degradation rates
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    x : np.ndarray, optional
        Optional expression data. If None, uses stored sigmoid values

    Returns
    -------
    np.ndarray
        Array of degradation energies for each cell
    """
    genes = get_genes_used(adata)

    # Get degradation rates
    g = adata.var[degradation_key].values[genes]

    # Get sigmoid parameters
    threshold = adata.var['sigmoid_threshold'].values[genes]
    exponent = adata.var['sigmoid_exponent'].values[genes]

    # Get sigmoid activations
    if x is not None:
        sig = np.nan_to_num(sigmoid(x, threshold[None, :], exponent[None, :]))
    else:
        idx = (adata.obs[cluster_key] == cluster).values
        sig = get_matrix(adata, 'sigmoid', genes=genes)[idx]

    
    # Compute integral
    integral = int_sig_act_inv(sig, threshold, exponent)
    degradation_energy = np.sum(g[None, :] * integral, axis=1)

    return degradation_energy


def _bias_energy(
    adata: AnnData,
    cluster: str,
    spliced_key: str,
    cluster_key: str = 'cell_type',
    x: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate bias energy component: -I^T s.

    Computes the energy contribution from external inputs or biases using
    the cluster-specific bias vector I and sigmoid activations.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with fitted parameters
    cluster : str
        Cluster name to use for bias vector
    spliced_key : str
        Key in adata.layers for spliced counts
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    x : np.ndarray, optional
        Optional expression data. If None, uses stored sigmoid values

    Returns
    -------
    np.ndarray
        Array of bias energies for each cell
    """

    genes = get_genes_used(adata)

    # Get sigmoid activations
    if x is not None:
        threshold = adata.var['sigmoid_threshold'].values[genes]
        exponent = adata.var['sigmoid_exponent'].values[genes]
        sig = np.nan_to_num(sigmoid(x, threshold[None, :], exponent[None, :]))
    else:
        idx = (adata.obs[cluster_key] == cluster).values
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
    cluster_key: str = 'cell_type',
    x: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate gene-wise degradation energy decomposition.

    Computes the degradation energy contribution for each gene separately,
    allowing analysis of which genes contribute most to the total degradation energy.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with fitted parameters
    cluster : str
        Cluster name to use for parameters
    spliced_key : str, optional (default: 'Ms')
        Key in adata.layers for spliced counts
    degradation_key : str, optional (default: 'gamma')
        Key in adata.var for degradation rates
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    x : np.ndarray, optional
        Optional expression data. If None, uses stored sigmoid values

    Returns
    -------
    np.ndarray
        Array of shape (n_cells, n_genes) with degradation energy per gene
    """
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
            idx = (adata.obs[cluster_key] == cluster).values
        sig = get_matrix(adata, 'sigmoid', genes=genes)[idx]

    threshold = adata.var['sigmoid_threshold'].values[genes]
    exponent = adata.var['sigmoid_exponent'].values[genes]
    integral = int_sig_act_inv(sig, threshold, exponent)

    return g[None, :] * integral


def decompose_bias_energy(
    adata: AnnData,
    cluster: str,
    spliced_key: str = 'Ms',
    cluster_key: str = 'cell_type',
    x: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate gene-wise bias energy decomposition.

    Computes the bias energy contribution for each gene separately,
    allowing analysis of which genes contribute most to the total bias energy.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with fitted parameters
    cluster : str
        Cluster name to use for bias vector
    spliced_key : str, optional (default: 'Ms')
        Key in adata.layers for spliced counts
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    x : np.ndarray, optional
        Optional expression data. If None, uses stored sigmoid values

    Returns
    -------
    np.ndarray
        Array of shape (n_cells, n_genes) with bias energy per gene
    """
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
            idx = (adata.obs[cluster_key] == cluster).values
        sig = get_matrix(adata, 'sigmoid', genes=genes)[idx]

    I = adata.var[f'I_{cluster}'].values[genes]
    return -I[None, :] * sig


def decompose_interaction_energy(
    adata: AnnData,
    cluster: str,
    side: str = 'in',
    spliced_key: str = 'Ms',
    cluster_key: str = 'cell_type',
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
            idx = (adata.obs[cluster_key] == cluster).values
        sig = get_matrix(adata, 'sigmoid', genes=genes)[idx]

    W = adata.varp[f'W_{cluster}']

    if side == 'out':
        return -0.5 * (sig @ W.T) * sig
    else:  # 'in'
        return -0.5 * (sig @ W) * sig
