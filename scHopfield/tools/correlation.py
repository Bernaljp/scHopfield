"""Correlation analysis between energies, genes, and cell types."""

import numpy as np
import pandas as pd
import itertools
from typing import Optional
from anndata import AnnData
import hoggorm as ho

from .._utils.io import get_matrix, to_numpy, get_genes_used


def energy_gene_correlation(
    adata: AnnData,
    spliced_key: str = 'Ms',
    cluster_key: str = 'cell_type',
    copy: bool = False
) -> Optional[AnnData]:
    """
    Correlate energies with gene expression.

    Computes Pearson correlation between energy values and each gene's
    expression for each cluster.

    Adapted from Landscape.energy_genes_correlation.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with computed energies
    spliced_key : str, optional (default: 'Ms')
        Key in adata.layers for expression data
    copy : bool, optional (default: False)
        If True, return a copy instead of modifying in-place

    Returns
    -------
    AnnData or None
        Returns adata if copy=True, otherwise None.
        Adds to adata.var for each cluster and energy type:
        - 'correlation_total_{cluster}'
        - 'correlation_interaction_{cluster}'
        - 'correlation_degradation_{cluster}'
        - 'correlation_bias_{cluster}'
    """
    adata = adata.copy() if copy else adata

    genes = get_genes_used(adata)
    gene_names = adata.var.index[genes]

    # Get clusters
    clusters = adata.obs[cluster_key].unique()

    # Initialize arrays for all cells
    energies = np.zeros((4, adata.n_obs))

    for cluster in clusters:
        if cluster == 'all':
            continue

        # Get cluster cells
        cells = adata.obs[cluster_key] == cluster

        # Get energies
        energies[0, cells] = adata.obs[f'energy_total_{cluster}'].values[cells]
        energies[1, cells] = adata.obs[f'energy_interaction_{cluster}'].values[cells]
        energies[2, cells] = adata.obs[f'energy_degradation_{cluster}'].values[cells]
        energies[3, cells] = adata.obs[f'energy_bias_{cluster}'].values[cells]

        # Get expression
        X = to_numpy(get_matrix(adata, spliced_key, genes=genes)[cells].T)

        # Compute correlations
        correlations = np.nan_to_num(np.corrcoef(np.vstack((energies[:, cells], X)))[:4, 4:])

        # Initialize columns if not present
        for i, etype in enumerate(['total', 'interaction', 'degradation', 'bias']):
            col = f'correlation_{etype}_{cluster}'
            if col not in adata.var:
                adata.var[col] = 0.0
            adata.var.loc[gene_names, col] = correlations[i, :]

    # Compute for 'all' cells
    X_all = to_numpy(get_matrix(adata, spliced_key, genes=genes).T)
    correlations_all = np.nan_to_num(np.corrcoef(np.vstack((energies, X_all)))[:4, 4:])

    for i, etype in enumerate(['total', 'interaction', 'degradation', 'bias']):
        col = f'correlation_{etype}_all'
        if col not in adata.var:
            adata.var[col] = 0.0
        adata.var.loc[gene_names, col] = correlations_all[i, :]

    return adata if copy else None


def celltype_correlation(
    adata: AnnData,
    spliced_key: str = 'Ms',
    cluster_key: str = 'cell_type',
    modified: bool = True,
    all_genes: bool = False,
    copy: bool = False
) -> Optional[AnnData]:
    """
    Compute correlation between cell types based on gene expression.

    Uses RV coefficient to measure similarity between cell type expression profiles.

    Adapted from Landscape.celltype_correlation.

    Parameters
    ----------
    adata : AnnData
        Annotated data object
    spliced_key : str, optional (default: 'Ms')
        Key in adata.layers for expression data
    modified : bool, optional (default: True)
        If True, use modified RV2 coefficient; if False, use RV coefficient
    all_genes : bool, optional (default: False)
        If True, use all genes; if False, use only genes from analysis
    copy : bool, optional (default: False)
        If True, return a copy instead of modifying in-place

    Returns
    -------
    AnnData or None
        Returns adata if copy=True, otherwise None.
        Adds to adata.uns['scHopfield']:
        - 'celltype_correlation': DataFrame with pairwise correlations
    """
    adata = adata.copy() if copy else adata

    keys = adata.obs[cluster_key].unique()

    corr_f = ho.mat_corr_coeff.RV2coeff if modified else ho.mat_corr_coeff.RVcoeff

    rv = pd.DataFrame(index=keys, columns=keys, data=1.0)

    genes = None if all_genes else get_genes_used(adata)
    counts = get_matrix(adata, spliced_key, genes=genes)

    for k1, k2 in itertools.combinations(keys, 2):
        expr_k1 = to_numpy(counts[adata.obs[cluster_key] == k1])
        expr_k2 = to_numpy(counts[adata.obs[cluster_key] == k2])
        rv.loc[k1, k2] = corr_f([expr_k1.T, expr_k2.T])[0, 1]
        rv.loc[k2, k1] = rv.loc[k1, k2]

    adata.uns['scHopfield']['celltype_correlation'] = rv

    return adata if copy else None


def future_celltype_correlation(
    adata: AnnData,
    spliced_key: str = 'Ms',
    cluster_key: str = 'cell_type',
    modified: bool = True,
    copy: bool = False
) -> Optional[AnnData]:
    """
    Compute correlation between cell types based on predicted future states.

    Adapted from Landscape.future_celltype_correlation.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with fitted interactions
    spliced_key : str, optional (default: 'Ms')
        Key in adata.layers for expression data
    modified : bool, optional (default: True)
        If True, use modified RV2 coefficient
    copy : bool, optional (default: False)
        If True, return a copy instead of modifying in-place

    Returns
    -------
    AnnData or None
        Returns adata if copy=True, otherwise None.
        Adds to adata.uns['scHopfield']:
        - 'future_celltype_correlation': DataFrame with pairwise correlations
    """
    adata = adata.copy() if copy else adata

    genes = get_genes_used(adata)
    keys = adata.obs[cluster_key].unique()

    corr_f = ho.mat_corr_coeff.RV2coeff if modified else ho.mat_corr_coeff.RVcoeff

    rv = pd.DataFrame(index=keys, columns=keys, data=1.0)
    counts = get_matrix(adata, spliced_key, genes=genes)

    threshold = adata.var['sigmoid_threshold'].values[genes]
    exponent = adata.var['sigmoid_exponent'].values[genes]

    for k1, k2 in itertools.combinations(keys, 2):
        from .._utils.math import sigmoid

        counts_k1 = to_numpy(counts[adata.obs[cluster_key] == k1])
        counts_k2 = to_numpy(counts[adata.obs[cluster_key] == k2])

        sig_k1 = sigmoid(counts_k1, threshold[None, :], exponent[None, :])
        sig_k2 = sigmoid(counts_k2, threshold[None, :], exponent[None, :])

        W_k1 = adata.varp[f'W_{k1}']
        W_k2 = adata.varp[f'W_{k2}']

        future_k1 = (W_k1 @ sig_k1.T)
        future_k2 = (W_k2 @ sig_k2.T)

        rv.loc[k1, k2] = corr_f([future_k1, future_k2])[0, 1]
        rv.loc[k2, k1] = rv.loc[k1, k2]

    adata.uns['scHopfield']['future_celltype_correlation'] = rv

    return adata if copy else None
