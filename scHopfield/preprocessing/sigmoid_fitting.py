"""Preprocessing functions for sigmoid fitting."""

import numpy as np
from typing import Union, List, Optional
from anndata import AnnData

from .._utils.math import fit_sigmoid, sigmoid
from .._utils.io import get_matrix, write_to_adata, parse_genes, to_numpy


def fit_all_sigmoids(
    adata: AnnData,
    genes: Union[None, List[str], List[bool], List[int]] = None,
    spliced_key: str = 'Ms',
    min_th: float = 0.05,
    copy: bool = False
) -> Optional[AnnData]:
    """
    Fit sigmoid functions to gene expression data.

    Fits sigmoid parameters (threshold, exponent, offset) for each gene
    to model the cumulative distribution of expression values.

    Parameters
    ----------
    adata : AnnData
        Annotated data object
    genes : None, list of str, list of int, or array of bool, optional
        Gene subset to use. If None, uses all genes.
    spliced_key : str, optional (default: 'Ms')
        Key in adata.layers for spliced counts
    min_th : float, optional (default: 0.05)
        Minimum threshold as fraction of max expression
    copy : bool, optional (default: False)
        If True, return a copy instead of modifying in-place

    Returns
    -------
    AnnData or None
        Returns adata if copy=True, otherwise None.
        Adds to adata.var:
        - 'sigmoid_threshold': threshold parameter k
        - 'sigmoid_exponent': exponent parameter n
        - 'sigmoid_offset': offset parameter
        - 'sigmoid_mse': mean squared error of fit

    Notes
    -----
    Stores gene indices used in adata.var['scHopfield_used']
    """
    adata = adata.copy() if copy else adata

    # Initialize scHopfield namespace
    if 'scHopfield' not in adata.uns:
        adata.uns['scHopfield'] = {}

    # Store spliced_key for downstream functions
    adata.uns['scHopfield']['spliced_key'] = spliced_key

    # Parse genes
    gene_indices = parse_genes(adata, genes)

    # Mark genes used in analysis (boolean column)
    adata.var['scHopfield_used'] = False
    adata.var.iloc[gene_indices, adata.var.columns.get_loc('scHopfield_used')] = True

    # Get expression data for selected genes
    x = to_numpy(get_matrix(adata, spliced_key, genes=gene_indices).T)

    # Fit sigmoid to each gene
    results = np.array([fit_sigmoid(g, min_th=min_th) for g in x])

    # Store results in adata.var (initialize with zeros for all genes)
    adata.var['sigmoid_threshold'] = 0.0
    adata.var['sigmoid_exponent'] = 0.0
    adata.var['sigmoid_offset'] = 0.0
    adata.var['sigmoid_mse'] = 0.0

    # Fill in values for selected genes
    adata.var.iloc[gene_indices, adata.var.columns.get_loc('sigmoid_threshold')] = results[:, 0]
    adata.var.iloc[gene_indices, adata.var.columns.get_loc('sigmoid_exponent')] = results[:, 1]
    adata.var.iloc[gene_indices, adata.var.columns.get_loc('sigmoid_offset')] = results[:, 2]
    adata.var.iloc[gene_indices, adata.var.columns.get_loc('sigmoid_mse')] = results[:, 3]

    return adata if copy else None


def compute_sigmoid(
    adata: AnnData,
    spliced_key: str = 'Ms',
    layer_key: str = 'sigmoid',
    copy: bool = False
) -> Optional[AnnData]:
    """
    Compute sigmoid-transformed expression values.

    Uses previously fitted sigmoid parameters from fit_all_sigmoids.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with fitted sigmoid parameters
    spliced_key : str, optional (default: 'Ms')
        Key in adata.layers for spliced counts
    layer_key : str, optional (default: 'sigmoid')
        Key for storing sigmoid-transformed data in adata.layers
    copy : bool, optional (default: False)
        If True, return a copy instead of modifying in-place

    Returns
    -------
    AnnData or None
        Returns adata if copy=True, otherwise None.
        Adds adata.layers[layer_key] with sigmoid-transformed expression
    """
    adata = adata.copy() if copy else adata

    # Get gene indices
    if 'scHopfield_used' not in adata.var:
        raise ValueError("No sigmoid parameters found. Run fit_all_sigmoids() first.")
    gene_indices = np.where(adata.var['scHopfield_used'].values)[0]

    # Store spliced_key if not already set
    if 'scHopfield' not in adata.uns:
        adata.uns['scHopfield'] = {}
    if 'spliced_key' not in adata.uns['scHopfield']:
        adata.uns['scHopfield']['spliced_key'] = spliced_key

    # Get expression data
    x = to_numpy(get_matrix(adata, spliced_key, genes=gene_indices))

    # Get sigmoid parameters
    threshold = adata.var['sigmoid_threshold'].values[gene_indices]
    exponent = adata.var['sigmoid_exponent'].values[gene_indices]

    # Compute sigmoid
    sig = np.nan_to_num(sigmoid(x, threshold[None, :], exponent[None, :]))

    # Create full matrix with zeros for unused genes
    sigmoids = np.zeros(adata.layers[spliced_key].shape, dtype=sig.dtype)
    sigmoids[:, gene_indices] = sig

    # Store in layers
    adata.layers[layer_key] = sigmoids

    return adata if copy else None
