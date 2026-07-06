"""Preprocessing functions for sigmoid fitting."""

import numpy as np
from typing import Union, List, Optional
from anndata import AnnData

from .._utils.math import fit_sigmoid, fit_sigmoid_bimodal, sigmoid, hill_regime
from .._utils.io import get_matrix, parse_genes, to_numpy


def fit_all_sigmoids(
    adata: AnnData,
    genes: Union[None, List[str], List[bool], List[int]] = None,
    spliced_key: str = 'Ms',
    min_th: float = 0.05,
    n_min: float = 1.0,
    n_max: float = 8.0,
    refine: bool = True,
    bimodal: bool = False,
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
    n_min, n_max : float, optional (default: 1.0, 8.0)
        Bounds on the fitted Hill exponent (passed to ``fit_sigmoid``).
    refine : bool, optional (default: True)
        Refine each closed-form fit with a bounded nonlinear least-squares step.
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

    # Fit sigmoid to each gene. In bimodal mode, genes whose expression CDF has two
    # regimes get a two-component Hill (a*H(k1,n1)+(1-a)*H(k2,n2)); the second component
    # and mixing weight are stored so compute_sigmoid / the energy can use the
    # regime-specific Hill per cell (single-Hill genes have mix=1, component2=component1).
    if bimodal:
        res = [fit_sigmoid_bimodal(g, min_th=min_th, n_min=n_min, n_max=n_max) for g in x]
        # (k1, n1, k2, n2, a, offset, mse, is_bimodal)
        k1 = np.array([r[0] for r in res]); n1 = np.array([r[1] for r in res])
        k2 = np.array([r[2] for r in res]); n2 = np.array([r[3] for r in res])
        a = np.array([r[4] for r in res]); off = np.array([r[5] for r in res])
        mse = np.array([r[6] for r in res])
        cols = dict(sigmoid_threshold=k1, sigmoid_exponent=n1, sigmoid_threshold2=k2,
                    sigmoid_exponent2=n2, sigmoid_mix=a, sigmoid_offset=off, sigmoid_mse=mse)
    else:
        results = np.array([
            fit_sigmoid(g, min_th=min_th, n_min=n_min, n_max=n_max, refine=refine)
            for g in x
        ])
        cols = dict(sigmoid_threshold=results[:, 0], sigmoid_exponent=results[:, 1],
                    sigmoid_offset=results[:, 2], sigmoid_mse=results[:, 3])

    # Store results in adata.var (initialize with defaults for all genes; single-Hill
    # default so downstream code that ignores the bimodal columns is unaffected).
    for col, default in [('sigmoid_threshold', 0.0), ('sigmoid_exponent', 0.0),
                         ('sigmoid_offset', 0.0), ('sigmoid_mse', 0.0),
                         ('sigmoid_threshold2', 0.0), ('sigmoid_exponent2', 0.0),
                         ('sigmoid_mix', 1.0)]:
        if col not in adata.var:
            adata.var[col] = default
    # component 2 defaults to component 1 (mix=1 -> pure single Hill) for single fits
    if 'sigmoid_threshold2' not in cols:
        cols['sigmoid_threshold2'] = cols['sigmoid_threshold']
        cols['sigmoid_exponent2'] = cols['sigmoid_exponent']
        cols['sigmoid_mix'] = np.ones_like(cols['sigmoid_threshold'])
    for col, vals in cols.items():
        adata.var.iloc[gene_indices, adata.var.columns.get_loc(col)] = vals

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

    # Compute sigmoid. For bimodal (double-sigmoid) genes, assign each cell to the closer
    # Hill component and use that regime's activation; single-Hill genes (mix=1, or no
    # bimodal columns) fall through to the ordinary single Hill.
    if 'sigmoid_mix' in adata.var.columns and \
            (adata.var['sigmoid_mix'].values[gene_indices] < 1 - 1e-9).any():
        k2 = adata.var['sigmoid_threshold2'].values[gene_indices]
        n2 = adata.var['sigmoid_exponent2'].values[gene_indices]
        reg = hill_regime(x, threshold[None, :], k2[None, :])
        sig1 = sigmoid(x, threshold[None, :], exponent[None, :])
        sig2 = sigmoid(x, k2[None, :], n2[None, :])
        sig = np.nan_to_num(np.where(reg == 1, sig2, sig1))
    else:
        sig = np.nan_to_num(sigmoid(x, threshold[None, :], exponent[None, :]))

    # Create full matrix with zeros for unused genes
    sigmoids = np.zeros(adata.layers[spliced_key].shape, dtype=sig.dtype)
    sigmoids[:, gene_indices] = sig

    # Store in layers
    adata.layers[layer_key] = sigmoids

    return adata if copy else None
