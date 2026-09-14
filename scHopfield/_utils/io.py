"""I/O utility functions for scHopfield."""

import numpy as np
import scipy.sparse as sp


def to_numpy(matrix):
    """
    Convert matrix to NumPy array (handles sparse matrices).

    Args:
        matrix (np.ndarray or scipy.sparse matrix): Input matrix.

    Returns:
        np.ndarray: Dense NumPy array.
    """
    if sp.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def get_matrix(adata, key, genes=None):
    """
    Retrieve a matrix from AnnData layers.

    Parameters
    ----------
    adata : AnnData
        Annotated data object
    key : str
        Key in adata.layers
    genes : array-like, optional
        Gene indices to subset

    Returns
    -------
    np.ndarray
        Requested matrix
    """
    if genes is None:
        return adata.layers[key]
    else:
        return adata.layers[key][:, genes]


def write_to_adata(adata, key, value):
    """
    Write data to appropriate location in AnnData based on shape.

    Parameters
    ----------
    adata : AnnData
        Annotated data object
    key : str
        Key for storing the data
    value : array-like
        Data to store
    """
    shape = np.shape(value)

    if len(shape) == 1:
        if shape[0] == adata.n_obs:
            adata.obs[key] = value
        elif shape[0] == adata.n_vars:
            adata.var[key] = value
        else:
            if 'scHopfield' not in adata.uns:
                adata.uns['scHopfield'] = {}
            adata.uns['scHopfield'][key] = value
    elif len(shape) == 2:
        if shape[0] == adata.n_vars and shape[1] == adata.n_vars:
            adata.varp[key] = value
        elif shape[0] == adata.n_vars:
            adata.varm[key] = value
        elif shape[0] == adata.n_obs and shape[1] == adata.n_vars:
            adata.layers[key] = value
        elif shape[0] == adata.n_obs and shape[1] == adata.n_obs:
            adata.obsp[key] = value
        elif shape[0] == adata.n_obs:
            adata.obsm[key] = value
        else:
            if 'scHopfield' not in adata.uns:
                adata.uns['scHopfield'] = {}
            adata.uns['scHopfield'][key] = value
    else:
        if 'scHopfield' not in adata.uns:
            adata.uns['scHopfield'] = {}
        adata.uns['scHopfield'][key] = value


def parse_genes(adata, genes):
    """
    Parse gene identifiers to indices.

    Parameters
    ----------
    adata : AnnData
        Annotated data object
    genes : None, list of str, list of int, or list of bool
        Gene specification

    Returns
    -------
    np.ndarray
        Gene indices
    """
    if genes is None:
        return np.arange(adata.n_vars)

    if isinstance(genes[0], str):
        gene_indices = adata.var.index.get_indexer_for(genes)
        if np.any(gene_indices == -1):
            missing = np.array(genes)[gene_indices == -1]
            raise ValueError(f"Gene names not found: {missing}")
        return gene_indices
    elif isinstance(genes[0], (int, np.int64, np.int32, np.int16, np.int8)):
        return np.array(genes)
    elif isinstance(genes[0], (bool, np.bool_)):
        if len(genes) != adata.n_vars:
            raise ValueError("Boolean mask must match number of genes")
        return np.where(genes)[0]
    else:
        raise ValueError("Invalid gene specification")


def get_genes_used(adata):
    """
    Get gene indices used in previous scHopfield analysis.

    Parameters
    ----------
    adata : AnnData
        Annotated data object

    Returns
    -------
    np.ndarray
        Gene indices used in analysis
    """
    if 'scHopfield_used' not in adata.var:
        raise ValueError("No scHopfield analysis found. Run pp.fit_all_sigmoids() first.")
    return np.where(adata.var['scHopfield_used'].values)[0]


def ensure_sigmoid_layer(adata, spliced_key=None):
    """
    Ensure ``adata.layers['sigmoid']`` exists, computing it on-the-fly if absent.

    If the layer is already present this is a no-op.  Otherwise
    ``pp.compute_sigmoid`` is called using *spliced_key* (or the value stored
    in ``adata.uns['scHopfield']['spliced_key']`` as a fallback).

    Parameters
    ----------
    adata : AnnData
        Annotated data object with fitted sigmoid parameters.
    spliced_key : str, optional
        Key in ``adata.layers`` for raw spliced counts.  Falls back to the
        value stored by ``fit_all_sigmoids`` in
        ``adata.uns['scHopfield']['spliced_key']``, then to ``'Ms'``.

    Raises
    ------
    ValueError
        If sigmoid parameters (``sigmoid_threshold``) are not present in
        ``adata.var``, meaning ``fit_all_sigmoids`` has not been run yet.
    """
    if 'sigmoid' in adata.layers:
        return

    if 'sigmoid_threshold' not in adata.var.columns:
        raise ValueError(
            "Sigmoid parameters not found in adata.var. "
            "Run sch.pp.fit_all_sigmoids() before calling this function."
        )

    resolved_key = (
        spliced_key
        or adata.uns.get('scHopfield', {}).get('spliced_key')
        or 'Ms'
    )

    # Lazy import avoids circular dependency (preprocessing imports from _utils.io)
    from ..preprocessing import compute_sigmoid
    compute_sigmoid(adata, spliced_key=resolved_key)



def get_cluster_genes(adata, cluster_key, order=None):
    """
    Get used genes, gene names, and filtered clusters.
    """
    genes = get_genes_used(adata)
    gene_names = adata.var.index[genes]

    clusters = adata.obs[cluster_key].unique().tolist()
    if order is not None:
        clusters = [c for c in order if c in clusters]

    return genes, gene_names, clusters


def regime_rule(adata):
    """How the object assigns cells to Hill components: ``'posterior'`` or ``'nearest'``.

    A maximum-likelihood fit records ``'posterior'``. Objects fitted by the least-squares method, or
    built by hand, carry no record and keep the nearest-threshold rule they were fitted with.
    """
    return adata.uns.get('scHopfield', {}).get('sigmoid_assignment', 'nearest')


def assign_regime(adata, X, genes=None):
    """Hill component of each entry of ``X`` (cells by genes) under the object's own rule.

    Returns an ``int8`` array in which 1 selects component 2, or ``None`` for a single-Hill fit.
    """
    from .math import hill_regime
    from .hill_mle import posterior_regime
    k1, n1, k2, n2, a = get_hill_params(adata, genes, with_mix=True)
    if k2 is None:
        return None
    X = np.asarray(X, dtype=float)
    k1, n1, k2, n2, a = (np.asarray(z, dtype=float)[None, :] for z in (k1, n1, k2, n2, a))
    if regime_rule(adata) == 'posterior':
        tau = None
        if 'sigmoid_active_min' in adata.var.columns:
            idx = slice(None) if genes is None else genes
            tau = adata.var['sigmoid_active_min'].values[idx].astype(float)[None, :]
        return posterior_regime(X, k1, n1, k2, n2, a, tau)
    return hill_regime(X, k1, k2)


def observed_regime(adata, genes=None, spliced_key='Ms'):
    """Hill component of every cell for every gene, read once from the observed state.

    Returns an ``int8`` array ``(n_cells, n_genes)`` in which 1 selects component 2, or ``None``
    for a single-Hill fit. The assignment follows the object's own rule (:func:`regime_rule`). It is
    the regime every evaluation away from the observed state should hold fixed, so that a cell keeps
    the mode it was assigned to when an integration, a clamp or a finite difference moves it.
    """
    if get_hill_params(adata, genes)[2] is None:
        return None
    X = to_numpy(get_matrix(adata, spliced_key, genes=genes))
    return assign_regime(adata, X, genes)


def get_hill_params(adata, genes=None, with_mix=False):
    """Return ``(k1, n1, k2, n2)`` for the requested genes.

    ``k2`` and ``n2`` are ``None`` when the object was fitted single-Hill, or when no gene
    in the selection was accepted as two-component, so callers can pass the result straight
    into :func:`scHopfield._utils.math.sigmoid_regime` without branching. ``with_mix`` appends the
    mixture weight of component 1 (``None`` with the other two), which the posterior rule needs.

    Single-Hill genes inside a bimodal fit are stored with ``mix = 1`` and component 2
    copied from component 1, so they are already inert under the regime switch.
    """
    idx = slice(None) if genes is None else genes
    k1 = adata.var['sigmoid_threshold'].values[idx]
    n1 = adata.var['sigmoid_exponent'].values[idx]
    single = (k1, n1, None, None, None) if with_mix else (k1, n1, None, None)
    if 'sigmoid_mix' not in adata.var.columns:
        return single
    mix = adata.var['sigmoid_mix'].values[idx]
    if not bool((mix < 1 - 1e-9).any()):
        return single
    out = (k1, n1, adata.var['sigmoid_threshold2'].values[idx],
           adata.var['sigmoid_exponent2'].values[idx])
    return out + (mix,) if with_mix else out
