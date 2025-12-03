"""I/O utility functions for scHopfield."""

import numpy as np
import scipy.sparse as sp
from typing import Union, List, Optional


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


def get_cluster_key(adata):
    """
    Get cluster key used in previous scHopfield analysis.

    Parameters
    ----------
    adata : AnnData
        Annotated data object

    Returns
    -------
    str
        Cluster key from analysis
    """
    if 'scHopfield' not in adata.uns:
        raise ValueError("No scHopfield analysis found in adata")
    if 'cluster_key' not in adata.uns['scHopfield']:
        raise ValueError("No cluster_key found in scHopfield analysis")
    return adata.uns['scHopfield']['cluster_key']
