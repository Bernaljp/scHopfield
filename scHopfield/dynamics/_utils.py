"""Private helper utilities for the dynamics module."""
import numpy as np
from typing import Dict, Optional, Tuple
from anndata import AnnData
from .._utils.io import get_matrix, to_numpy


def _parse_perturb_genes(gene_names, perturb_condition, validate_non_negative=False):
    """Parse perturbation dict → (indices, values) as np.ndarray pair.
    Returns empty arrays (not None) when no genes match."""
    gene_to_idx = {name: i for i, name in enumerate(gene_names)}
    indices, values = [], []
    for gene, value in perturb_condition.items():
        if gene not in gene_to_idx:
            continue
        if validate_non_negative and value < 0:
            raise ValueError(f"Perturbation value must be non-negative, got {value} for {gene}")
        indices.append(gene_to_idx[gene])
        values.append(value)
    return np.array(indices, dtype=np.int64), np.array(values, dtype=np.float64)


def _get_W_matrix(adata, cluster, use_cluster_specific=True):
    """Get W matrix with fallback to W_all. Fixes solver.py crash on missing cluster key."""
    if use_cluster_specific and f'W_{cluster}' in adata.varp:
        return adata.varp[f'W_{cluster}']
    elif 'W_all' in adata.varp:
        return adata.varp['W_all']
    raise ValueError(f"No W matrix found for cluster '{cluster}'. Run fit_interactions first.")


def _compute_x_bounds(X, x_max_percentile, multiplier=2.0):
    """Return (x_min=0.0, x_max) bounds. x_max=None if x_max_percentile is None."""
    if x_max_percentile is not None:
        return 0.0, np.percentile(X, x_max_percentile, axis=0) * multiplier
    return 0.0, None


def _update_scHopfield_uns(adata, **kwargs):
    """Merge kwargs into adata.uns['scHopfield'], creating it if absent."""
    if 'scHopfield' not in adata.uns:
        adata.uns['scHopfield'] = {}
    adata.uns['scHopfield'].update(kwargs)
