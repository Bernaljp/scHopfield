"""Network comparison and analysis."""

import numpy as np
import pandas as pd
import itertools
from typing import Optional
from anndata import AnnData

from .._utils.io import get_genes_used


def network_correlations(
    adata: AnnData,
    cluster_key: str = 'cell_type',
    copy: bool = False
) -> Optional[AnnData]:
    """
    Compute various similarity metrics between cluster interaction networks.

    Adapted from Landscape.network_correlations.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with fitted interaction matrices
    copy : bool, optional (default: False)
        If True, return a copy instead of modifying in-place

    Returns
    -------
    AnnData or None
        Returns adata if copy=True, otherwise None.
        Adds to adata.uns['scHopfield']['network_correlations']:
        - 'jaccard': Jaccard index (binary overlap)
        - 'hamming': Hamming distance (binary difference)
        - 'euclidean': Euclidean distance (continuous)
        - 'pearson': Pearson correlation (continuous)
        - 'pearson_bin': Pearson correlation (binary)
        - 'mean_col_corr': Mean column-wise correlation
        - 'singular': Singular value distance
    """
    adata = adata.copy() if copy else adata

    keys = adata.obs[cluster_key].unique()

    # Initialize result DataFrames
    jaccard = pd.DataFrame(index=keys, columns=keys, data=1.0)
    hamming = pd.DataFrame(index=keys, columns=keys, data=0.0)
    pearson = pd.DataFrame(index=keys, columns=keys, data=1.0)
    pearson_bin = pd.DataFrame(index=keys, columns=keys, data=1.0)
    euclidean = pd.DataFrame(index=keys, columns=keys, data=0.0)
    mean_col = pd.DataFrame(index=keys, columns=keys, data=1.0)
    singular = pd.DataFrame(index=keys, columns=keys, data=0.0)

    # Compute singular values for each network
    svs = {k: np.linalg.svd(adata.varp[f'W_{k}'], compute_uv=False) for k in keys}

    # Compute pairwise metrics
    for k1, k2 in itertools.combinations(keys, 2):
        w1 = adata.varp[f'W_{k1}']
        w2 = adata.varp[f'W_{k2}']
        bw1 = np.sign(w1)
        bw2 = np.sign(w2)

        # Pearson correlation
        pearson.loc[k1, k2] = pearson.loc[k2, k1] = np.corrcoef(w1.ravel(), w2.ravel())[0, 1]

        # Pearson correlation (binary)
        pearson_bin.loc[k1, k2] = pearson_bin.loc[k2, k1] = np.corrcoef(bw1.ravel(), bw2.ravel())[0, 1]

        # Euclidean distance
        euclidean.loc[k1, k2] = euclidean.loc[k2, k1] = np.linalg.norm(w1 - w2)

        # Hamming distance
        hamming.loc[k1, k2] = hamming.loc[k2, k1] = np.count_nonzero(bw1 != bw2)

        # Jaccard index
        intersection = np.logical_and(bw1, bw2)
        union = np.logical_or(bw1, bw2)
        jaccard.loc[k1, k2] = jaccard.loc[k2, k1] = intersection.sum() / union.sum()

        # Mean column-wise correlation
        mean_col_corr = np.mean(np.diag(np.corrcoef(w1, w2, rowvar=False)[:w1.shape[0], :w1.shape[0]]))
        mean_col.loc[k1, k2] = mean_col.loc[k2, k1] = mean_col_corr

        # Singular value distance
        singular.loc[k1, k2] = singular.loc[k2, k1] = np.linalg.norm(svs[k1] - svs[k2])

    # Store all metrics
    if 'network_correlations' not in adata.uns['scHopfield']:
        adata.uns['scHopfield']['network_correlations'] = {}

    adata.uns['scHopfield']['network_correlations']['jaccard'] = jaccard
    adata.uns['scHopfield']['network_correlations']['hamming'] = hamming
    adata.uns['scHopfield']['network_correlations']['euclidean'] = euclidean
    adata.uns['scHopfield']['network_correlations']['pearson'] = pearson
    adata.uns['scHopfield']['network_correlations']['pearson_bin'] = pearson_bin
    adata.uns['scHopfield']['network_correlations']['mean_col_corr'] = mean_col
    adata.uns['scHopfield']['network_correlations']['singular'] = singular

    return adata if copy else None
