import numpy as np
import pandas as pd
from typing import Optional
import logging
import itertools
from anndata import AnnData
from schopfield._core.landscape import Landscape
from schopfield.utils.data import to_numpy
from schopfield.tools.analysis import get_matrix
import hoggorm.mat_corr_coeff

logger = logging.getLogger(__name__)

def energy_genes_correlation(landscape: 'Landscape') -> None:
    """Compute correlations between energy components and gene expression for each cluster.

    Calculates Pearson correlations between total, interaction, degradation, and bias energies
    and gene expression profiles, storing results in landscape attributes.

    Args:
        landscape: Landscape object containing adata, W, and energy attributes.

    Raises:
        ValueError: If W, cluster_key, spliced_matrix_key, genes, or energy attributes are not initialized.

    Notes:
        Stores results in landscape.correlation, correlation_interaction, correlation_degradation,
        and correlation_bias. Requires energy attributes (E, E_interaction, E_degradation, E_bias)
        to be populated, e.g., via get_energies.
    """
    logger.info("Computing energy-gene correlations")

    # Validate parameters
    if not landscape.W or not landscape.cluster_key or not landscape.spliced_matrix_key or landscape.genes is None or len(landscape.genes) == 0:
        raise ValueError("Required parameters (W, cluster_key, spliced_matrix_key, genes) not initialized")
    if not all(hasattr(landscape, attr) for attr in ['E', 'E_interaction', 'E_degradation', 'E_bias']):
        raise ValueError("Energy attributes (E, E_interaction, E_degradation, E_bias) not initialized")

    # Initialize energy array
    energies = np.zeros((4, landscape.adata.n_obs))

    # Initialize correlation dictionaries
    landscape.correlation = {}
    landscape.correlation_interaction = {}
    landscape.correlation_degradation = {}
    landscape.correlation_bias = {}

    # Compute correlations for each cluster
    for k in landscape.W.keys():
        if k == 'all':
            continue
        cells = landscape.adata.obs[landscape.cluster_key] == k
        n_cells = sum(cells)

        # Assign energies
        energies[0, cells] = landscape.E.get(k, np.zeros(n_cells))
        energies[1, cells] = landscape.E_interaction.get(k, np.zeros(n_cells))
        energies[2, cells] = landscape.E_degradation.get(k, np.zeros(n_cells))
        energies[3, cells] = landscape.E_bias.get(k, np.zeros(n_cells))

        # Extract expression data
        X = to_numpy(get_matrix(landscape.adata, landscape.spliced_matrix_key, genes=landscape.genes)[cells].T)

        # Compute correlations
        correlations = np.nan_to_num(np.corrcoef(np.vstack((energies[:, cells], X)))[:4, 4:])
        landscape.correlation[k], landscape.correlation_interaction[k], landscape.correlation_degradation[k], landscape.correlation_bias[k] = correlations

    # Compute correlations for 'all'
    X = to_numpy(get_matrix(landscape.adata, landscape.spliced_matrix_key, genes=landscape.genes).T)
    correlations = np.nan_to_num(np.corrcoef(np.vstack((energies, X)))[:4, 4:])
    landscape.correlation['all'], landscape.correlation_interaction['all'], landscape.correlation_degradation['all'], landscape.correlation_bias['all'] = correlations

def celltype_correlation(landscape: 'Landscape', modified: bool = True, all_genes: bool = False) -> None:
    """Compute correlation between cell types based on gene expression profiles.

    Uses RV or modified RV2 coefficient from hoggorm to calculate pairwise correlations between
    cell types, storing results in a DataFrame.

    Args:
        landscape: Landscape object containing adata and parameters.
        modified: If True, use modified RV2 coefficient; otherwise, use RV coefficient. Defaults to True.
        all_genes: If True, use all genes; otherwise, use landscape.genes. Defaults to False.

    Raises:
        ValueError: If cluster_key, spliced_matrix_key, or genes (if not all_genes) are not initialized.
        ImportError: If hoggorm.mat_corr_coeff is not available.

    Notes:
        Stores results in landscape.cells_correlation as a pandas DataFrame.
        Requires hoggorm for RV coefficients.
    """
    logger.info(f"Computing cell type correlations (modified={modified}, all_genes={all_genes})")

    # Validate parameters
    if not landscape.cluster_key or not landscape.spliced_matrix_key:
        raise ValueError("Required parameters (cluster_key, spliced_matrix_key) not initialized")
    if not all_genes and (landscape.genes is None or len(landscape.genes) == 0):
        raise ValueError("Genes not initialized; set all_genes=True or provide genes")

    # Get unique cell types
    keys = landscape.adata.obs[landscape.cluster_key].unique()

    # Select correlation function
    try:
        corr_f = hoggorm.mat_corr_coeff.RV2coeff if modified else hoggorm.mat_corr_coeff.RVcoeff
    except AttributeError:
        raise ImportError("hoggorm.mat_corr_coeff not available; install with 'pip install hoggorm'")

    # Initialize correlation DataFrame
    rv = pd.DataFrame(index=keys, columns=keys, data=1.0)

    # Get expression data
    genes_to_consider = None if all_genes else landscape.genes
    counts = get_matrix(landscape.adata, landscape.spliced_matrix_key, genes=genes_to_consider)

    # Compute pairwise correlations
    for k1, k2 in itertools.combinations(keys, 2):
        expr_k1 = to_numpy(counts[landscape.adata.obs[landscape.cluster_key] == k1])
        expr_k2 = to_numpy(counts[landscape.adata.obs[landscape.cluster_key] == k2])
        rv.loc[k1, k2] = rv.loc[k2, k1] = corr_f([expr_k1.T, expr_k2.T])[0, 1]

    # Store results
    landscape.cells_correlation = rv

def network_correlations(landscape: 'Landscape') -> None:
    """Compute correlations and distances between interaction networks of cell types.

    Calculates various metrics (Jaccard, Hamming, Euclidean, Pearson, etc.) between interaction
    matrices W for each cell type, storing results in DataFrames.

    Args:
        landscape: Landscape object containing adata and W.

    Raises:
        ValueError: If W or cluster_key are not initialized.

    Notes:
        Stores results in landscape.jaccard, hamming, euclidean, pearson, pearson_bin,
        mean_col_corr, and singular.
    """
    logger.info("Computing network correlations")

    # Validate parameters
    if not landscape.W or not landscape.cluster_key:
        raise ValueError("Required parameters (W, cluster_key) not initialized")

    # Get unique cell types
    keys = landscape.adata.obs[landscape.cluster_key].unique()

    # Initialize DataFrames
    jaccard = pd.DataFrame(index=keys, columns=keys, data=1.0)
    hamming = pd.DataFrame(index=keys, columns=keys, data=0.0)
    pearson = pd.DataFrame(index=keys, columns=keys, data=1.0)
    pearson_bin = pd.DataFrame(index=keys, columns=keys, data=1.0)
    euclidean = pd.DataFrame(index=keys, columns=keys, data=0.0)
    mean_col = pd.DataFrame(index=keys, columns=keys, data=1.0)
    singular = pd.DataFrame(index=keys, columns=keys, data=0.0)

    # Compute singular values
    svs = {k: np.linalg.svd(landscape.W[k], compute_uv=False) for k in keys}

    # Compute pairwise metrics
    for k1, k2 in itertools.combinations(keys, 2):
        w1, w2 = landscape.W[k1], landscape.W[k2]
        bw1, bw2 = np.sign(w1), np.sign(w2)

        # Pearson correlation
        pearson.loc[k1, k2] = pearson.loc[k2, k1] = np.corrcoef(w1.ravel(), w2.ravel())[0, 1]

        # Pearson correlation for binary
        pearson_bin.loc[k1, k2] = pearson_bin.loc[k2, k1] = np.corrcoef(bw1.ravel(), bw2.ravel())[0, 1]

        # Euclidean distance
        euclidean.loc[k1, k2] = euclidean.loc[k2, k1] = np.linalg.norm(w1 - w2)

        # Hamming distance
        hamming.loc[k1, k2] = hamming.loc[k2, k1] = np.count_nonzero(bw1 != bw2)

        # Jaccard index
        intersection = np.logical_and(bw1, bw2).sum()
        union = np.logical_or(bw1, bw2).sum()
        jaccard.loc[k1, k2] = jaccard.loc[k2, k1] = intersection / union if union > 0 else 1.0

        # Mean column-wise Pearson correlation
        corr_matrix = np.corrcoef(w1, w2, rowvar=False)[:w1.shape[0], w1.shape[0]:]
        mean_col.loc[k1, k2] = mean_col.loc[k2, k1] = np.mean(np.diag(corr_matrix))

        # Singular value distance
        singular.loc[k1, k2] = singular.loc[k2, k1] = np.linalg.norm(svs[k1] - svs[k2])

    # Store results
    landscape.jaccard = jaccard
    landscape.hamming = hamming
    landscape.euclidean = euclidean
    landscape.pearson = pearson
    landscape.pearson_bin = pearson_bin
    landscape.mean_col_corr = mean_col
    landscape.singular = singular