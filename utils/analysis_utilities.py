"""
Analysis utility functions for scHopfield package.

This module contains utility functions commonly used in analysis workflows,
extracted from notebook implementations to provide reusable components.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List


def change_spines(ax: Union[plt.Axes, List[plt.Axes]]) -> None:
    """
    Modify axes spines for better visualization.

    Args:
        ax: Matplotlib axes object or list of axes objects
    """
    # Handle case where ax is a list (from some plotting functions)
    if isinstance(ax, list):
        axes_list = ax
    else:
        axes_list = [ax]

    for single_ax in axes_list:
        # Skip if not a valid axes object
        if not hasattr(single_ax, 'get_children'):
            continue

        for ch in single_ax.get_children():
            try:
                ch.set_alpha(0.5)
            except:
                continue

        for spine in single_ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
            spine.set_alpha(1)


def extract_cluster_colors(adata, cluster_key: str) -> Dict[str, np.ndarray]:
    """
    Extract cluster colors from adata visualization.

    Args:
        adata: AnnData object
        cluster_key: Key for cluster information in adata.obs

    Returns:
        Dictionary mapping cluster names to color arrays
    """
    # Extract colors
    colors = {}
    unique_clusters = adata.obs[cluster_key].unique()

    for i, cluster in enumerate(unique_clusters):
        # Get first cell of this cluster to extract color
        cluster_mask = adata.obs[cluster_key] == cluster
        if cluster_mask.any():
            # Convert tuple to list to allow modification
            color_array = list(plt.cm.tab10(i)[:4])  # RGBA values
            color_array[3] = 1  # Set alpha to 1
            colors[cluster] = np.array(color_array)

    return colors


def prepare_scaffold_matrix(adata, base_GRN: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare scaffold matrix from base GRN data.

    Args:
        adata: AnnData object
        base_GRN: Base gene regulatory network dataframe

    Returns:
        Scaffold matrix as DataFrame
    """
    # Get the list of genes to use (keep the original names for later)
    genes = adata.var.index[adata.var['use_for_dynamics']]
    genes_lower = genes.str.lower()

    # Prepare scaffold matrix with original names as index/columns
    scaffold = pd.DataFrame(0, index=genes, columns=genes)

    # Lowercase mappings for alignment
    tf_lc = [c.lower() for c in base_GRN.columns if c != 'gene_short_name']
    tf2real = {c.lower(): c for c in base_GRN.columns if c != 'gene_short_name'}
    target_lc = [g.lower() for g in base_GRN['gene_short_name']]
    target2real = {g.lower(): g for g in base_GRN['gene_short_name']}
    genes_lc_to_real = dict(zip(genes_lower, genes))

    # Keep only TFs and targets that are present in the adata genes
    selected_tfs = [t for t in tf_lc if t in genes_lc_to_real]
    selected_targets = [t for t in target_lc if t in genes_lc_to_real]

    # Build a scaffold DataFrame in lowercase for easy alignment
    scaffold_lc = pd.DataFrame(0, index=genes_lower, columns=genes_lower)

    # Lowercase the GRN gene_short_name for merging
    base_GRN_lc = base_GRN.copy()
    base_GRN_lc['gene_short_name'] = base_GRN_lc['gene_short_name'].str.lower()
    base_GRN_lc.columns = [c.lower() if c != 'gene_short_name' else c for c in base_GRN_lc.columns]

    # Fill the values using DataFrame broadcasting (much more efficient!)
    common_tfs = [t for t in selected_tfs if t in base_GRN_lc.columns]
    common_targets = list(set(selected_targets) & set(base_GRN_lc['gene_short_name']))

    sub_grn = base_GRN_lc.set_index('gene_short_name').loc[common_targets, common_tfs]
    scaffold_lc.loc[common_tfs, common_targets] = sub_grn.T.values  # transpose due to indexing

    # Map back to original-case scaffold
    idx_map = [genes_lc_to_real[lc] for lc in scaffold_lc.index]
    col_map = [genes_lc_to_real[lc] for lc in scaffold_lc.columns]
    scaffold.loc[idx_map, col_map] = scaffold_lc.values

    print(f"TFs in scaffold: {len(common_tfs)}")
    print(f"Target genes in scaffold: {len(common_targets)}")
    return scaffold


def get_correlation_table(ls, n_top_genes: int = 20, which_correlation: str = 'total') -> pd.DataFrame:
    """
    Get correlation table for top genes.

    Args:
        ls: LandscapeAnalyzer instance
        n_top_genes: Number of top genes to include
        which_correlation: Type of correlation ('total', 'interaction', etc.)

    Returns:
        DataFrame with top correlated genes per cluster
    """
    corr = 'correlation_'+which_correlation.lower() if which_correlation.lower()!='total' else 'correlation'
    assert hasattr(ls, corr), f'No {corr} attribute found in Landscape object'
    corrs_dict = getattr(ls, corr)
    order = ls.adata.obs[ls.cluster_key].unique()
    df = pd.DataFrame(index=range(n_top_genes), columns=pd.MultiIndex.from_product([order, ['Gene', 'Correlation']]))
    for k in order:
        corrs = corrs_dict[k]
        indices = np.argsort(corrs)[::-1][:n_top_genes]
        genes = ls.gene_names[indices]
        corrs = corrs[indices]
        df[(k, 'Gene')] = genes
        df[(k, 'Correlation')] = corrs
    return df