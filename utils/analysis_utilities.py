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
    # Get genes to use (preserve original order/case)
    genes = adata.var.index[adata.var['use_for_dynamics']]
    genes_lower = genes.str.lower()

    # Lowercase the relevant base_GRN data and columns for case-insensitive matching
    grn = base_GRN.copy()
    grn['gene_short_name'] = grn['gene_short_name'].str.lower()
    grn.columns = [c.lower() if c != 'gene_short_name' else c for c in grn.columns]

    # Filter only TF columns (not gene_short_name) that are present in the genes list
    tf_cols = [c for c in grn.columns if c != 'gene_short_name' and c in genes_lower.values]
    grn = grn[grn['gene_short_name'].isin(genes_lower.values)]

    # Scaffold initialized as zeros
    scaffold = pd.DataFrame(0, index=genes, columns=genes)
    # Reindex grn to match scaffold's index/column order (using lowercase for matching)
    grn = grn.set_index('gene_short_name').reindex(genes_lower, fill_value=0)[tf_cols]
    grn.columns = genes[genes_lower.isin(tf_cols)].values

    # Set tf/target values using pandas assignment and alignment
    scaffold.update(grn.T)  # ensures targets (from base_GRN rows) match scaffold cols

    print(f"TFs in scaffold: {len(tf_cols)}")
    print(f"Target genes in scaffold: {grn.shape[0]}")
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