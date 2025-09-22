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
    # Ensure case-insensitive handling of gene names
    genes_to_use = list(adata.var['use_for_dynamics'].values)
    scaffold = pd.DataFrame(0, index=adata.var.index[adata.var['use_for_dynamics']],
                          columns=adata.var.index[adata.var['use_for_dynamics']])

    # Convert gene names to lowercase for case-insensitive comparison
    tfs = list(set(base_GRN.columns.str.lower()) & set(scaffold.index.str.lower()))
    target_genes = list(set(base_GRN['gene_short_name'].str.lower().values) & set(scaffold.columns.str.lower()))

    # Create a mapping from lowercase to original case
    index_mapping = {gene.lower(): gene for gene in scaffold.index}
    column_mapping = {gene.lower(): gene for gene in scaffold.columns}
    grn_tf_mapping = {gene.lower(): gene for gene in base_GRN.columns if gene != 'gene_short_name'}
    grn_target_mapping = {gene.lower(): gene for gene in base_GRN['gene_short_name'].values}

    # Populate the scaffold matrix with case-insensitive matching
    for tf_lower in tfs:
        tf_original = index_mapping[tf_lower]
        grn_tf_original = grn_tf_mapping[tf_lower]

        for target_lower in target_genes:
            target_original = column_mapping[target_lower]
            grn_target_original = grn_target_mapping[target_lower]

            # Find the value in the base_GRN
            mask = base_GRN['gene_short_name'] == grn_target_original
            if mask.any():
                value = base_GRN.loc[mask, grn_tf_original].values[0]
                scaffold.loc[tf_original, target_original] = value

    print(f"TFs in scaffold: {len(tfs)}")
    print(f"Target genes in scaffold: {len(target_genes)}")

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