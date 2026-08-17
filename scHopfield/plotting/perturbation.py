"""
Visualization functions for perturbation simulation results.

References
----------
Logic for the transition vector field is inspired by the perturbation
simulation workflow in CellOracle:
Kamimoto et al. (2023). Nature. https://doi.org/10.1038/s41586-022-05688-9
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Union, Tuple
from anndata import AnnData

from .._utils.io import get_genes_used


def _get_perturbed_genes(adata):
    """Get list of perturbed gene names from adata.uns."""
    if 'scHopfield' in adata.uns and 'perturb_condition' in adata.uns['scHopfield']:
        return list(adata.uns['scHopfield']['perturb_condition'].keys())
    return []


def _filter_perturbed_genes(gene_names, perturbed_genes):
    """Return mask for genes that are NOT perturbed."""
    return ~np.isin(gene_names, perturbed_genes)


def plot_perturbation_effect_heatmap(
    adata: AnnData,
    cluster_key: str = 'cell_type',
    n_genes: int = 30,
    figsize: Tuple[float, float] = (12, 8),
    cmap: str = 'RdBu_r',
    center: float = 0,
    cluster_cols: bool = True,
    cluster_rows: bool = False,
    order: Optional[List[str]] = None,
    colors: Optional[Dict[str, str]] = None
) -> sns.matrix.ClusterGrid:
    """
    Plot heatmap of perturbation effects across clusters and genes.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with simulation results (delta_X layer)
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    n_genes : int, optional (default: 30)
        Number of top affected genes to show
    figsize : tuple, optional
        Figure size
    cmap : str, optional (default: 'RdBu_r')
        Colormap
    center : float, optional (default: 0)
        Center value for colormap
    cluster_cols : bool, optional (default: True)
        If True, cluster columns (clusters) with dendrogram
    cluster_rows : bool, optional (default: False)
        If True, cluster rows (genes) with dendrogram
    order : list, optional
        Order of clusters to plot. Ignored if cluster_cols=True.
    colors : dict, optional
        Dictionary mapping cluster names to colors for column color bar

    Returns
    -------
    sns.matrix.ClusterGrid
        ClusterGrid object with the heatmap
    """
    if 'delta_X' not in adata.layers:
        raise ValueError("No simulation results found. Run simulate_shift first.")

    genes = get_genes_used(adata)
    gene_names = adata.var_names[genes].values
    delta_X = adata.layers['delta_X'][:, genes]

    clusters = adata.obs[cluster_key].unique()

    # Calculate mean delta per cluster and gene
    cluster_effects = {}
    for cluster in clusters:
        mask = (adata.obs[cluster_key] == cluster).values
        cluster_effects[cluster] = delta_X[mask, :].mean(axis=0)

    df = pd.DataFrame(cluster_effects, index=gene_names)

    # Exclude perturbed genes
    perturbed_genes = _get_perturbed_genes(adata)
    df = df.loc[~df.index.isin(perturbed_genes)]

    # Select top genes by variance across clusters
    gene_variance = df.var(axis=1)
    top_genes = gene_variance.nlargest(n_genes).index
    df = df.loc[top_genes]

    # Apply order if specified and not clustering
    if order is not None and not cluster_cols:
        order = [c for c in order if c in df.columns]
        df = df[order]

    # Create column colors if colors dict provided
    col_colors = None
    if colors is not None:
        col_colors = pd.Series([colors.get(c, '#cccccc') for c in df.columns], index=df.columns)

    # Plot with clustermap
    g = sns.clustermap(
        df, cmap=cmap, center=center, figsize=figsize,
        col_cluster=cluster_cols, row_cluster=cluster_rows,
        xticklabels=True, yticklabels=True,
        cbar_kws={'label': 'Mean Δ Expression'},
        col_colors=col_colors,
        dendrogram_ratio=(0.1, 0.15)
    )

    g.ax_heatmap.set_xlabel('Cluster', fontsize=11)
    g.ax_heatmap.set_ylabel('Gene', fontsize=11)

    # Get perturbation info for title
    title = 'Perturbation Effects by Cluster'
    if 'scHopfield' in adata.uns and 'perturb_condition' in adata.uns['scHopfield']:
        perturb = adata.uns['scHopfield']['perturb_condition']
        perturb_str = ', '.join([f"{k}={v}" for k, v in perturb.items()])
        title = f'Perturbation Effects: {perturb_str}'

    g.fig.suptitle(title, fontsize=12, fontweight='bold', y=1.02)

    return g


def plot_perturbation_magnitude(
    adata: AnnData,
    cluster_key: str = 'cell_type',
    basis: str = 'umap',
    figsize: Tuple[float, float] = (12, 5),
    cmap: str = 'viridis',
    vmax: Optional[float] = None,
    order: Optional[List[str]] = None,
    colors: Optional[Dict[str, str]] = None
) -> plt.Figure:
    """
    Plot perturbation magnitude on embedding and as boxplot.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with simulation results
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    basis : str, optional (default: 'umap')
        Embedding basis
    figsize : tuple, optional
        Figure size
    cmap : str, optional (default: 'viridis')
        Colormap for scatter plot
    vmax : float, optional
        Maximum value for colormap
    order : list, optional
        Order of clusters in boxplot. If None, sorts by median magnitude.
    colors : dict, optional
        Dictionary mapping cluster names to colors for boxplot

    Returns
    -------
    plt.Figure
        Figure with plots
    """
    if 'delta_X' not in adata.layers:
        raise ValueError("No simulation results found. Run simulate_shift first.")

    # Calculate magnitude (excluding perturbed genes)
    genes = get_genes_used(adata)
    gene_names = adata.var_names[genes].values
    perturbed_genes = _get_perturbed_genes(adata)
    gene_mask = _filter_perturbed_genes(gene_names, perturbed_genes)

    delta_X = adata.layers['delta_X'][:, genes][:, gene_mask]
    magnitude = np.linalg.norm(delta_X, axis=1)
    adata.obs['perturbation_magnitude'] = magnitude

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Scatter plot on embedding
    embedding_key = f'X_{basis}'
    if embedding_key in adata.obsm:
        coords = adata.obsm[embedding_key]
        sc = axes[0].scatter(
            coords[:, 0], coords[:, 1],
            c=magnitude, cmap=cmap, s=10, alpha=0.7,
            vmax=vmax, rasterized=True
        )
        axes[0].set_xlabel(f'{basis.upper()} 1', fontsize=10)
        axes[0].set_ylabel(f'{basis.upper()} 2', fontsize=10)
        axes[0].set_title('Perturbation Magnitude', fontsize=12, fontweight='bold')
        plt.colorbar(sc, ax=axes[0], label='||Δx||')
        axes[0].axis('equal')
    else:
        axes[0].text(0.5, 0.5, f'Embedding {embedding_key} not found',
                    ha='center', va='center', transform=axes[0].transAxes)

    # Boxplot by cluster
    df = pd.DataFrame({
        'Cluster': adata.obs[cluster_key].values,
        'Magnitude': magnitude
    })

    # Determine order
    if order is None:
        order = df.groupby('Cluster')['Magnitude'].median().sort_values(ascending=False).index.tolist()

    # Create palette from colors dict
    palette = None
    if colors is not None:
        palette = [colors.get(c, '#cccccc') for c in order]

    sns.boxplot(data=df, x='Cluster', y='Magnitude', order=order, palette=palette, ax=axes[1])
    axes[1].set_xlabel('Cluster', fontsize=10)
    axes[1].set_ylabel('Perturbation Magnitude', fontsize=10)
    axes[1].set_title('Effect by Cluster', fontsize=12, fontweight='bold')
    if len(order) > 5:
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def plot_gene_response(
    adata: AnnData,
    genes: Union[str, List[str]],
    cluster_key: str = 'cell_type',
    figsize: Optional[Tuple[float, float]] = None,
    order: Optional[List[str]] = None,
    colors: Optional[Dict[str, str]] = None
) -> plt.Figure:
    """
    Plot expression change for specific genes across clusters.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with simulation results
    genes : str or list
        Gene(s) to plot
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    figsize : tuple, optional
        Figure size
    order : list, optional
        Order of clusters to plot. If None, sorts by median effect.
    colors : dict, optional
        Dictionary mapping cluster names to colors

    Returns
    -------
    plt.Figure
        Figure with plots
    """
    if 'delta_X' not in adata.layers:
        raise ValueError("No simulation results found. Run simulate_shift first.")

    if isinstance(genes, str):
        genes = [genes]

    n_genes = len(genes)
    if figsize is None:
        figsize = (5 * n_genes, 5)

    fig, axes = plt.subplots(1, n_genes, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    sch_genes = get_genes_used(adata)
    gene_names = adata.var_names[sch_genes].values

    for i, gene in enumerate(genes):
        ax = axes[i]

        if gene not in gene_names:
            ax.text(0.5, 0.5, f'{gene} not in analysis',
                   ha='center', va='center', transform=ax.transAxes)
            continue

        gene_idx = np.where(gene_names == gene)[0][0]
        delta = adata.layers['delta_X'][:, sch_genes[gene_idx]]

        df = pd.DataFrame({
            'Cluster': adata.obs[cluster_key].values,
            'Δ Expression': delta
        })

        # Determine order
        if order is None:
            plot_order = df.groupby('Cluster')['Δ Expression'].median().sort_values().index.tolist()
        else:
            plot_order = order

        # Create palette from colors dict
        palette = None
        if colors is not None:
            palette = [colors.get(c, '#cccccc') for c in plot_order]

        sns.violinplot(data=df, x='Cluster', y='Δ Expression',
                      order=plot_order, ax=ax, palette=palette, inner='box')
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_title(f'{gene}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Cluster', fontsize=10)
        ax.set_ylabel('Δ Expression', fontsize=10)
        if len(plot_order) > 5:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def plot_top_affected_genes_bar(
    adata: AnnData,
    n_genes: int = 20,
    cluster: Optional[str] = None,
    cluster_key: str = 'cell_type',
    figsize: Tuple[float, float] = (10, 8),
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Bar plot of top affected genes showing direction and magnitude.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with simulation results
    n_genes : int, optional (default: 20)
        Number of genes to show
    cluster : str, optional
        Specific cluster to analyze. If None, uses all cells.
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    figsize : tuple, optional
        Figure size
    ax : plt.Axes, optional
        Axes to plot on

    Returns
    -------
    plt.Axes
        Axes with plot
    """
    if 'delta_X' not in adata.layers:
        raise ValueError("No simulation results found. Run simulate_shift first.")

    genes = get_genes_used(adata)
    gene_names = adata.var_names[genes].values

    # Exclude perturbed genes
    perturbed_genes = _get_perturbed_genes(adata)
    gene_mask = _filter_perturbed_genes(gene_names, perturbed_genes)
    gene_names = gene_names[gene_mask]

    if cluster is not None:
        mask = (adata.obs[cluster_key] == cluster).values
        delta_X = adata.layers['delta_X'][mask, :][:, genes][:, gene_mask]
        title_suffix = f' ({cluster})'
    else:
        delta_X = adata.layers['delta_X'][:, genes][:, gene_mask]
        title_suffix = ' (All cells)'

    mean_delta = delta_X.mean(axis=0)
    abs_delta = np.abs(mean_delta)

    # Get top genes
    top_idx = np.argsort(abs_delta)[-n_genes:]
    top_genes = gene_names[top_idx]
    top_values = mean_delta[top_idx]

    # Sort by actual value (not absolute)
    sort_idx = np.argsort(top_values)
    top_genes = top_genes[sort_idx]
    top_values = top_values[sort_idx]

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    colors = ['#1f77b4' if v > 0 else '#d62728' for v in top_values]
    ax.barh(range(len(top_genes)), top_values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(top_genes)))
    ax.set_yticklabels(top_genes)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Mean Δ Expression', fontsize=11)
    ax.set_ylabel('Gene', fontsize=11)
    ax.set_title(f'Top Affected Genes{title_suffix}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#1f77b4', edgecolor='black', label='Upregulated'),
                       Patch(facecolor='#d62728', edgecolor='black', label='Downregulated')]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    return ax


def plot_simulation_comparison(
    adata: AnnData,
    gene: str,
    cluster_key: str = 'cell_type',
    figsize: Tuple[float, float] = (12, 5),
    order: Optional[List[str]] = None,
    colors: Optional[Dict[str, str]] = None
) -> plt.Figure:
    """
    Compare original and simulated expression for a gene.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with simulation results
    gene : str
        Gene to compare
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    figsize : tuple, optional
        Figure size
    order : list, optional
        Order of clusters in boxplot. If None, sorts by median delta.
    colors : dict, optional
        Dictionary mapping cluster names to colors for boxplot

    Returns
    -------
    plt.Figure
        Figure with comparison plots
    """
    genes = get_genes_used(adata)
    gene_names = adata.var_names[genes].values

    if gene not in gene_names:
        raise ValueError(f"Gene '{gene}' not found in analysis")

    gene_idx = np.where(gene_names == gene)[0][0]
    sch_gene_idx = genes[gene_idx]

    # Get expression data from spliced layer
    spliced_key = adata.uns.get('scHopfield', {}).get('spliced_key', 'Ms')
    if spliced_key in adata.layers:
        original = adata.layers[spliced_key][:, sch_gene_idx]
    else:
        original = adata.X[:, sch_gene_idx]
    if hasattr(original, 'toarray'):
        original = original.toarray().flatten()

    simulated = adata.layers['simulated_count'][:, sch_gene_idx]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Histogram comparison
    axes[0].hist(original, bins=50, alpha=0.5, label='Original', density=True)
    axes[0].hist(simulated, bins=50, alpha=0.5, label='Simulated', density=True)
    axes[0].set_xlabel('Expression', fontsize=10)
    axes[0].set_ylabel('Density', fontsize=10)
    axes[0].set_title(f'{gene} Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()

    # Scatter: original vs simulated
    axes[1].scatter(original, simulated, alpha=0.3, s=5, rasterized=True)
    lims = [min(original.min(), simulated.min()), max(original.max(), simulated.max())]
    axes[1].plot(lims, lims, 'r--', alpha=0.5, label='y=x')
    axes[1].set_xlabel('Original', fontsize=10)
    axes[1].set_ylabel('Simulated', fontsize=10)
    axes[1].set_title('Original vs Simulated', fontsize=12, fontweight='bold')
    axes[1].legend()

    # Delta by cluster
    delta = simulated - original
    df = pd.DataFrame({
        'Cluster': adata.obs[cluster_key].values,
        'Δ Expression': delta
    })

    # Determine order
    if order is None:
        plot_order = df.groupby('Cluster')['Δ Expression'].median().sort_values().index.tolist()
    else:
        plot_order = order

    # Create palette from colors dict
    palette = None
    if colors is not None:
        palette = [colors.get(c, '#cccccc') for c in plot_order]

    sns.boxplot(data=df, x='Cluster', y='Δ Expression', order=plot_order, palette=palette, ax=axes[2])
    axes[2].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Cluster', fontsize=10)
    axes[2].set_ylabel('Δ Expression', fontsize=10)
    axes[2].set_title('Change by Cluster', fontsize=12, fontweight='bold')
    if len(plot_order) > 5:
        axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    return fig
