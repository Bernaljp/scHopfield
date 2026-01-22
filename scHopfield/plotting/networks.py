"""Plotting functions for network visualization."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Union
from anndata import AnnData

from .._utils.io import get_genes_used


def plot_interaction_matrix(
    adata: AnnData,
    cluster: str,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot interaction matrix heatmap.

    Parameters
    ----------
    adata : AnnData
        Annotated data object
    cluster : str
        Cluster name
    ax : plt.Axes, optional
        Axes to plot on

    Returns
    -------
    plt.Axes
        Axes with plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    W = adata.varp[f'W_{cluster}']
    im = ax.imshow(W, cmap='RdBu_r', **kwargs)
    ax.set_title(f'Interaction Matrix: {cluster}')
    plt.colorbar(im, ax=ax)

    return ax


def plot_network_centrality_rank(
    adata: AnnData,
    metric: str = 'degree_centrality_all',
    clusters: Optional[Union[str, List[str]]] = None,
    cluster_key: str = 'cell_type',
    n_genes: int = 50,
    colors: Optional[Dict[str, str]] = None,
    skip_first_n: int = 0,
    figsize: Optional[tuple] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot top genes ranked by network centrality score.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with computed centrality metrics
    metric : str, optional (default: 'degree_centrality_all')
        Centrality metric to plot. Available: 'degree_all', 'degree_centrality_all',
        'degree_in', 'degree_centrality_in', 'degree_out', 'degree_centrality_out',
        'betweenness_centrality', 'eigenvector_centrality'
    clusters : str or list, optional
        Cluster(s) to plot. If None, plots all clusters
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    n_genes : int, optional (default: 50)
        Number of top genes to show
    colors : dict, optional
        Colors for each cluster
    skip_first_n : int, optional (default: 0)
        Skip top N genes
    figsize : tuple, optional
        Figure size
    ax : plt.Axes, optional
        Axes to plot on

    Returns
    -------
    plt.Axes
        Axes with plot
    """
    genes = get_genes_used(adata)
    gene_names = adata.var.index[genes]

    if clusters is None:
        clusters = adata.obs[cluster_key].unique().tolist()
    elif isinstance(clusters, str):
        clusters = [clusters]

    n_clusters = len(clusters)
    size_per_gene = 0.2

    if ax is None:
        if figsize is None:
            figsize = (5, n_genes * size_per_gene)
        fig, ax = plt.subplots(figsize=figsize)

    for cluster in clusters:
        col_name = f'{metric}_{cluster}'
        if col_name not in adata.var.columns:
            print(f"Warning: No {metric} data for {cluster}, skipping...")
            continue

        # Get scores and sort
        scores = adata.var[col_name].values[genes]
        sorted_idx = np.argsort(scores)[::-1][skip_first_n:n_genes+skip_first_n]
        top_genes = gene_names[sorted_idx]
        top_scores = scores[sorted_idx]

        # Plot
        color = colors[cluster] if colors is not None and cluster in colors else 'tab:blue'
        ax.scatter(top_scores, range(len(top_scores)), color=color, label=cluster)

    ax.set_yticks(range(len(top_genes)), top_genes)
    ax.invert_yaxis()
    ax.set_xlabel(metric.replace('_', ' ').capitalize())
    ax.set_ylabel('Gene')

    if n_clusters > 1:
        ax.legend()

    return ax


def plot_centrality_comparison(
    adata: AnnData,
    cluster1: str,
    cluster2: str,
    metric: str = 'degree_centrality_all',
    cluster_key: str = 'cell_type',
    percentile: float = 99,
    annotate: bool = True,
    ignore_genes: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Compare network centrality scores between two clusters.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with computed centrality metrics
    cluster1 : str
        First cluster name (x-axis)
    cluster2 : str
        Second cluster name (y-axis)
    metric : str, optional (default: 'degree_centrality_all')
        Centrality metric to compare. Available: 'degree_all', 'degree_centrality_all',
        'degree_in', 'degree_centrality_in', 'degree_out', 'degree_centrality_out',
        'betweenness_centrality', 'eigenvector_centrality'
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    percentile : float, optional (default: 99)
        Percentile threshold for highlighting genes
    annotate : bool, optional (default: True)
        Whether to annotate high-scoring genes
    ignore_genes : list, optional
        List of genes to exclude from plotting
    ax : plt.Axes, optional
        Axes to plot on

    Returns
    -------
    plt.Axes
        Axes with plot
    """
    genes = get_genes_used(adata)
    gene_names = adata.var.index[genes]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Get centrality scores
    col1 = f'{metric}_{cluster1}'
    col2 = f'{metric}_{cluster2}'

    if col1 not in adata.var.columns or col2 not in adata.var.columns:
        raise ValueError(
            f"Centrality data not found. Please run sch.tl.compute_network_centrality() first."
        )

    scores1 = adata.var[col1].values[genes]
    scores2 = adata.var[col2].values[genes]

    # Remove ignored genes
    if ignore_genes is not None:
        mask = ~np.isin(gene_names, ignore_genes)
        gene_names = gene_names[mask]
        scores1 = scores1[mask]
        scores2 = scores2[mask]

    # Identify high-scoring genes
    thresh1 = np.percentile(scores1, percentile)
    thresh2 = np.percentile(scores2, percentile)
    high_genes = (scores1 >= thresh1) | (scores2 >= thresh2)

    # Plot
    ax.scatter(scores1[~high_genes], scores2[~high_genes], c='lightgray', s=2)
    ax.scatter(scores1[high_genes], scores2[high_genes], c='none', edgecolors='b')

    # Annotate
    if annotate:
        x_shift = (scores1.max() - scores1.min()) * 0.03
        y_shift = (scores2.max() - scores2.min()) * 0.03
        arrow_dict = {"width": 0.5, "headwidth": 0.5, "headlength": 1, "color": "black"}

        for gene in gene_names[high_genes]:
            idx = np.where(gene_names == gene)[0][0]
            x, y = scores1[idx], scores2[idx]
            ax.annotate(
                gene, xy=(x, y),
                xytext=(x + x_shift, y + y_shift),
                color="black",
                arrowprops=arrow_dict,
                fontsize=10
            )

    ax.set_xlabel(cluster1)
    ax.set_ylabel(cluster2)
    ax.set_title(f'{metric.replace("_", " ").capitalize()}')

    return ax


def plot_gene_centrality(
    adata: AnnData,
    gene: str,
    cluster_key: str = 'cell_type',
    metrics: Optional[List[str]] = None,
    order: Optional[List[str]] = None,
    colors: Optional[Dict[str, str]] = None,
    figsize: tuple = (12, 4)
) -> plt.Figure:
    """
    Plot network centrality scores for a specific gene across clusters.

    Compares multiple centrality metrics for a single gene across different
    clusters, useful for understanding gene importance in different contexts.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with computed centrality metrics
    gene : str
        Gene name to plot
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    metrics : list, optional
        List of centrality metrics to plot. If None, plots:
        ['degree_centrality_all', 'betweenness_centrality', 'eigenvector_centrality']
        Available metrics: 'degree_all', 'degree_centrality_all', 'degree_in',
        'degree_centrality_in', 'degree_out', 'degree_centrality_out',
        'betweenness_centrality', 'eigenvector_centrality'
    order : list, optional
        Order of clusters to display
    colors : dict, optional
        Colors for each cluster
    figsize : tuple, optional (default: (12, 4))
        Figure size

    Returns
    -------
    plt.Figure
        Figure with subplots
    """
    try:
        import seaborn as sns
    except ImportError:
        raise ImportError(
            "Seaborn is required for this plot. "
            "Install it with: pip install seaborn"
        )

    genes = get_genes_used(adata)
    gene_names = adata.var.index[genes]

    # Check if gene exists
    if gene not in gene_names:
        raise ValueError(f"Gene '{gene}' not found in the dataset")

    gene_idx = np.where(gene_names == gene)[0][0]

    if metrics is None:
        metrics = ['degree_centrality_all', 'betweenness_centrality', 'eigenvector_centrality']

    clusters = adata.obs[cluster_key].unique().tolist()
    if order is not None:
        clusters = [c for c in order if c in clusters]

    # Collect data
    data_list = []
    for cluster in clusters:
        for metric in metrics:
            col_name = f'{metric}_{cluster}'
            if col_name not in adata.var.columns:
                print(f"Warning: No {metric} data for {cluster}, skipping...")
                continue

            value = adata.var[col_name].values[genes][gene_idx]
            data_list.append({
                'cluster': cluster,
                'metric': metric.replace('_', '\n'),
                'value': value
            })

    if not data_list:
        raise ValueError("No centrality data found. Please run sch.tl.compute_network_centrality() first.")

    import pandas as pd
    df = pd.DataFrame(data_list)

    # Create figure
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, tight_layout=True)
    if len(metrics) == 1:
        axes = [axes]

    # Plot each metric
    unique_metrics = df['metric'].unique()
    for i, (ax, metric) in enumerate(zip(axes, unique_metrics)):
        subset = df[df['metric'] == metric]

        # Create stripplot
        sns.stripplot(
            data=subset,
            y='cluster',
            x='value',
            size=10,
            orient='h',
            linewidth=1,
            edgecolor='w',
            order=order if order is not None else clusters,
            palette=colors,
            ax=ax
        )

        # Style axes
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.yaxis.grid(True)
        ax.tick_params(bottom=False, left=False, right=False, top=False)

        ax.set_xlabel(metric)

        # Remove y-labels except for first plot
        if i > 0:
            ax.set_ylabel(None)
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel('Cluster')

    fig.suptitle(f'Network centrality for {gene}', fontsize=14, y=1.02)

    return fig


def plot_centrality_scatter(
    adata: AnnData,
    x_metric: str,
    y_metric: str,
    cluster_key: str = 'cell_type',
    order: Optional[List[str]] = None,
    colors: Optional[Dict[str, str]] = None,
    n_top_genes: int = 3,
    filter_threshold: Optional[tuple] = None,
    figsize: Optional[tuple] = None
) -> plt.Figure:
    """
    Plot scatter of two centrality metrics for all clusters.

    Creates a grid showing relationship between two centrality metrics
    across all clusters, annotating top genes.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with computed centrality metrics
    x_metric : str
        Centrality metric for x-axis
    y_metric : str
        Centrality metric for y-axis
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    order : list, optional
        Order of clusters to display
    colors : dict, optional
        Colors for each cluster
    n_top_genes : int, optional (default: 3)
        Number of top genes to annotate per cluster
    filter_threshold : tuple, optional
        (metric_name, operator, value) to filter genes before finding top.
        E.g., ('degree_centrality', '<', 0.5) to find high betweenness
        genes with low degree
    figsize : tuple, optional
        Figure size. If None, auto-calculated based on number of clusters

    Returns
    -------
    plt.Figure
        Figure with subplots
    """
    genes = get_genes_used(adata)
    gene_names = adata.var.index[genes]

    clusters = adata.obs[cluster_key].unique().tolist()
    if order is not None:
        clusters = [c for c in order if c in clusters]

    n_clusters = len(clusters)
    ncols = 4
    nrows = int(np.ceil(n_clusters / ncols))

    if figsize is None:
        figsize = (20, nrows * 5)

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    axs = np.atleast_1d(axs).flatten()

    for i, cluster in enumerate(clusters):
        ax = axs[i]

        # Get centrality scores
        x_col = f'{x_metric}_{cluster}'
        y_col = f'{y_metric}_{cluster}'

        if x_col not in adata.var.columns or y_col not in adata.var.columns:
            ax.text(0.5, 0.5, f'No data for\n{cluster}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            continue

        x_scores = adata.var[x_col].values[genes]
        y_scores = adata.var[y_col].values[genes]

        # Apply filter if specified
        if filter_threshold is not None:
            filter_metric, operator, threshold = filter_threshold
            filter_col = f'{filter_metric}_{cluster}'
            filter_scores = adata.var[filter_col].values[genes]

            if operator == '<':
                mask = filter_scores < threshold
            elif operator == '>':
                mask = filter_scores > threshold
            elif operator == '<=':
                mask = filter_scores <= threshold
            elif operator == '>=':
                mask = filter_scores >= threshold
            else:
                mask = np.ones(len(genes), dtype=bool)
        else:
            mask = np.ones(len(genes), dtype=bool)

        # Find top genes (within filtered set if applicable)
        if filter_threshold is not None:
            # Find top genes based on y_metric within filtered set
            filtered_indices = np.where(mask)[0]
            if len(filtered_indices) > 0:
                top_idx_within_filtered = np.argsort(y_scores[filtered_indices])[::-1][:n_top_genes]
                top_idx = filtered_indices[top_idx_within_filtered]
            else:
                top_idx = np.array([])
        else:
            top_idx = np.argsort(y_scores)[::-1][:n_top_genes]

        # Plot
        color = colors[cluster] if colors is not None and cluster in colors else 'tab:blue'
        ax.scatter(x_scores, y_scores, color=color, s=10, alpha=0.6)

        # Annotate top genes
        for idx in top_idx:
            ax.annotate(
                gene_names[idx],
                (x_scores[idx], y_scores[idx]),
                fontsize=10,
                ha='right',
                va='bottom',
                color='black'
            )

        ax.set_title(cluster)
        ax.set_xlabel(x_metric.replace('_', ' ').title())
        ax.set_ylabel(y_metric.replace('_', ' ').title())

    # Hide extra subplots
    for i in range(n_clusters, len(axs)):
        axs[i].axis('off')

    plt.tight_layout()
    return fig
