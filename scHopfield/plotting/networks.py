"""Plotting functions for network visualization."""

import numpy as np
import pandas as pd
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

    # Collect all top genes across clusters for consistent y-axis
    all_top_genes = []
    plot_data = []

    for cluster in clusters:
        col_name = f'{metric}_{cluster}'
        if col_name not in adata.var.columns:
            print(f"Warning: No {metric} data for {cluster}, skipping...")
            continue

        # Get scores and sort
        scores = adata.var[col_name].values[genes]
        sorted_idx = np.argsort(scores)[::-1][skip_first_n:n_genes+skip_first_n]
        top_genes_cluster = gene_names[sorted_idx]
        top_scores = scores[sorted_idx]

        # Store for plotting
        plot_data.append({
            'cluster': cluster,
            'genes': top_genes_cluster,
            'scores': top_scores
        })

        # Collect unique top genes
        for gene in top_genes_cluster:
            if gene not in all_top_genes:
                all_top_genes.append(gene)

    # Limit to n_genes if we have multiple clusters
    if len(all_top_genes) > n_genes:
        all_top_genes = all_top_genes[:n_genes]

    # Plot each cluster
    for data in plot_data:
        cluster = data['cluster']
        genes_cluster = data['genes']
        scores = data['scores']

        # Map genes to y-positions based on all_top_genes
        y_positions = [all_top_genes.index(gene) for gene in genes_cluster if gene in all_top_genes]
        scores_filtered = [scores[i] for i, gene in enumerate(genes_cluster) if gene in all_top_genes]

        # Plot
        color = colors[cluster] if colors is not None and cluster in colors else 'tab:blue'
        ax.scatter(scores_filtered, y_positions, color=color, label=cluster, s=50, alpha=0.7)

    ax.set_yticks(range(len(all_top_genes)), all_top_genes)
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


def _linspace_iterator(start, stop, num):
    """Helper function for annotation positioning."""
    if num == 1:
        yield start
        return
    step = (stop - start) / float(num - 1)
    for i in range(num):
        yield start + i * step


def _annotate_points(ax, x_data, y_data, labels, offset_x_fraction=0.1, offset_y_fraction=0.1):
    """
    Helper function to annotate points with adaptive positioning.

    Parameters
    ----------
    ax : plt.Axes
        Axes to annotate on
    x_data : array-like
        X coordinates of points
    y_data : array-like
        Y coordinates of points
    labels : array-like
        Labels for each point
    offset_x_fraction : float, optional (default: 0.1)
        Horizontal offset fraction
    offset_y_fraction : float, optional (default: 0.1)
        Vertical offset fraction
    """
    n_positive = sum(y >= 0 for y in y_data)
    n_negative = sum(y < 0 for y in y_data)
    n_total = len(y_data) // 2
    frac_positive = n_positive / n_total if n_total > 0 else 1
    frac_negative = n_negative / n_total if n_total > 0 else 1
    offsets_positive = _linspace_iterator(-0.25 * offset_y_fraction * frac_positive,
                                          1.75 * offset_y_fraction * frac_positive,
                                          n_positive)
    offsets_negative = _linspace_iterator(-0.25 * offset_y_fraction * frac_negative,
                                          1.75 * offset_y_fraction * frac_negative,
                                          n_negative)
    offset_x = offset_x_fraction

    for name, x, y in zip(labels, x_data, y_data):
        offset_y = next(offsets_positive) if y >= 0 else next(offsets_negative)

        # Convert offset to display coordinates
        offset_x_data = offset_x * ax.figure.dpi
        offset_y_data = offset_y * ax.figure.dpi

        # Determine text position based on y-value
        if y < 0:
            xytext = (offset_x_data, offset_y_data)
            ha = 'left'
        else:
            xytext = (-offset_x_data, -offset_y_data)
            ha = 'right'

        # Annotate the point
        ax.annotate(name, xy=(x, y), xytext=xytext, fontsize=8, ha=ha,
                   textcoords='offset points',
                   arrowprops=dict(arrowstyle="->", color='gray', lw=0.5))


def plot_eigenvalue_spectrum(
    adata: AnnData,
    clusters: Optional[Union[str, List[str]]] = None,
    cluster_key: str = 'cell_type',
    colors: Optional[Dict[str, str]] = None,
    highlight_extremes: bool = True,
    figsize: Optional[tuple] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot eigenvalue spectrum in the complex plane.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with computed eigenanalysis
    clusters : str or list, optional
        Cluster(s) to plot. If None, plots all clusters
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    colors : dict, optional
        Colors for each cluster
    highlight_extremes : bool, optional (default: True)
        Whether to highlight eigenvalues with max/min real parts
    figsize : tuple, optional
        Figure size
    ax : plt.Axes, optional
        Axes to plot on

    Returns
    -------
    plt.Axes
        Axes with plot
    """
    if 'eigenanalysis' not in adata.uns['scHopfield']:
        raise ValueError(
            "Eigenanalysis not found. Please run sch.tl.compute_eigenanalysis() first."
        )

    if clusters is None:
        clusters = adata.obs[cluster_key].unique().tolist()
    elif isinstance(clusters, str):
        clusters = [clusters]

    if ax is None:
        if figsize is None:
            figsize = (8, 8)
        fig, ax = plt.subplots(figsize=figsize)

    for cluster in clusters:
        eigenvalues = adata.uns['scHopfield']['eigenanalysis'][f'eigenvalues_{cluster}']

        color = colors[cluster] if colors is not None and cluster in colors else 'tab:blue'
        ax.scatter(eigenvalues.real, eigenvalues.imag, color=color, alpha=0.6,
                  s=15, label=cluster)

        if highlight_extremes:
            # Highlight max and min real eigenvalues
            idx_max = np.argmax(eigenvalues.real)
            idx_min = np.argmin(eigenvalues.real)

            ax.scatter(eigenvalues[idx_max].real, eigenvalues[idx_max].imag,
                      color='blue', edgecolor='black', s=100, zorder=3,
                      marker='*', label=f'{cluster} max Re(λ)')
            ax.scatter(eigenvalues[idx_min].real, eigenvalues[idx_min].imag,
                      color='red', edgecolor='black', s=100, zorder=3,
                      marker='*', label=f'{cluster} min Re(λ)')

    ax.set_xlabel('Re(λ)')
    ax.set_ylabel('Im(λ)')
    ax.set_title('Eigenvalue Spectrum')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    return ax


def plot_eigenvector_components(
    adata: AnnData,
    cluster: str,
    which: str = 'max',
    n_genes: int = 10,
    cluster_key: str = 'cell_type',
    color: Optional[str] = None,
    annotate: bool = True,
    figsize: tuple = (10, 5),
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot sorted eigenvector components with top gene annotations.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with computed eigenanalysis
    cluster : str
        Cluster name
    which : str, optional (default: 'max')
        Which eigenvalue: 'max' or 'min'
    n_genes : int, optional (default: 10)
        Number of top genes to annotate
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    color : str, optional
        Color for plot. If None, uses blue for 'max', red for 'min'
    annotate : bool, optional (default: True)
        Whether to annotate top genes
    figsize : tuple, optional (default: (10, 5))
        Figure size
    ax : plt.Axes, optional
        Axes to plot on

    Returns
    -------
    plt.Axes
        Axes with plot
    """
    if 'eigenanalysis' not in adata.uns['scHopfield']:
        raise ValueError(
            "Eigenanalysis not found. Please run sch.tl.compute_eigenanalysis() first."
        )

    eigenvalues = adata.uns['scHopfield']['eigenanalysis'][f'eigenvalues_{cluster}']
    eigenvectors = adata.uns['scHopfield']['eigenanalysis'][f'eigenvectors_{cluster}']
    gene_names = adata.uns['scHopfield']['eigenanalysis']['gene_names']

    # Select eigenvalue
    if which == 'max':
        idx = np.argmax(eigenvalues.real)
        default_color = 'blue'
        title_prefix = 'Max'
    elif which == 'min':
        idx = np.argmin(eigenvalues.real)
        default_color = 'red'
        title_prefix = 'Min'
    else:
        raise ValueError("which must be 'max' or 'min'")

    if color is None:
        color = default_color

    eigenvector = eigenvectors[:, idx]
    eigenvalue = eigenvalues[idx]

    # Sort by eigenvector value
    sorted_indices = np.argsort(eigenvector.real)
    sorted_eigenvector = eigenvector[sorted_indices]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot(sorted_eigenvector.real, '.', color=color)
    ax.set_ylabel('Component value')
    ax.set_xticks([])
    ax.set_title(f'{cluster} - {title_prefix} Eigenvalue Eigenvector (λ={eigenvalue.real:.3f})')

    if annotate:
        # Get top genes by absolute value
        sorted_abs = np.argsort(np.abs(eigenvector))[::-1]
        top_indices = sorted_abs[:n_genes]

        # Find positions in sorted array
        x_data = [np.where(sorted_indices == idx)[0][0] for idx in top_indices]
        y_data = eigenvector[top_indices].real
        names = gene_names[top_indices]

        _annotate_points(ax, x_data, y_data, names, offset_x_fraction=0.2, offset_y_fraction=0.1)

    return ax


def plot_eigenanalysis_grid(
    adata: AnnData,
    cluster_key: str = 'cell_type',
    order: Optional[List[str]] = None,
    colors: Optional[Dict[str, str]] = None,
    n_genes: int = 10,
    figsize: Optional[tuple] = None
) -> plt.Figure:
    """
    Plot comprehensive eigenanalysis grid for all clusters.

    Creates a grid with 3 columns per cluster:
    1. Eigenvalue spectrum with max/min highlighted
    2. Top eigenvector for max eigenvalue
    3. Top eigenvector for min eigenvalue

    Parameters
    ----------
    adata : AnnData
        Annotated data object with computed eigenanalysis
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    order : list, optional
        Order of clusters to display
    colors : dict, optional
        Colors for each cluster
    n_genes : int, optional (default: 10)
        Number of top genes to annotate
    figsize : tuple, optional
        Figure size. If None, auto-calculated

    Returns
    -------
    plt.Figure
        Figure with grid of plots
    """
    if 'eigenanalysis' not in adata.uns['scHopfield']:
        raise ValueError(
            "Eigenanalysis not found. Please run sch.tl.compute_eigenanalysis() first."
        )

    clusters = adata.obs[cluster_key].unique().tolist()
    if order is not None:
        clusters = [c for c in order if c in clusters]

    n_clusters = len(clusters)
    if figsize is None:
        figsize = (16, 4 * n_clusters)

    fig, axs = plt.subplots(n_clusters, 3, figsize=figsize)

    # Handle single cluster case
    if n_clusters == 1:
        axs = axs.reshape(1, -1)

    for i, cluster in enumerate(clusters):
        color = colors[cluster] if colors is not None and cluster in colors else 'tab:blue'

        # Column 1: Eigenvalue spectrum
        plot_eigenvalue_spectrum(
            adata,
            clusters=cluster,
            cluster_key=cluster_key,
            colors=colors,
            highlight_extremes=True,
            ax=axs[i, 0]
        )
        axs[i, 0].legend().remove()  # Remove legend for cleaner look

        # Column 2: Max eigenvalue eigenvector
        plot_eigenvector_components(
            adata,
            cluster=cluster,
            which='max',
            n_genes=n_genes,
            cluster_key=cluster_key,
            color='blue',
            ax=axs[i, 1]
        )

        # Column 3: Min eigenvalue eigenvector
        plot_eigenvector_components(
            adata,
            cluster=cluster,
            which='min',
            n_genes=n_genes,
            cluster_key=cluster_key,
            color='red',
            ax=axs[i, 2]
        )

    plt.tight_layout()
    return fig


def plot_grn_network(
    adata: AnnData,
    cluster: str,
    genes: Optional[List[str]] = None,
    cluster_key: str = 'cell_type',
    score_size: Optional[str] = None,
    size_threshold: float = 0,
    cmap: Union[str, 'Colormap'] = 'RdBu_r',
    topn: Optional[int] = None,
    w_quantile: float = 0.99,
    figsize: tuple = (10, 10),
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Generate a Gene Regulatory Network (GRN) graph for a cluster.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with interaction matrices
    cluster : str
        Cluster name
    genes : list, optional
        List of gene names to include. If None, uses all genes
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    score_size : str, optional
        Column in adata.var (with cluster suffix) to use for node sizes.
        Example: 'degree_centrality_out' will use 'degree_centrality_out_{cluster}'
    size_threshold : float, optional (default: 0)
        Threshold for displaying node labels (as fraction of max size)
    cmap : str or Colormap, optional (default: 'RdBu_r')
        Colormap for edge coloring
    topn : int, optional
        Number of top genes to retain based on size
    w_quantile : float, optional (default: 0.99)
        Quantile threshold for filtering weak edges
    figsize : tuple, optional (default: (10, 10))
        Figure size
    ax : plt.Axes, optional
        Axes to plot on

    Returns
    -------
    plt.Axes
        Axes with network plot
    """
    try:
        import networkx as nx
        from matplotlib.colors import Colormap
    except ImportError:
        raise ImportError(
            "NetworkX is required for this plot. "
            "Install it with: pip install networkx"
        )

    gene_list = get_genes_used(adata)
    gene_names_all = adata.var.index[gene_list]

    if genes is None:
        genes = gene_names_all.tolist()

    # Get interaction matrix
    W = adata.varp[f'W_{cluster}'].copy()

    # Threshold edges based on weight quantile
    threshold = np.quantile(np.abs(W), w_quantile)
    W[np.abs(W) < threshold] = 0

    # Create DataFrame representation
    df = pd.DataFrame(W.T, index=gene_names_all, columns=gene_names_all)

    # Compute node sizes
    if score_size is None:
        sizes = np.abs(W).sum(axis=0) + np.abs(W).sum(axis=1)
    else:
        score_col = f'{score_size}_{cluster}'
        if score_col not in adata.var.columns:
            raise ValueError(f"Column '{score_col}' not found in adata.var")
        sizes = np.array([
            adata.var.loc[g, score_col] if g in adata.var.index else 0
            for g in gene_names_all
        ])

    # Filter top genes based on size
    topq = np.sort(sizes)[-topn] if topn is not None else 0
    dropids = gene_names_all[sizes < topq]

    # Normalize sizes for better visualization
    size_multiplier = 1000 / max(sizes) if max(sizes) > 0 else 1
    gene_mask = np.isin(gene_names_all, genes) & (sizes >= topq)
    sizes_filtered = sizes[gene_mask]
    sizes_filtered = size_multiplier * sizes_filtered

    # Remove genes below threshold
    genes_filtered = [g for g in genes if g not in dropids]
    df.drop(index=dropids, columns=dropids, inplace=True)
    df = df.loc[genes_filtered, genes_filtered]

    # Define node labels (hide small ones)
    labels = {
        gene: gene if size / 1000 > size_threshold else ''
        for gene, size in zip(df.index, sizes_filtered)
    }

    # Create directed graph
    G = nx.from_pandas_adjacency(df, create_using=nx.DiGraph)
    Gp = nx.from_pandas_adjacency(df.abs(), create_using=nx.DiGraph)

    # Compute edge weights for visualization
    weights = np.array([abs(G[u][v]['weight']) for u, v in G.edges()])
    weights_signed = 10 * np.array([G[u][v]['weight'] for u, v in G.edges()])
    if weights.size > 0:
        weights = 1.5 * np.log1p(weights) / np.log1p(weights.max())

    # Define node positions
    pos = nx.circular_layout(G)

    # Define axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Validate colormap input
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    elif not isinstance(cmap, Colormap):
        from matplotlib.colors import Colormap as MplColormap
        if not isinstance(cmap, MplColormap):
            raise ValueError("`cmap` must be a string or a matplotlib.colors.Colormap instance")

    # Compute colormap normalization
    vmax = weights.max() if weights.size > 0 else 1

    # Draw network graph
    nx.draw_networkx(
        G, pos, node_size=sizes_filtered, width=weights, with_labels=True, labels=labels,
        edge_color=weights_signed, edge_cmap=cmap, edge_vmin=-vmax, edge_vmax=vmax, ax=ax
    )

    ax.set_title(f'{cluster} - GRN')
    ax.axis('off')

    return ax


def plot_grn_subset(
    adata: AnnData,
    cluster: str,
    selected_genes: List[str],
    cluster_key: str = 'cell_type',
    score_size: Optional[str] = None,
    node_positions: Optional[Dict[str, tuple]] = None,
    prune_threshold: float = 0,
    selected_edges: Optional[List[tuple]] = None,
    node_color: str = 'white',
    label_offset: float = 0.11,
    label_size: int = 12,
    variable_width: bool = False,
    figsize: tuple = (10, 10),
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot a Gene Regulatory Network (GRN) for a user-defined subset of genes.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with interaction matrices
    cluster : str
        Cluster name
    selected_genes : list
        List of genes to include in the graph
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    score_size : str, optional
        Column in adata.var (with cluster suffix) to use for node sizes
    node_positions : dict, optional
        Dictionary with custom node positions {gene: (x, y)}
    prune_threshold : float, optional (default: 0)
        Edges below this threshold (absolute value) will be removed
    selected_edges : list of tuples, optional
        List of user-defined edges (tuples) to plot
    node_color : str, optional (default: 'white')
        Color of nodes
    label_offset : float, optional (default: 0.11)
        Distance of labels from nodes
    label_size : int, optional (default: 12)
        Font size for labels
    variable_width : bool, optional (default: False)
        Whether to use variable edge widths based on weight
    figsize : tuple, optional (default: (10, 10))
        Figure size
    ax : plt.Axes, optional
        Axes to plot on

    Returns
    -------
    plt.Axes
        Axes with network plot
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "NetworkX is required for this plot. "
            "Install it with: pip install networkx"
        )

    gene_list = get_genes_used(adata)
    gene_names_all = adata.var.index[gene_list]

    # Get interaction matrix
    W = adata.varp[f'W_{cluster}']

    # Convert adjacency matrix to DataFrame
    df = pd.DataFrame(W.T, index=gene_names_all, columns=gene_names_all)

    # Subset the graph to only the selected nodes
    df = df.loc[selected_genes, selected_genes]

    # Prune weak edges
    df[df.abs() < prune_threshold] = 0

    # Filter only user-defined edges (if provided)
    if selected_edges:
        mask = np.zeros_like(df, dtype=bool)
        for u, v in selected_edges:
            if u in df.index and v in df.columns:
                mask[df.index.get_loc(u), df.columns.get_loc(v)] = True
        df[~mask] = 0

    # Compute node sizes
    if score_size is None:
        sizes = df.abs().sum(axis=0) + df.abs().sum(axis=1)
    else:
        score_col = f'{score_size}_{cluster}'
        if score_col not in adata.var.columns:
            raise ValueError(f"Column '{score_col}' not found in adata.var")
        sizes = pd.Series([
            adata.var.loc[g, score_col] if g in adata.var.index else 0
            for g in selected_genes
        ], index=selected_genes)

    # Normalize sizes
    size_multiplier = 1000 / sizes.max() if sizes.max() > 0 else 1
    sizes = size_multiplier * sizes
    node_size_dict = {node: size for node, size in sizes.items()}

    # Define node labels
    labels = {gene: gene for gene in selected_genes}

    # Create directed graph
    G = nx.from_pandas_adjacency(df, create_using=nx.DiGraph)

    # Compute edge weights
    edge_list = [(u, v) for u, v in G.edges() if abs(G[u][v]['weight']) >= prune_threshold]
    if not edge_list:
        print("Warning: No edges remain after filtering")
        edge_list = []

    weights = np.array([abs(G[u][v]['weight']) for u, v in edge_list]) if edge_list else np.array([])
    weights_signed = np.array([G[u][v]['weight'] for u, v in edge_list]) if edge_list else np.array([])

    # Normalize edge widths
    if weights.size > 0:
        weights = 2 * np.log1p(weights) / np.log1p(weights.max())

    # Use predefined node positions if provided, otherwise default to spring layout
    if node_positions:
        pos = {gene: node_positions[gene] for gene in selected_genes if gene in node_positions}
    else:
        pos = {}

    if len(pos) < len(selected_genes):
        default_layout = nx.spring_layout(G)
        for node in selected_genes:
            if node not in pos:
                pos[node] = default_layout.get(node, (0, 0))

    # Define axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Set fixed edge colors (Red for positive, Blue for negative)
    edge_colors = ['red' if w > 0 else 'blue' for w in weights_signed] if len(weights_signed) > 0 else []

    # Handle bidirectional edges: shift arcs slightly to avoid overlap
    curved_edges = set()
    for u, v in edge_list:
        if u == v:
            continue
        if (v, u) in edge_list and (v, u) not in curved_edges:
            curved_edges.add((u, v))
            curved_edges.add((v, u))

    # Adjust margins for each node
    min_margin = 0.02
    max_margin = 0.1
    node_margins = {node: np.clip(size / 2000, min_margin, max_margin) for node, size in node_size_dict.items()}

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=list(sizes.values), node_color=node_color,
                          edgecolors='black', linewidths=1.5, ax=ax, alpha=0.9)

    # Move labels outside the nodes
    adjusted_pos = {k: (v[0], v[1] + label_offset) for k, v in pos.items()}
    nx.draw_networkx_labels(G, adjusted_pos, labels, font_size=label_size, ax=ax)

    # Draw edges with adjusted arrows
    for idx, edge in enumerate(edge_list):
        u, v = edge
        width = weights[idx] if variable_width and len(weights) > 0 else 1
        style = "arc3,rad=0.15" if edge in curved_edges else "arc3,rad=0"
        nx.draw_networkx_edges(G, pos, edgelist=[edge], width=width,
                              edge_color=[edge_colors[idx]], ax=ax,
                              arrows=True,
                              min_source_margin=node_margins.get(u, min_margin),
                              min_target_margin=node_margins.get(v, min_margin),
                              connectionstyle=style)

    ax.set_title(f'{cluster} - Subset GRN')
    ax.axis('off')

    return ax
