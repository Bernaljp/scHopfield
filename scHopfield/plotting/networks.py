"""Plotting functions for network visualization."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from typing import Optional, List, Dict, Union
from anndata import AnnData

from .._utils.io import get_genes_used, get_cluster_genes


def plot_interaction_matrix(
    adata: AnnData,
    cluster: str,
    top_n: Optional[int] = None,
    sort_by: str = 'degree',
    ax: Optional[plt.Axes] = None,
    cmap: str = 'RdBu_r',
    show_labels: bool = True,
    label_fontsize: int = 8,
    **kwargs
) -> plt.Axes:
    """
    Plot interaction matrix heatmap.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with fitted interactions
    cluster : str
        Cluster name
    top_n : int, optional
        Number of top genes to show. If None, shows all genes.
    sort_by : str, optional (default: 'degree')
        How to select top genes: 'degree' (sum of absolute weights),
        'variance' (variance of weights), or 'none' (original order)
    ax : plt.Axes, optional
        Axes to plot on. If None, creates new figure.
    cmap : str, optional (default: 'RdBu_r')
        Colormap for the heatmap
    show_labels : bool, optional (default: True)
        Whether to show gene name labels on axes
    label_fontsize : int, optional (default: 8)
        Font size for gene labels
    **kwargs
        Additional keyword arguments for imshow (e.g., vmin, vmax)

    Returns
    -------
    plt.Axes
        Axes with plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Get interaction matrix
    W = adata.varp[f'W_{cluster}']

    # Get gene names
    genes = get_genes_used(adata)
    gene_names = adata.var_names[genes].values

    # Select top genes if requested
    if top_n is not None and top_n < W.shape[0]:
        if sort_by == 'degree':
            # Sort by sum of absolute weights (total connectivity)
            degree = np.abs(W).sum(axis=0) + np.abs(W).sum(axis=1)
            top_idx = np.argsort(degree)[-top_n:]
        elif sort_by == 'variance':
            # Sort by variance of weights
            variance = np.var(W, axis=0) + np.var(W, axis=1)
            top_idx = np.argsort(variance)[-top_n:]
        else:
            # Keep original order, just take first N
            top_idx = np.arange(top_n)

        W = W[np.ix_(top_idx, top_idx)]
        gene_names = gene_names[top_idx]

    # Plot heatmap
    vmax = kwargs.pop('vmax', np.abs(W).max())
    vmin = kwargs.pop('vmin', -vmax)
    im = ax.imshow(W, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)

    # Add labels
    if show_labels and len(gene_names) <= 50:
        ax.set_xticks(np.arange(len(gene_names)))
        ax.set_yticks(np.arange(len(gene_names)))
        ax.set_xticklabels(gene_names, rotation=90, fontsize=label_fontsize)
        ax.set_yticklabels(gene_names, fontsize=label_fontsize)
    else:
        ax.set_xlabel('Genes', fontsize=10)
        ax.set_ylabel('Genes', fontsize=10)

    ax.set_title(f'Interaction Matrix: {cluster}', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Interaction strength')

    return ax


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
    genes, gene_names, clusters = get_cluster_genes(adata, cluster_key, order)

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
    """
    Generate evenly spaced values as an iterator.

    Helper function for creating evenly spaced annotation positions.
    Similar to numpy.linspace but returns an iterator instead of an array.

    Parameters
    ----------
    start : float
        Starting value
    stop : float
        Ending value
    num : int
        Number of values to generate

    Yields
    ------
    float
        Evenly spaced values from start to stop (inclusive)
    """
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
    cmap: str = 'RdBu_r',
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
    # Gp = nx.from_pandas_adjacency(df.abs(), create_using=nx.DiGraph)

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


