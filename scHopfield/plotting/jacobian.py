"""Plotting functions for Jacobian analysis and visualization."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Union
from anndata import AnnData

from .._utils.io import get_genes_used


def plot_jacobian_eigenvalue_spectrum(
    adata: AnnData,
    cluster_key: str = 'cell_type',
    order: Optional[List[str]] = None,
    colors: Optional[Dict[str, str]] = None,
    figsize: tuple = (15, 15),
    sharex: bool = True,
    sharey: bool = True
) -> plt.Figure:
    """
    Plot Jacobian eigenvalues in complex plane for each cluster.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with computed Jacobian eigenvalues in obsm
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    order : list, optional
        Order of clusters to display
    colors : dict, optional
        Colors for each cluster
    figsize : tuple, optional (default: (15, 15))
        Figure size
    sharex : bool, optional (default: True)
        Share x-axis across subplots
    sharey : bool, optional (default: True)
        Share y-axis across subplots

    Returns
    -------
    plt.Figure
        Figure with eigenvalue spectra
    """
    if 'jacobian_eigenvalues' not in adata.obsm:
        raise ValueError(
            "Jacobian eigenvalues not found. "
            "Please run sch.tl.compute_jacobians() first."
        )

    eigenvalues = adata.obsm['jacobian_eigenvalues']

    clusters = adata.obs[cluster_key].unique().tolist()
    if order is not None:
        clusters = [c for c in order if c in clusters]

    n_clusters = len(clusters)
    ncols = 2
    nrows = int(np.ceil(n_clusters / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize,
                            sharex=sharex, sharey=sharey, tight_layout=True)
    axs = np.atleast_1d(axs).flatten()

    for i, cluster in enumerate(clusters):
        ax = axs[i]
        mask = (adata.obs[cluster_key] == cluster).values
        evals_cluster = eigenvalues[mask]

        color = colors[cluster] if colors is not None and cluster in colors else 'tab:blue'
        ax.scatter(evals_cluster.real.flatten(), evals_cluster.imag.flatten(),
                  color=color, s=2, alpha=0.5)

        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
        ax.set_title(cluster)
        ax.set_xlabel('Re(λ)')
        ax.set_ylabel('Im(λ)')

    # Hide extra subplots
    for i in range(n_clusters, len(axs)):
        axs[i].axis('off')

    return fig


def plot_jacobian_eigenvalue_boxplots(
    adata: AnnData,
    cluster_key: str = 'cell_type',
    order: Optional[List[str]] = None,
    colors: Optional[Dict[str, str]] = None,
    figsize: tuple = (15, 15)
) -> plt.Figure:
    """
    Plot boxplots of positive real and imaginary parts of Jacobian eigenvalues.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with computed Jacobian eigenvalues
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    order : list, optional
        Order of clusters to display
    colors : dict, optional
        Colors for each cluster
    figsize : tuple, optional (default: (15, 15))
        Figure size

    Returns
    -------
    plt.Figure
        Figure with boxplots
    """
    try:
        import seaborn as sns
    except ImportError:
        raise ImportError(
            "Seaborn is required for this plot. "
            "Install it with: pip install seaborn"
        )

    if 'jacobian_eigenvalues' not in adata.obsm:
        raise ValueError(
            "Jacobian eigenvalues not found. "
            "Please run sch.tl.compute_jacobians() first."
        )

    eigenvalues = adata.obsm['jacobian_eigenvalues']
    genes = get_genes_used(adata)

    # Prepare data for plotting
    df_real = pd.DataFrame(eigenvalues[:, genes].real, index=adata.obs_names)
    df_real['cluster'] = adata.obs[cluster_key]
    df_real = df_real.melt(id_vars='cluster', value_name='eigenvalue').drop(columns='variable')
    df_real = df_real[df_real['eigenvalue'] > 0]

    df_imag = pd.DataFrame(eigenvalues[:, genes].imag, index=adata.obs_names)
    df_imag['cluster'] = adata.obs[cluster_key]
    df_imag = df_imag.melt(id_vars='cluster', value_name='eigenvalue').drop(columns='variable')
    df_imag = df_imag[df_imag['eigenvalue'] > 0]

    # Create figure
    fig, axs = plt.subplots(2, 1, figsize=figsize, sharey=False, tight_layout=True)

    # Real part
    if colors:
        palette_real = [colors.get(c, 'gray') for c in (order if order else df_real['cluster'].unique())]
    else:
        palette_real = None

    sns.boxplot(
        data=df_real, x='cluster', y='eigenvalue',
        showfliers=False, ax=axs[0],
        order=order, palette=palette_real
    )
    axs[0].set_title("Positive Real Part of Eigenvalues", fontsize=14)
    axs[0].set_xlabel("Cluster", fontsize=12)
    axs[0].set_ylabel("Eigenvalue (Real)", fontsize=12)
    axs[0].tick_params(axis='x', rotation=45)
    axs[0].grid(axis='y', linestyle='--', alpha=0.4)

    # Imaginary part
    if colors:
        palette_imag = [colors.get(c, 'gray') for c in (order if order else df_imag['cluster'].unique())]
    else:
        palette_imag = None

    sns.boxplot(
        data=df_imag, x='cluster', y='eigenvalue',
        showfliers=False, ax=axs[1],
        order=order, palette=palette_imag
    )
    axs[1].set_title("Positive Imaginary Part of Eigenvalues", fontsize=14)
    axs[1].set_xlabel("Cluster", fontsize=12)
    axs[1].set_ylabel("Eigenvalue (Imaginary)", fontsize=12)
    axs[1].tick_params(axis='x', rotation=45)
    axs[1].grid(axis='y', linestyle='--', alpha=0.4)

    return fig


def plot_jacobian_stats_boxplots(
    adata: AnnData,
    cluster_key: str = 'cell_type',
    order: Optional[List[str]] = None,
    colors: Optional[Dict[str, str]] = None,
    figsize: tuple = (15, 15)
) -> plt.Figure:
    """
    Plot boxplots of Jacobian summary statistics.

    Plots distributions of:
    - Number of positive eigenvalues
    - Jacobian trace
    - Rotational part magnitude (if available)

    Parameters
    ----------
    adata : AnnData
        Annotated data object with computed Jacobian stats
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    order : list, optional
        Order of clusters to display
    colors : dict, optional
        Colors for each cluster
    figsize : tuple, optional (default: (15, 15))
        Figure size

    Returns
    -------
    plt.Figure
        Figure with boxplots
    """
    try:
        import seaborn as sns
    except ImportError:
        raise ImportError(
            "Seaborn is required for this plot. "
            "Install it with: pip install seaborn"
        )

    # Check which stats are available
    has_positive = 'jacobian_positive_evals' in adata.obs
    has_trace = 'jacobian_trace' in adata.obs
    has_rotational = 'jacobian_rotational' in adata.obs

    if not any([has_positive, has_trace, has_rotational]):
        raise ValueError(
            "No Jacobian stats found. "
            "Please run sch.tl.compute_jacobian_stats() first."
        )

    n_plots = sum([has_positive, has_trace, has_rotational])
    fig, axs = plt.subplots(n_plots, 1, figsize=figsize, tight_layout=True)
    if n_plots == 1:
        axs = [axs]

    plot_idx = 0

    # Color palette
    if colors and order:
        palette = [colors.get(c, 'gray') for c in order]
    else:
        palette = None

    # Plot positive eigenvalues
    if has_positive:
        data = adata.obs[[cluster_key, 'jacobian_positive_evals']].copy()
        sns.boxplot(
            data=data, x=cluster_key, y='jacobian_positive_evals',
            showfliers=True, order=order, palette=palette, ax=axs[plot_idx]
        )
        axs[plot_idx].set_ylabel("Number of Positive Real Eigenvalues")
        axs[plot_idx].set_title("Distribution of Positive Eigenvalue Counts per Cell Type")
        axs[plot_idx].tick_params(axis='x', rotation=45)
        plot_idx += 1

    # Plot trace
    if has_trace:
        data = adata.obs[[cluster_key, 'jacobian_trace']].copy()
        sns.boxplot(
            data=data, x=cluster_key, y='jacobian_trace',
            showfliers=False, order=order, palette=palette, ax=axs[plot_idx]
        )
        axs[plot_idx].set_ylabel("Trace of the Jacobian")
        axs[plot_idx].set_title("Distribution of Jacobian Trace per Cell Type")
        axs[plot_idx].tick_params(axis='x', rotation=45)
        plot_idx += 1

    # Plot rotational part
    if has_rotational:
        data = adata.obs[[cluster_key, 'jacobian_rotational']].copy()
        sns.boxplot(
            data=data, x=cluster_key, y='jacobian_rotational',
            showfliers=False, order=order, palette=palette, ax=axs[plot_idx]
        )
        axs[plot_idx].set_ylabel("Rotational Part of the Jacobian")
        axs[plot_idx].set_title("Distribution of Jacobian Rotational Part per Cell Type")
        axs[plot_idx].tick_params(axis='x', rotation=45)
        plot_idx += 1

    return fig


def plot_jacobian_element_grid(
    adata: AnnData,
    gene_pairs: List[tuple],
    ncols: int = 2,
    figsize: Optional[tuple] = None,
    **scatter_kwargs
) -> plt.Figure:
    """
    Plot grid of Jacobian elements (partial derivatives) on UMAP.

    This function requires dynamo for UMAP plotting.
    Each subplot shows df_i/dx_j for a gene pair (i, j).

    Parameters
    ----------
    adata : AnnData
        Annotated data object with computed Jacobian elements
    gene_pairs : list of tuples
        List of (gene_i, gene_j) pairs to plot
    ncols : int, optional (default: 2)
        Number of columns in grid
    figsize : tuple, optional
        Figure size. If None, auto-calculated
    **scatter_kwargs
        Additional arguments passed to plotting function

    Returns
    -------
    plt.Figure
        Figure with grid of plots
    """
    n_pairs = len(gene_pairs)
    nrows = int(np.ceil(n_pairs / ncols))

    if figsize is None:
        figsize = (5 * ncols, 5 * nrows)

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, tight_layout=True)
    axs = np.atleast_1d(axs).flatten()

    for idx, (gene_i, gene_j) in enumerate(gene_pairs):
        col_name = f'jacobian_df_{gene_i}_dx_{gene_j}'

        if col_name not in adata.obs:
            axs[idx].text(0.5, 0.5, f'{col_name}\nnot found',
                         ha='center', va='center', transform=axs[idx].transAxes)
            axs[idx].axis('off')
            continue

        # Simple scatter plot - users can customize with dynamo if needed
        from matplotlib.colors import TwoSlopeNorm
        values = adata.obs[col_name].values
        vmax = np.abs(values).max()

        # Plot on UMAP if available
        if 'X_umap' in adata.obsm:
            umap_coords = adata.obsm['X_umap']
            scatter = axs[idx].scatter(
                umap_coords[:, 0],
                umap_coords[:, 1],
                c=values,
                cmap='coolwarm',
                norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
                s=scatter_kwargs.get('s', 5),
                alpha=scatter_kwargs.get('alpha', 0.7)
            )
            plt.colorbar(scatter, ax=axs[idx])
            axs[idx].set_title(f'∂{gene_i}/∂{gene_j}')
            axs[idx].set_xlabel('UMAP 1')
            axs[idx].set_ylabel('UMAP 2')
        else:
            axs[idx].text(0.5, 0.5, 'UMAP not available',
                         ha='center', va='center', transform=axs[idx].transAxes)
            axs[idx].axis('off')

    # Hide extra subplots
    for i in range(n_pairs, len(axs)):
        axs[i].axis('off')

    return fig
