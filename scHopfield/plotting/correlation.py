"""Plotting functions for energy-gene correlations."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Dict
from anndata import AnnData

from .._utils.io import get_genes_used


def plot_gene_correlation_scatter(
    adata: AnnData,
    clus1: str,
    clus2: str,
    energy: str = 'total',
    cluster_key: str = 'cell_type',
    ax: Optional[plt.Axes] = None,
    annotate: Optional[int] = None,
    clus1_low: float = -0.5,
    clus1_high: float = 0.5,
    clus2_low: float = -0.5,
    clus2_high: float = 0.5
) -> plt.Axes:
    """
    Plot scatter of gene correlations between two clusters.

    Creates a scatter plot comparing the gene correlations with energy
    landscapes between two clusters, highlighting genes with divergent
    behavior (strongly positive in one cluster, strongly negative in the other).

    Parameters
    ----------
    adata : AnnData
        Annotated data object with computed correlations
    clus1 : str
        First cluster name (x-axis)
    clus2 : str
        Second cluster name (y-axis)
    energy : str, optional (default: 'total')
        Energy type: 'total', 'interaction', 'degradation', or 'bias'
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    ax : plt.Axes, optional
        Axes to plot on. If None, creates new figure
    annotate : int, optional
        If provided, annotates top N divergent genes
    clus1_low : float, optional (default: -0.5)
        Lower threshold for clus1 to identify divergent genes
    clus1_high : float, optional (default: 0.5)
        Upper threshold for clus1 to identify divergent genes
    clus2_low : float, optional (default: -0.5)
        Lower threshold for clus2 to identify divergent genes
    clus2_high : float, optional (default: 0.5)
        Upper threshold for clus2 to identify divergent genes

    Returns
    -------
    plt.Axes
        Axes with plot
    """
    genes = get_genes_used(adata)
    gene_names = adata.var.index[genes]

    # Get correlations
    corr_col1 = f'correlation_{energy}_{clus1}'
    corr_col2 = f'correlation_{energy}_{clus2}'

    if corr_col1 not in adata.var.columns or corr_col2 not in adata.var.columns:
        raise ValueError(
            f"Correlation data not found. Please run sch.tl.energy_gene_correlation() first."
        )

    corr1 = adata.var[corr_col1].values[genes]
    corr2 = adata.var[corr_col2].values[genes]

    # Create a new figure and axes if none are provided
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), tight_layout=True)

    # Set the limits for the axes
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))

    # Identify correlations that are in opposite corners of the plot
    positions_corners = np.logical_or(
        np.logical_and(corr1 >= clus1_high, corr2 <= clus2_low),
        np.logical_and(corr1 <= clus1_low, corr2 >= clus2_high)
    )

    corr_corners = np.where(positions_corners)[0]
    corr_center = np.where(~positions_corners)[0]

    # Plot the correlations using different colors for clarity
    ax.scatter(corr1[corr_corners], corr2[corr_corners], c='k', s=0.6, label='Divergent Correlations')
    ax.scatter(corr1[corr_center], corr2[corr_center], c='lightgray', s=0.5, label='Other Correlations')

    # Annotate top N divergent genes if requested
    if annotate is not None:
        nn = annotate
        # Get top N genes with the highest absolute correlation values
        cor_indices = np.argsort((corr1[corr_corners])**2 + (corr2[corr_corners])**2)[-nn:]
        # Get the names of the top N genes
        gois = gene_names[corr_corners][cor_indices]

        # Adding labels for the top N genes
        arrow_dict = {"width": 0.5, "headwidth": 0.5, "headlength": 1, "color": "gray"}
        for gg, xx, yy in zip(gois, corr1[corr_corners][cor_indices], corr2[corr_corners][cor_indices]):
            rand_shift_1 = np.random.uniform(-0.08, 0.08)
            rand_shift_2 = np.random.uniform(-0.08, 0.08)
            ax.annotate(gg, xy=(xx, yy), xytext=(xx+rand_shift_1, yy+rand_shift_2), arrowprops=arrow_dict)

    # Add reference lines
    ax.vlines([clus1_low, clus1_high], ymin=-1, ymax=1, linestyles='dashed', color='r')
    ax.hlines([clus2_low, clus2_high], xmin=-1, xmax=1, linestyles='dashed', color='r')
    ax.set_xlabel(clus1)
    ax.set_ylabel(clus2)

    return ax


def plot_correlations_grid(
    adata: AnnData,
    cluster_key: str = 'cell_type',
    energy: str = 'total',
    order: Optional[List[str]] = None,
    colors: Optional[Union[List, Dict]] = None,
    x_low: float = -0.5,
    x_high: float = 0.5,
    y_low: float = -0.5,
    y_high: float = 0.5,
    **kwargs
) -> plt.Figure:
    """
    Plot grid of correlation scatter plots between all pairs of clusters.

    Creates a matrix where the diagonal shows cluster names and the
    off-diagonal plots show gene correlation scatter plots between clusters.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with computed correlations
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    energy : str, optional (default: 'total')
        Energy type: 'total', 'interaction', 'degradation', or 'bias'
    order : list, optional
        Order of clusters to display. If None, uses all unique clusters
    colors : list or dict, optional
        Colors for each cluster. If dict, maps cluster names to colors.
        If list, colors in order matching `order` parameter.
        Colors should be RGBA tuples or RGB tuples.
    x_low : float, optional (default: -0.5)
        Lower x threshold for highlighting divergent genes
    x_high : float, optional (default: 0.5)
        Upper x threshold for highlighting divergent genes
    y_low : float, optional (default: -0.5)
        Lower y threshold for highlighting divergent genes
    y_high : float, optional (default: 0.5)
        Upper y threshold for highlighting divergent genes
    **kwargs
        Additional arguments:
        - figsize : tuple (default: (15, 15))
        - tight_layout : bool (default: True)

    Returns
    -------
    plt.Figure
        Figure with correlation grid
    """
    if order is None:
        cell_types = adata.obs[cluster_key].unique().tolist()
    else:
        cell_types = order

    n = len(cell_types)
    figsize = kwargs.get('figsize', (15, 15))
    tight_layout = kwargs.get('tight_layout', True)

    # Convert colors to dict if it's a list
    if colors is not None and not isinstance(colors, dict):
        colors = {cell_types[i]: colors[i] for i in range(len(cell_types))}

    fig, axs = plt.subplots(n, n, figsize=figsize, tight_layout=tight_layout)

    # Handle case where n=1 (axs is not an array)
    if n == 1:
        axs = np.array([[axs]])
    elif n > 1 and axs.ndim == 1:
        axs = axs.reshape(n, n)

    for i in range(n):
        for j in range(i, n):
            if i == j:
                # Diagonal: show cluster name
                for spine in axs[i, j].spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(2)
                    if colors is not None:
                        spine.set_color(colors[cell_types[i]])

                # Remove ticks
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])

                # Add text in the middle
                text = cell_types[i]
                text = text.replace(' ', '\n', 1)
                text = text.replace('-', '-\n')
                axs[i, j].text(
                    0.5, 0.5, text,
                    ha='center', va='center',
                    fontsize=18, fontweight='bold',
                    fontname='serif',
                    transform=axs[i, j].transAxes
                )

                # Set background color
                if colors is not None:
                    c = list(colors[cell_types[i]])
                    if len(c) == 4:  # RGBA
                        c[-1] = 0.2  # Set alpha
                    elif len(c) == 3:  # RGB, add alpha
                        c = list(c) + [0.2]
                    axs[i, j].set_facecolor(c)
            else:
                # Upper triangle: turn off
                axs[i, j].axis('off')

                # Lower triangle: plot correlation scatter
                plot_gene_correlation_scatter(
                    adata,
                    clus1=cell_types[i],
                    clus2=cell_types[j],
                    energy=energy,
                    cluster_key=cluster_key,
                    ax=axs[j, i],
                    clus1_low=x_low,
                    clus1_high=x_high,
                    clus2_low=y_low,
                    clus2_high=y_high
                )

                # Clean up axes
                axs[j, i].set_xticks([])
                axs[j, i].set_yticks([])
                axs[j, i].set_xlabel('')
                axs[j, i].set_ylabel('')

                # Add ticks for edges
                if i == 0:  # First column
                    axs[j, i].set_yticks([-1, -0.5, 0, 0.5, 1])
                if j == n - 1:  # Last row
                    axs[j, i].set_xticks([-1, -0.5, 0, 0.5, 1])

    return fig
