"""Plotting functions for gene-level analysis."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from anndata import AnnData
from .._utils.math import sigmoid
from .._utils.io import get_matrix, to_numpy


def plot_sigmoid_fit(
    adata: AnnData,
    gene: str,
    spliced_key: str = 'Ms',
    color_clusters: bool = False,
    cluster_key: str = 'cell_type',
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot sigmoid fit for a gene showing expression CDF and fitted curve.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with fitted sigmoid parameters
    gene : str
        Gene name to plot
    spliced_key : str, optional (default: 'Ms')
        Layer key for spliced expression
    color_clusters : bool, optional (default: False)
        If True, color points by cluster
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels (used if color_clusters=True)
    ax : plt.Axes, optional
        Axes to plot on. If None, creates new figure.
    **kwargs
        Additional keyword arguments:
        - c1: color for expression data (default: 'gray')
        - c2: color for fitted curve (default: 'red')
        - alpha: transparency for scatter points (default: 0.5)
        - s: size for scatter points (default: 10)

    Returns
    -------
    plt.Axes
        Axes with plot

    Examples
    --------
    >>> import scHopfield as sch
    >>> sch.pl.plot_sigmoid_fit(adata, 'Gata1')
    >>> sch.pl.plot_sigmoid_fit(adata, 'Gata1', color_clusters=True)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    # Check if sigmoid parameters exist
    if 'sigmoid_threshold' not in adata.var.columns:
        raise ValueError("Sigmoid parameters not found. Run sch.pp.fit_all_sigmoids() first.")

    # Get gene index
    if gene not in adata.var_names:
        raise ValueError(f"Gene '{gene}' not found in adata.var_names")

    gene_idx = adata.var_names.get_loc(gene)

    # Check if this gene was used in fitting
    if 'scHopfield_used' in adata.var.columns and not adata.var['scHopfield_used'].iloc[gene_idx]:
        ax.text(0.5, 0.5, f'{gene}\nNot included in analysis',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel('Expression')
        ax.set_ylabel('CDF')
        return ax

    # Get expression data for this gene
    gexp = to_numpy(get_matrix(adata, spliced_key, genes=[gene_idx])).flatten()

    # Sort expression and create empirical CDF
    sorted_expr = np.sort(gexp)
    empirical_cdf = np.linspace(0, 1, len(sorted_expr))

    # Get sigmoid parameters
    threshold = adata.var['sigmoid_threshold'].iloc[gene_idx]
    exponent = adata.var['sigmoid_exponent'].iloc[gene_idx]
    offset = adata.var['sigmoid_offset'].iloc[gene_idx]
    mse = adata.var['sigmoid_mse'].iloc[gene_idx]

    # Plot expression vs CDF
    c1 = kwargs.get('c1', 'gray')
    c2 = kwargs.get('c2', 'red')
    alpha = kwargs.get('alpha', 0.5)
    size = kwargs.get('s', 10)

    if color_clusters and cluster_key in adata.obs.columns:
        # Color by cluster
        for cluster in adata.obs[cluster_key].unique():
            cluster_mask = (adata.obs[cluster_key] == cluster).values
            cluster_expr = np.sort(gexp[cluster_mask])
            cluster_cdf = np.linspace(0, 1, len(cluster_expr))
            ax.scatter(cluster_expr, cluster_cdf, s=size, alpha=alpha,
                      label=f'{cluster}', rasterized=True)
    else:
        # Single color
        ax.scatter(sorted_expr, empirical_cdf, s=size, alpha=alpha,
                  color=c1, label='Expression', rasterized=True)

    # Compute fitted sigmoid curve
    # The formula includes offset: sigmoid(x) * (1 - offset) + offset
    fitted_curve = sigmoid(sorted_expr, threshold, exponent) * (1 - offset) + offset

    # Plot fitted curve
    ax.plot(sorted_expr, fitted_curve, '-', linewidth=2.5,
           color=c2, label='Sigmoid fit', zorder=10)

    # Add sigmoid formula as text
    sigmoid_formula = r"$\frac{{x^{{{:.2f}}}}}{{x^{{{:.2f}}} + {:.2f}^{{{:.2f}}}}}$".format(
        exponent, exponent, threshold, exponent
    )

    # Position text in upper left
    textstr = f'{sigmoid_formula}\nMSE = {mse:.4f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Styling
    ax.set_xlabel('Expression', fontsize=11)
    ax.set_ylabel('Cumulative Distribution', fontsize=11)
    ax.set_title(f'{gene}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xlim(left=0)
    ax.set_ylim([0, 1])

    return ax
