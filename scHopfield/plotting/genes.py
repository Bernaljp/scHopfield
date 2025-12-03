"""Plotting functions for gene-level analysis."""

import matplotlib.pyplot as plt
from typing import Optional
from anndata import AnnData


def plot_sigmoid_fit(
    adata: AnnData,
    gene: str,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot sigmoid fit for a gene.

    Parameters
    ----------
    adata : AnnData
        Annotated data object
    gene : str
        Gene name
    ax : plt.Axes, optional
        Axes to plot on

    Returns
    -------
    plt.Axes
        Axes with plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    # Placeholder implementation
    ax.set_title(f'Sigmoid Fit: {gene}')
    ax.set_xlabel('Expression')
    ax.set_ylabel('Sigmoid activation')

    return ax
