"""Plotting functions for network visualization."""

import matplotlib.pyplot as plt
from typing import Optional
from anndata import AnnData


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
