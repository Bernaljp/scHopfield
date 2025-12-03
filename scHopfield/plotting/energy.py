"""Plotting functions for energy landscapes."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from anndata import AnnData

from .._utils.io import get_cluster_key


def plot_energy_landscape(
    adata: AnnData,
    cluster: str,
    basis: str = 'umap',
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot energy landscape on embedding space.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with energy embedding
    cluster : str
        Cluster name
    basis : str, optional
        Embedding basis
    ax : plt.Axes, optional
        Axes to plot on

    Returns
    -------
    plt.Axes
        Axes with plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    grid_X = adata.uns['scHopfield'][f'grid_X_{cluster}']
    grid_Y = adata.uns['scHopfield'][f'grid_Y_{cluster}']
    grid_energy = adata.uns['scHopfield'][f'grid_energy_{cluster}']

    im = ax.contourf(grid_X, grid_Y, grid_energy, levels=20, cmap='viridis', **kwargs)
    ax.set_xlabel(f'{basis.upper()} 1')
    ax.set_ylabel(f'{basis.upper()} 2')
    ax.set_title(f'Energy Landscape: {cluster}')
    plt.colorbar(im, ax=ax, label='Energy')

    return ax


def plot_energy_components(
    adata: AnnData,
    cluster: str,
    basis: str = 'umap'
) -> plt.Figure:
    """
    Plot all energy components (total, interaction, degradation, bias).

    Parameters
    ----------
    adata : AnnData
        Annotated data object
    cluster : str
        Cluster name
    basis : str, optional
        Embedding basis

    Returns
    -------
    plt.Figure
        Figure with subplots
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    energy_types = ['total', 'interaction', 'degradation', 'bias']
    for ax, etype in zip(axes.flat, energy_types):
        grid_X = adata.uns['scHopfield'][f'grid_X_{cluster}']
        grid_Y = adata.uns['scHopfield'][f'grid_Y_{cluster}']
        grid_energy = adata.uns['scHopfield'][f'grid_energy_{etype}_{cluster}']

        im = ax.contourf(grid_X, grid_Y, grid_energy, levels=20, cmap='viridis')
        ax.set_title(f'{etype.capitalize()} Energy')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    return fig
