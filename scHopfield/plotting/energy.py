"""Plotting functions for energy landscapes."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union, List, Dict
from anndata import AnnData


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
    Plot all energy components (total, interaction, degradation, bias) for a cluster.

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


def plot_energy_boxplots(
    adata: AnnData,
    cluster_key: str = 'cell_type',
    order: Optional[List[str]] = None,
    plot_energy: str = 'all',
    colors: Optional[Union[List, Dict]] = None,
    **fig_kws
) -> Union[np.ndarray, plt.Axes]:
    """
    Plot energy distributions for different clusters using boxplots.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with computed energies
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    order : list, optional
        Order of clusters to display in the boxplots
    plot_energy : str, optional (default: 'all')
        Which energy to plot: 'all', 'total', 'interaction', 'degradation', or 'bias'
    colors : list or dict, optional
        Colors for each cluster
    **fig_kws
        Additional keyword arguments for plt.subplots()

    Returns
    -------
    np.ndarray or plt.Axes
        Array of axes (if plot_energy='all') or single axes
    """
    if order is None:
        order = adata.obs[cluster_key].unique().tolist()

    if plot_energy == 'all':
        fig, axs = plt.subplots(2, 2, **fig_kws)
        axs[0, 0].set_title('Total Energy')
        axs[0, 1].set_title('Interaction Energy')
        axs[1, 0].set_title('Degradation Energy')
        axs[1, 1].set_title('Bias Energy')
        axs = axs.flatten()

        energy_cols = ['energy_total', 'energy_interaction', 'energy_degradation', 'energy_bias']
    else:
        fig, axs = plt.subplots(1, 1, **fig_kws)
        axs = np.array([axs])
        energy_cols = [f'energy_{plot_energy.lower()}']

    # Handle colors
    if colors is not None:
        if isinstance(colors, dict):
            color_map = colors
        else:
            assert isinstance(colors, list) and len(colors) >= len(order), \
                "Colors should be a list of length at least equal to the number of clusters."
            color_map = {k: colors[i] for i, k in enumerate(order)}
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[color_map[k] for k in order])

    # Create boxplots
    for energy_col, ax in zip(energy_cols, axs):
        df = pd.DataFrame({
            'Cluster': adata.obs[cluster_key],
            'Energy': adata.obs[energy_col]
        })
        sns.boxplot(data=df, x='Cluster', y='Energy', order=order, ax=ax)

    plt.tight_layout()
    return axs


def plot_energy_scatters(
    adata: AnnData,
    cluster_key: str = 'cell_type',
    basis: str = 'umap',
    order: Optional[List[str]] = None,
    plot_energy: str = 'all',
    show_legend: bool = False,
    **fig_kws
) -> Union[np.ndarray, plt.Axes]:
    """
    Plot energy landscapes for different clusters using 3D scatter plots.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with computed energies
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    basis : str, optional (default: 'umap')
        The basis used for embedding
    order : list, optional
        Order of clusters to display
    plot_energy : str, optional (default: 'all')
        Which energy to plot: 'all', 'total', 'interaction', 'degradation', or 'bias'
    show_legend : bool, optional (default: False)
        Whether to show legend
    **fig_kws
        Additional keyword arguments for plt.subplots()

    Returns
    -------
    np.ndarray or plt.Axes
        Array of axes (if plot_energy='all') or single axes
    """
    if order is None:
        order = adata.obs[cluster_key].unique().tolist()

    if plot_energy == 'all':
        fig, axs = plt.subplots(2, 2, subplot_kw={'projection': '3d'}, **fig_kws)
        axs[0, 0].set_title('Total Energy')
        axs[0, 1].set_title('Interaction Energy')
        axs[1, 0].set_title('Degradation Energy')
        axs[1, 1].set_title('Bias Energy')

        axs = axs.flatten()
        energy_cols = ['energy_total', 'energy_interaction', 'energy_degradation', 'energy_bias']
    else:
        fig, axs = plt.subplots(1, 1, subplot_kw={'projection': '3d'}, **fig_kws)
        axs = np.array([axs])
        energy_cols = [f'energy_{plot_energy.lower()}']

    # Plot each cluster
    for k in order:
        cluster_mask = adata.obs[cluster_key] == k
        cells = adata.obsm[f'X_{basis}'][cluster_mask, :2]

        for ax, energy_col in zip(axs, energy_cols):
            energies = adata.obs[energy_col].values[cluster_mask]
            ax.scatter(*cells.T, energies, label=k)

    if show_legend:
        for ax in axs:
            ax.legend()

    plt.tight_layout()
    return axs
