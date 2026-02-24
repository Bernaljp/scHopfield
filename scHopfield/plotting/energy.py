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
    palette: Optional[str] = None,
    show_points: bool = False,
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
        Colors for each cluster. Overrides palette.
    palette : str, optional
        Seaborn palette name (e.g., 'Set2', 'husl', 'tab10')
    show_points : bool, optional (default: False)
        If True, overlay individual points as strip plot
    **fig_kws
        Additional keyword arguments for plt.subplots()

    Returns
    -------
    np.ndarray or plt.Axes
        Array of axes (if plot_energy='all') or single axes

    Examples
    --------
    >>> import scHopfield as sch
    >>> sch.pl.plot_energy_boxplots(adata, cluster_key='cell_type')
    >>> sch.pl.plot_energy_boxplots(adata, plot_energy='interaction', palette='Set2')
    """
    if order is None:
        order = adata.obs[cluster_key].unique().tolist()

    # Set up figure
    if plot_energy == 'all':
        fig_kws.setdefault('figsize', (14, 10))
        fig, axs = plt.subplots(2, 2, **fig_kws)
        titles = ['Total Energy', 'Interaction Energy', 'Degradation Energy', 'Bias Energy']
        for ax, title in zip(axs.flatten(), titles):
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        axs = axs.flatten()
        energy_cols = ['energy_total', 'energy_interaction', 'energy_degradation', 'energy_bias']
    else:
        fig_kws.setdefault('figsize', (10, 6))
        fig, axs = plt.subplots(1, 1, **fig_kws)
        axs = np.array([axs])
        energy_cols = [f'energy_{plot_energy.lower()}']
        axs[0].set_title(f'{plot_energy.capitalize()} Energy', fontsize=12, fontweight='bold', pad=10)

    # Handle colors
    plot_palette = None
    if colors is not None:
        if isinstance(colors, dict):
            plot_palette = [colors.get(k, 'gray') for k in order]
        elif isinstance(colors, list):
            assert len(colors) >= len(order), \
                "Colors list should have at least as many colors as clusters."
            plot_palette = colors[:len(order)]
    elif palette is not None:
        plot_palette = palette

    # Create boxplots
    for energy_col, ax in zip(energy_cols, axs):
        # Check if energy column exists
        if energy_col not in adata.obs.columns:
            ax.text(0.5, 0.5, f'{energy_col} not found\nRun sch.tl.compute_energies() first',
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            continue

        df = pd.DataFrame({
            'Cluster': adata.obs[cluster_key],
            'Energy': adata.obs[energy_col]
        })

        # Create boxplot with better styling
        bp = sns.boxplot(
            data=df, x='Cluster', y='Energy', order=order,
            ax=ax, palette=plot_palette,
            linewidth=1.5,
            fliersize=3,
            width=0.6
        )

        # Optionally add strip plot for individual points
        if show_points:
            sns.stripplot(
                data=df, x='Cluster', y='Energy', order=order,
                ax=ax, color='black', alpha=0.3, size=2,
                jitter=True
            )

        # Styling
        ax.set_xlabel('Cell Type', fontsize=10, fontweight='bold')
        ax.set_ylabel('Energy', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Rotate x-axis labels if many clusters
        if len(order) > 5:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    return fig if plot_energy == 'all' else axs[0]


def plot_energy_scatters(
    adata: AnnData,
    cluster_key: str = 'cell_type',
    basis: str = 'umap',
    order: Optional[List[str]] = None,
    plot_energy: str = 'all',
    show_legend: bool = True,
    colors: Optional[Union[List, Dict]] = None,
    palette: Optional[str] = None,
    alpha: float = 0.6,
    s: float = 20,
    elev: float = 30,
    azim: float = -60,
    **fig_kws
) -> Union[plt.Figure, plt.Axes]:
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
    show_legend : bool, optional (default: True)
        Whether to show legend
    colors : list or dict, optional
        Colors for each cluster. Overrides palette.
    palette : str, optional
        Seaborn or matplotlib colormap name (e.g., 'tab10', 'Set2')
    alpha : float, optional (default: 0.6)
        Transparency of points
    s : float, optional (default: 20)
        Size of points
    elev : float, optional (default: 30)
        Elevation viewing angle
    azim : float, optional (default: -60)
        Azimuthal viewing angle
    **fig_kws
        Additional keyword arguments for plt.subplots()

    Returns
    -------
    plt.Figure or plt.Axes
        Figure (if plot_energy='all') or single axes

    Examples
    --------
    >>> import scHopfield as sch
    >>> sch.pl.plot_energy_scatters(adata, cluster_key='cell_type')
    >>> sch.pl.plot_energy_scatters(adata, plot_energy='interaction', palette='tab10')
    """
    if order is None:
        order = adata.obs[cluster_key].unique().tolist()

    # Set up figure
    if plot_energy == 'all':
        fig_kws.setdefault('figsize', (16, 12))
        fig, axs = plt.subplots(2, 2, subplot_kw={'projection': '3d'}, **fig_kws)
        titles = ['Total Energy', 'Interaction Energy', 'Degradation Energy', 'Bias Energy']
        for ax, title in zip(axs.flatten(), titles):
            ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
        axs = axs.flatten()
        energy_cols = ['energy_total', 'energy_interaction', 'energy_degradation', 'energy_bias']
    else:
        fig_kws.setdefault('figsize', (10, 8))
        fig, axs = plt.subplots(1, 1, subplot_kw={'projection': '3d'}, **fig_kws)
        axs = np.array([axs])
        energy_cols = [f'energy_{plot_energy.lower()}']
        axs[0].set_title(f'{plot_energy.capitalize()} Energy', fontsize=12, fontweight='bold', pad=15)

    # Handle colors
    import matplotlib.cm as cm
    if colors is not None:
        if isinstance(colors, dict):
            color_map = colors
        elif isinstance(colors, list):
            assert len(colors) >= len(order), \
                "Colors list should have at least as many colors as clusters."
            color_map = {k: colors[i] for i, k in enumerate(order)}
    elif palette is not None:
        # Use colormap
        cmap = cm.get_cmap(palette)
        color_map = {k: cmap(i / len(order)) for i, k in enumerate(order)}
    else:
        # Default to tab10
        cmap = cm.get_cmap('tab10')
        color_map = {k: cmap(i % 10) for i, k in enumerate(order)}

    # Check if embedding exists
    embedding_key = f'X_{basis}'
    if embedding_key not in adata.obsm:
        raise ValueError(f"Embedding '{embedding_key}' not found in adata.obsm. "
                        f"Available: {list(adata.obsm.keys())}")

    # Plot each cluster
    for ax, energy_col in zip(axs, energy_cols):
        # Check if energy column exists
        if energy_col not in adata.obs.columns:
            ax.text2D(0.5, 0.5, f'{energy_col} not found\nRun sch.tl.compute_energies() first',
                     ha='center', va='center', transform=ax.transAxes, fontsize=10)
            continue

        for k in order:
            cluster_mask = (adata.obs[cluster_key] == k).values
            cells = adata.obsm[embedding_key][cluster_mask, :2]
            energies = adata.obs[energy_col].values[cluster_mask]

            # Plot with cluster-specific color
            ax.scatter(cells[:, 0], cells[:, 1], energies,
                      c=[color_map[k]], label=k,
                      alpha=alpha, s=s, edgecolors='none')

        # Styling
        ax.set_xlabel(f'{basis.upper()} 1', fontsize=10, labelpad=8)
        ax.set_ylabel(f'{basis.upper()} 2', fontsize=10, labelpad=8)
        ax.set_zlabel('Energy', fontsize=10, labelpad=8)
        ax.view_init(elev=elev, azim=azim)

        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')

        # Legend
        if show_legend:
            # Place legend outside plot
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1),
                     frameon=True, framealpha=0.9, fontsize=8)

    plt.tight_layout()
    return fig if plot_energy == 'all' else axs[0]
