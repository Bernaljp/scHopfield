"""
Flow visualization functions for perturbation analysis.

This module contains pure visualization functions for flow analysis.
Computation functions are in:
- scHopfield.tools.flow: calculate_flow, calculate_grid_flow, calculate_inner_product
- scHopfield.tools.velocity: compute_velocity, compute_velocity_delta
- scHopfield.tools.embedding: project_to_embedding
- scHopfield.dynamics.simulation: calculate_trajectory_flow

References
----------
Logic for the transition vector field is inspired by the perturbation
simulation workflow in CellOracle:
Kamimoto et al. (2023). Nature. https://doi.org/10.1038/s41586-022-05688-9
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
import seaborn as sns
from typing import Optional, List, Dict, Tuple
from anndata import AnnData

from ..tools.flow import calculate_flow, calculate_grid_flow, calculate_inner_product
from .._utils.io import get_genes_used


# =============================================================================
# Main Plotting Functions
# =============================================================================

def plot_flow(
    adata: AnnData,
    flow_key: Optional[str] = None,
    basis: str = 'umap',
    ax: Optional[plt.Axes] = None,
    on_grid: bool = False,
    scale: float = 1.0,
    color: str = 'black',
    alpha: float = 0.8,
    show_background: bool = True,
    cluster_key: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    s: float = 10,
    figsize: Tuple[float, float] = (8, 8),
    title: Optional[str] = None,
    # Grid options
    n_grid: int = 40,
    n_neighbors: int = 200,
    min_mass: float = 1.0,
    recalculate: bool = False,
    n_jobs: int = 4,
    **quiver_kwargs
) -> plt.Axes:
    """
    Unified flow plotting function.

    Can plot flow vectors directly on cells or interpolated onto a grid.

    Parameters
    ----------
    adata : AnnData
        Annotated data with flow vectors
    flow_key : str, optional
        Key in adata.obsm for flow vectors.
        If None, uses 'perturbation_flow_{basis}'.
    basis : str, optional (default: 'umap')
        Embedding basis
    ax : plt.Axes, optional
        Axes to plot on. If None, creates new figure.
    on_grid : bool, optional (default: False)
        If True, interpolate flow to grid before plotting.
    scale : float, optional (default: 1.0)
        Scale factor for arrows
    color : str, optional (default: 'black')
        Arrow color
    alpha : float, optional (default: 0.8)
        Arrow transparency
    show_background : bool, optional (default: True)
        Show background scatter of cells
    cluster_key : str, optional
        Key for cluster labels (for coloring background)
    colors : dict, optional
        Dictionary mapping cluster names to colors
    s : float, optional (default: 10)
        Scatter point size
    figsize : tuple, optional
        Figure size if creating new figure
    title : str, optional
        Plot title. If None, auto-generates based on flow_key.
    n_grid : int, optional (default: 40)
        Number of grid points per dimension (when on_grid=True)
    n_neighbors : int, optional (default: 200)
        Number of neighbors for grid interpolation
    min_mass : float, optional (default: 1.0)
        Minimum probability mass to show arrows
    recalculate : bool, optional (default: False)
        If True, recalculate grid flow even if cached
    n_jobs : int, optional (default: 4)
        Number of parallel jobs
    **quiver_kwargs
        Additional arguments for matplotlib quiver

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    embedding_key = f'X_{basis}'
    embedding = adata.obsm[embedding_key]

    if flow_key is None:
        flow_key = f'perturbation_flow_{basis}'

    if flow_key not in adata.obsm:
        raise ValueError(f"Flow '{flow_key}' not found. Run calculate_flow first.")

    # Background scatter
    if show_background:
        if cluster_key is not None and colors is not None:
            c = [colors.get(cl, 'lightgray') for cl in adata.obs[cluster_key]]
        else:
            c = 'lightgray'
        ax.scatter(embedding[:, 0], embedding[:, 1], c=c,
                  s=s, alpha=0.5, rasterized=True)

    # Default quiver settings
    default_quiver = dict(
        headaxislength=4, headlength=5, headwidth=4,
        linewidths=0.5, width=0.003
    )
    default_quiver.update(quiver_kwargs)

    if on_grid:
        # Interpolate to grid
        grid_key = f'grid_flow_{flow_key}'
        if grid_key in adata.uns and not recalculate:
            grid_data = adata.uns[grid_key]
        else:
            grid_data = calculate_grid_flow(
                adata, flow_key=flow_key, basis=basis, n_grid=n_grid,
                n_neighbors=n_neighbors, min_mass=min_mass, n_jobs=n_jobs
            )
            adata.uns[grid_key] = grid_data

        grid_coords = grid_data['grid_coords']
        grid_flow = grid_data['grid_flow']
        mass_filter = grid_data['mass_filter']
        valid = ~mass_filter

        default_quiver['width'] = 0.004  # Slightly wider for grid

        ax.quiver(
            grid_coords[valid, 0], grid_coords[valid, 1],
            grid_flow[valid, 0], grid_flow[valid, 1],
            color=color, alpha=alpha, scale=scale,
            **default_quiver
        )
    else:
        # Plot directly on cells
        flow = adata.obsm[flow_key]
        ax.quiver(
            embedding[:, 0], embedding[:, 1],
            flow[:, 0], flow[:, 1],
            color=color, alpha=alpha, scale=scale,
            **default_quiver
        )

    # Title
    if title is None:
        if 'perturbation' in flow_key:
            if 'scHopfield' in adata.uns and 'perturb_condition' in adata.uns['scHopfield']:
                perturb = adata.uns['scHopfield']['perturb_condition']
                perturb_str = ', '.join([f"{k}={'KO' if v==0 else v}" for k, v in perturb.items()])
                title = f'Perturbation Flow: {perturb_str}'
            else:
                title = 'Perturbation Flow'
        else:
            title = flow_key.replace('_', ' ').title()

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    ax.set_aspect('equal')

    return ax


def plot_inner_product(
    adata: AnnData,
    basis: str = 'umap',
    by_cluster: bool = False,
    cluster_key: str = 'cell_type',
    ax: Optional[plt.Axes] = None,
    inner_product_key: str = 'perturbation_inner_product',
    vmin: float = -1,
    vmax: float = 1,
    cmap: str = 'RdBu_r',
    s: float = 15,
    figsize: Tuple[float, float] = (8, 8),
    title: Optional[str] = None,
    show_colorbar: bool = True,
    order: Optional[List[str]] = None,
    colors: Optional[Dict[str, str]] = None,
    on_grid: bool = False,
    n_grid: int = 40,
    min_mass: float = 1.0,
) -> plt.Axes:
    """
    Plot inner product values on embedding or by cluster.

    Parameters
    ----------
    adata : AnnData
        Annotated data with inner product calculated
    basis : str, optional (default: 'umap')
        Embedding basis
    by_cluster : bool, optional (default: False)
        If True, show boxplot by cluster. If False, show on embedding.
    cluster_key : str, optional (default: 'cell_type')
        Key for cluster labels
    ax : plt.Axes, optional
        Axes to plot on
    inner_product_key : str, optional (default: 'perturbation_inner_product')
        Key in adata.obs for inner product values
    vmin, vmax : float, optional
        Color scale limits
    cmap : str, optional (default: 'RdBu_r')
        Colormap
    s : float, optional (default: 15)
        Point size
    figsize : tuple, optional
        Figure size
    title : str, optional
        Plot title
    show_colorbar : bool, optional (default: True)
        Whether to show colorbar (embedding mode)
    order : list, optional
        Order of clusters (cluster mode)
    colors : dict, optional
        Colors for clusters (cluster mode)

    Returns
    -------
    plt.Axes
    """
    if inner_product_key not in adata.obs:
        raise ValueError(f"Inner product '{inner_product_key}' not found. "
                        "Run calculate_inner_product first.")

    if by_cluster:
        return _plot_inner_product_by_cluster(
            adata, cluster_key=cluster_key, ax=ax,
            inner_product_key=inner_product_key,
            figsize=figsize, title=title, order=order, colors=colors
        )
    else:
        return _plot_inner_product_on_embedding(
            adata, basis=basis, ax=ax, inner_product_key=inner_product_key,
            vmin=vmin, vmax=vmax, cmap=cmap, s=s, figsize=figsize,
            title=title, show_colorbar=show_colorbar,
            on_grid=on_grid, n_grid=n_grid, min_mass=min_mass
        )


def _plot_inner_product_on_embedding(
    adata: AnnData,
    basis: str = 'umap',
    ax: Optional[plt.Axes] = None,
    inner_product_key: str = 'perturbation_inner_product',
    vmin: float = -1,
    vmax: float = 1,
    cmap: str = 'RdBu_r',
    s: float = 15,
    figsize: Tuple[float, float] = (8, 8),
    title: Optional[str] = None,
    show_colorbar: bool = True,
    on_grid: bool = False,
    n_grid: int = 40,
    min_mass: float = 1.0,
) -> plt.Axes:
    """Plot inner product on embedding."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    embedding = adata.obsm[f'X_{basis}']
    inner_product = adata.obs[inner_product_key].values

    try:
        norm = mpl_colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    except Exception:
        norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax)

    if on_grid:
        from ..tools.flow import calculate_grid_scalar
        grid_data = calculate_grid_scalar(
            adata, scalar_key=inner_product_key, basis=basis, n_grid=n_grid,
            min_mass=min_mass
        )
        grid_coords = grid_data['grid_coords']
        grid_scalar = grid_data['grid_scalar']
        mass_filter = grid_data['mass_filter']
        valid = ~mass_filter
        
        # Use a slightly larger point size for grid, or fallback to s
        grid_s = s * 4 if on_grid else s
        
        sc = ax.scatter(grid_coords[valid, 0], grid_coords[valid, 1], c=grid_scalar[valid],
                       cmap=cmap, norm=norm, s=grid_s, rasterized=True)
    else:
        sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=inner_product,
                       cmap=cmap, norm=norm, s=s, rasterized=True)

    if show_colorbar:
        cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
        cbar.set_label('Inner Product', fontsize=10)

    if title is None:
        title = 'Inner Product\n(Perturbation vs Reference)'
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    ax.set_aspect('equal')

    return ax


def _plot_inner_product_by_cluster(
    adata: AnnData,
    cluster_key: str = 'cell_type',
    ax: Optional[plt.Axes] = None,
    inner_product_key: str = 'perturbation_inner_product',
    figsize: Tuple[float, float] = (10, 5),
    title: Optional[str] = None,
    order: Optional[List[str]] = None,
    colors: Optional[Dict[str, str]] = None,
) -> plt.Axes:
    """Plot inner product by cluster."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    df = pd.DataFrame({
        'Cluster': adata.obs[cluster_key].values,
        'Inner Product': adata.obs[inner_product_key].values
    })

    if order is None:
        order = df.groupby('Cluster')['Inner Product'].median().sort_values().index.tolist()

    palette = None
    if colors is not None:
        palette = [colors.get(c, '#cccccc') for c in order]

    sns.boxplot(data=df, x='Cluster', y='Inner Product', order=order,
               palette=palette, ax=ax)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Cluster', fontsize=11)
    ax.set_ylabel('Inner Product Score', fontsize=11)

    if title is None:
        title = 'Inner Product by Cluster'
    ax.set_title(title, fontsize=12, fontweight='bold')

    if len(order) > 5:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    ax.grid(True, alpha=0.3, axis='y')
    sns.despine()

    return ax


# =============================================================================
# Additional Plotting Functions
# =============================================================================


