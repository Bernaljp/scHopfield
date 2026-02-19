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
            title=title, show_colorbar=show_colorbar
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
) -> plt.Axes:
    """Plot inner product on embedding."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    embedding = adata.obsm[f'X_{basis}']
    inner_product = adata.obs[inner_product_key].values

    try:
        norm = mpl_colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    except:
        norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax)

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


def visualize_flow_comparison(
    adata: AnnData,
    flows: Optional[List[str]] = None,
    basis: str = 'umap',
    cluster_key: str = 'cell_type',
    colors: Optional[Dict[str, str]] = None,
    scale: float = 1.0,
    figsize: Tuple[float, float] = (20, 6),
    n_neighbors: int = 30,
    n_jobs: int = 4,
    use_cluster_specific: bool = True,
    **kwargs
) -> plt.Figure:
    """
    Multi-panel comparison of different flow fields.

    Parameters
    ----------
    adata : AnnData
        Annotated data with perturbation results
    flows : list of str, optional
        Flow types to compare. If None, uses ['original', 'perturbed', 'delta'].
        Valid options: 'original', 'perturbed', 'delta', or any flow_key in obsm.
    basis : str, optional (default: 'umap')
        Embedding basis
    cluster_key : str, optional (default: 'cell_type')
        Cluster key
    colors : dict, optional
        Cluster colors
    scale : float, optional (default: 1.0)
        Arrow scale
    figsize : tuple, optional
        Figure size
    n_neighbors : int, optional (default: 30)
        Neighbors for flow calculation
    n_jobs : int, optional (default: 4)
        Parallel jobs
    use_cluster_specific : bool, optional (default: True)
        Use cluster-specific GRNs

    Returns
    -------
    plt.Figure
    """
    if flows is None:
        flows = ['clusters', 'original', 'perturbed', 'delta']

    n_panels = len(flows)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    embedding = adata.obsm[f'X_{basis}']

    # Get perturbation info
    perturb_str = "Perturbation"
    if 'scHopfield' in adata.uns and 'perturb_condition' in adata.uns['scHopfield']:
        perturb = adata.uns['scHopfield']['perturb_condition']
        perturb_str = ', '.join([f"{k}={'KO' if v==0 else v}" for k, v in perturb.items()])

    flow_colors = {
        'original': '#3498DB',
        'perturbed': '#27AE60',
        'delta': '#E74C3C'
    }

    for i, flow_type in enumerate(flows):
        ax = axes[i]

        if flow_type == 'clusters':
            # Show clusters
            if colors is not None:
                c = [colors.get(cl, 'gray') for cl in adata.obs[cluster_key]]
            else:
                c = adata.obs[cluster_key].astype('category').cat.codes
            ax.scatter(embedding[:, 0], embedding[:, 1], c=c, s=10, alpha=0.7)
            ax.set_title('Clusters', fontsize=12, fontweight='bold')

        elif flow_type in ['original', 'perturbed', 'delta']:
            # Calculate and plot flow
            flow_key = f'{flow_type}_velocity_flow_{basis}' if flow_type != 'delta' else f'perturbation_flow_{basis}'

            if flow_key not in adata.obsm:
                # Calculate flow
                source = flow_type if flow_type != 'delta' else 'delta'
                calculate_flow(
                    adata, source=source, basis=basis,
                    cluster_key=cluster_key, use_cluster_specific=use_cluster_specific,
                    n_neighbors=n_neighbors, n_jobs=n_jobs, verbose=False
                )

            # Background
            if colors is not None:
                c = [colors.get(cl, 'lightgray') for cl in adata.obs[cluster_key]]
            else:
                c = 'lightgray'
            ax.scatter(embedding[:, 0], embedding[:, 1], c=c, s=10, alpha=0.5)

            # Flow arrows
            flow = adata.obsm[flow_key]
            ax.quiver(
                embedding[:, 0], embedding[:, 1],
                flow[:, 0], flow[:, 1],
                color=flow_colors.get(flow_type, 'black'),
                alpha=0.8, scale=scale,
                headaxislength=4, headlength=5, headwidth=4
            )

            title_map = {
                'original': 'Original Hopfield Velocity',
                'perturbed': f'Perturbed Velocity\n({perturb_str})',
                'delta': 'Delta Velocity\n(Perturbed - Original)'
            }
            ax.set_title(title_map.get(flow_type, flow_type), fontsize=12, fontweight='bold')

        else:
            # Custom flow key
            if flow_type in adata.obsm:
                # Background
                if colors is not None:
                    c = [colors.get(cl, 'lightgray') for cl in adata.obs[cluster_key]]
                else:
                    c = 'lightgray'
                ax.scatter(embedding[:, 0], embedding[:, 1], c=c, s=10, alpha=0.5)

                flow = adata.obsm[flow_type]
                ax.quiver(
                    embedding[:, 0], embedding[:, 1],
                    flow[:, 0], flow[:, 1],
                    color='black', alpha=0.8, scale=scale,
                    headaxislength=4, headlength=5, headwidth=4
                )
                ax.set_title(flow_type.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            else:
                ax.text(0.5, 0.5, f'Flow not found:\n{flow_type}',
                       ha='center', va='center', transform=ax.transAxes)

        ax.axis('off')
        ax.set_aspect('equal')

    fig.suptitle(f'Hopfield Velocity Analysis: {perturb_str}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def visualize_perturbation_flow(
    adata: AnnData,
    basis: str = 'umap',
    velocity_key: Optional[str] = None,
    cluster_key: str = 'cell_type',
    colors: Optional[Dict[str, str]] = None,
    scale_reference: float = 1.0,
    scale_perturbation: float = 1.0,
    figsize: Tuple[float, float] = (20, 10),
    vm: float = 1.0
) -> plt.Figure:
    """
    Create comprehensive visualization of perturbation flow analysis.

    Creates a 2x3 figure with:
    - Row 0: Clusters, Reference velocity, Perturbation flow
    - Row 1: Inner product on embedding, Inner product + flow, Inner product by cluster

    Parameters
    ----------
    adata : AnnData
        Annotated data with perturbation simulation results
    basis : str, optional (default: 'umap')
        Embedding basis
    velocity_key : str, optional
        Key for reference velocity
    cluster_key : str, optional (default: 'cell_type')
        Key for cluster labels
    colors : dict, optional
        Colors for clusters
    scale_reference : float, optional (default: 1.0)
        Scale for reference flow arrows
    scale_perturbation : float, optional (default: 1.0)
        Scale for perturbation flow arrows
    figsize : tuple, optional
        Figure size
    vm : float, optional (default: 1.0)
        Max value for inner product colorscale

    Returns
    -------
    plt.Figure
    """
    # Get perturbation info
    perturb_str = "Perturbation"
    if 'scHopfield' in adata.uns and 'perturb_condition' in adata.uns['scHopfield']:
        perturb = adata.uns['scHopfield']['perturb_condition']
        perturb_str = ', '.join([f"{k}={'KO' if v==0 else v}" for k, v in perturb.items()])

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    embedding = adata.obsm[f'X_{basis}']

    # Row 0, Col 0: Clusters
    ax = axes[0, 0]
    if colors is not None:
        c = [colors.get(cl, 'gray') for cl in adata.obs[cluster_key]]
    else:
        c = adata.obs[cluster_key].astype('category').cat.codes
    ax.scatter(embedding[:, 0], embedding[:, 1], c=c, s=10, alpha=0.7, rasterized=True)
    ax.set_title('Clusters', fontsize=12, fontweight='bold')
    ax.axis('off')
    ax.set_aspect('equal')

    # Row 0, Col 1: Reference velocity
    ax = axes[0, 1]
    if velocity_key is None:
        velocity_key = f'velocity_{basis}'
    try:
        plot_reference_flow(
            adata, basis=basis, velocity_key=velocity_key, ax=ax,
            scale=scale_reference, title='Reference Velocity'
        )
    except ValueError as e:
        ax.text(0.5, 0.5, f'No velocity data\n({e})',
               ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

    # Row 0, Col 2: Perturbation flow
    ax = axes[0, 2]
    try:
        plot_flow(
            adata, basis=basis, ax=ax, scale=scale_perturbation,
            cluster_key=cluster_key, colors=colors,
            title=f'Perturbation Flow\n({perturb_str})', color='#EC7063'
        )
    except ValueError as e:
        ax.text(0.5, 0.5, f'No perturbation flow\n({e})',
               ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

    # Row 1, Col 0: Inner product on embedding
    ax = axes[1, 0]
    try:
        plot_inner_product(
            adata, basis=basis, ax=ax, vmin=-vm, vmax=vm,
            title='Inner Product\n(Perturbation \u00d7 Reference)'
        )
    except ValueError as e:
        ax.text(0.5, 0.5, f'No inner product\n({e})',
               ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

    # Row 1, Col 1: Inner product + flow overlay
    ax = axes[1, 1]
    try:
        plot_inner_product(
            adata, basis=basis, ax=ax, vmin=-vm, vmax=vm,
            show_colorbar=False, s=10, title=''
        )
        plot_flow(
            adata, basis=basis, ax=ax,
            show_background=False, color='black', alpha=0.6
        )
        ax.set_title('Inner Product + Flow', fontsize=12, fontweight='bold')
    except ValueError:
        ax.axis('off')

    # Row 1, Col 2: Inner product by cluster
    ax = axes[1, 2]
    try:
        plot_inner_product(
            adata, by_cluster=True, cluster_key=cluster_key, ax=ax,
            colors=colors, title='Inner Product by Cluster'
        )
    except ValueError as e:
        ax.text(0.5, 0.5, f'No inner product\n({e})',
               ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

    fig.suptitle(f'Perturbation Analysis: {perturb_str}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


# =============================================================================
# Additional Plotting Functions
# =============================================================================

def plot_reference_flow(
    adata: AnnData,
    basis: str = 'umap',
    velocity_key: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    scale: float = 1.0,
    color: str = 'black',
    alpha: float = 0.8,
    show_background: bool = True,
    background_color: str = 'lightgray',
    s: float = 10,
    figsize: Tuple[float, float] = (8, 8),
    title: str = 'Reference Velocity',
    **quiver_kwargs
) -> plt.Axes:
    """
    Plot reference velocity flow (e.g., from scVelo).

    Parameters
    ----------
    adata : AnnData
        Annotated data with velocity
    basis : str, optional (default: 'umap')
        Embedding basis
    velocity_key : str, optional
        Key for velocity in obsm
    ax : plt.Axes, optional
        Axes to plot on
    scale : float, optional (default: 1.0)
        Scale factor for arrows
    color : str, optional (default: 'black')
        Arrow color
    alpha : float, optional (default: 0.8)
        Arrow transparency
    show_background : bool, optional (default: True)
        Show background scatter
    background_color : str, optional (default: 'lightgray')
        Background scatter color
    s : float, optional (default: 10)
        Scatter point size
    figsize : tuple, optional
        Figure size
    title : str, optional
        Plot title
    **quiver_kwargs
        Additional arguments for quiver plot

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    embedding = adata.obsm[f'X_{basis}']

    if velocity_key is None:
        velocity_key = f'velocity_{basis}'

    if velocity_key not in adata.obsm:
        raise ValueError(f"Velocity '{velocity_key}' not found in adata.obsm")

    velocity = adata.obsm[velocity_key]

    if show_background:
        ax.scatter(embedding[:, 0], embedding[:, 1], c=background_color,
                  s=s, alpha=0.5, rasterized=True)

    default_quiver = dict(headaxislength=4, headlength=5, headwidth=4,
                         linewidths=0.5, width=0.003)
    default_quiver.update(quiver_kwargs)

    ax.quiver(embedding[:, 0], embedding[:, 1],
             velocity[:, 0], velocity[:, 1],
             color=color, alpha=alpha, scale=scale,
             **default_quiver)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    ax.set_aspect('equal')

    return ax


def plot_ode_perturbation_flow(
    adata: AnnData,
    basis: str = 'umap',
    ax: Optional[plt.Axes] = None,
    scale: float = 1.0,
    color: str = '#9B59B6',
    alpha: float = 0.8,
    show_background: bool = True,
    cluster_key: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    s: float = 10,
    figsize: Tuple[float, float] = (8, 8),
    title: str = 'ODE Perturbation Flow',
    **quiver_kwargs
) -> plt.Axes:
    """
    Plot ODE trajectory perturbation flow.

    Parameters
    ----------
    adata : AnnData
        Annotated data with ODE perturbation flow
    basis : str, optional (default: 'umap')
        Embedding basis
    ax : plt.Axes, optional
        Axes to plot on
    scale : float, optional (default: 1.0)
        Arrow scale
    color : str, optional (default: '#9B59B6')
        Arrow color (purple)
    alpha : float, optional (default: 0.8)
        Arrow transparency
    show_background : bool, optional (default: True)
        Show background scatter
    cluster_key : str, optional
        Cluster key for coloring
    colors : dict, optional
        Cluster colors
    s : float, optional (default: 10)
        Point size
    figsize : tuple, optional
        Figure size
    title : str, optional
        Plot title
    **quiver_kwargs
        Additional quiver arguments

    Returns
    -------
    plt.Axes
    """
    flow_key = f'ode_perturbation_flow_{basis}'

    if flow_key not in adata.obsm:
        raise ValueError(f"ODE flow '{flow_key}' not found. "
                        "Run calculate_trajectory_flow first.")

    return plot_flow(
        adata, flow_key=flow_key, basis=basis, ax=ax,
        scale=scale, color=color, alpha=alpha,
        show_background=show_background, cluster_key=cluster_key,
        colors=colors, s=s, figsize=figsize, title=title,
        **quiver_kwargs
    )


def visualize_ode_perturbation(
    adata: AnnData,
    wt_trajectories: Dict[str, np.ndarray],
    perturbed_trajectories: Dict[str, np.ndarray],
    gene_perturbations: Dict[str, float],
    t_span: np.ndarray,
    cluster_key: str = 'cell_type',
    basis: str = 'umap',
    velocity_key: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    method: str = 'hopfield',
    figsize: Tuple[float, float] = (20, 10),
    scale_flow: float = 1.0,
    vm: float = 1.0
) -> plt.Figure:
    """
    Create comprehensive visualization of ODE perturbation analysis.

    Parameters
    ----------
    adata : AnnData
        Annotated data
    wt_trajectories : dict
        WT trajectories per cluster
    perturbed_trajectories : dict
        Perturbed trajectories per cluster
    gene_perturbations : dict
        Perturbation conditions
    t_span : np.ndarray
        Time points
    cluster_key : str, optional
        Cluster key
    basis : str, optional
        Embedding basis
    velocity_key : str, optional
        Reference velocity key
    colors : dict, optional
        Cluster colors
    method : str, optional (default: 'hopfield')
        Flow calculation method
    figsize : tuple, optional
        Figure size
    scale_flow : float, optional
        Arrow scale
    vm : float, optional
        Inner product colorscale max

    Returns
    -------
    plt.Figure
    """
    # Import and calculate flow
    from ..dynamics.simulation import calculate_trajectory_flow

    calculate_trajectory_flow(
        adata, wt_trajectories, perturbed_trajectories,
        cluster_key=cluster_key, basis=basis, method=method
    )

    # Calculate inner product
    if velocity_key is None:
        velocity_key = f'velocity_{basis}'

    if velocity_key in adata.obsm:
        flow_key = f'ode_perturbation_flow_{basis}'
        calculate_inner_product(
            adata, velocity_key, flow_key,
            store_key='ode_perturbation_inner_product'
        )

    # Create figure
    perturb_str = ', '.join([f"{k}={'KO' if v==0 else 'OE' if v>0 else v}"
                             for k, v in gene_perturbations.items()])

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    embedding = adata.obsm[f'X_{basis}']

    # Row 0, Col 0: Clusters
    ax = axes[0, 0]
    if colors is not None:
        c = [colors.get(cl, 'gray') for cl in adata.obs[cluster_key]]
    else:
        c = adata.obs[cluster_key].astype('category').cat.codes
    ax.scatter(embedding[:, 0], embedding[:, 1], c=c, s=10, alpha=0.7)
    ax.set_title('Clusters', fontsize=12, fontweight='bold')
    ax.axis('off')

    # Row 0, Col 1: Reference velocity
    ax = axes[0, 1]
    try:
        plot_reference_flow(adata, basis=basis, velocity_key=velocity_key,
                          ax=ax, scale=scale_flow*5, title='Reference Velocity')
    except:
        ax.text(0.5, 0.5, 'No velocity data', ha='center', va='center',
               transform=ax.transAxes)
        ax.axis('off')

    # Row 0, Col 2: ODE perturbation flow
    ax = axes[0, 2]
    plot_ode_perturbation_flow(
        adata, basis=basis, ax=ax, scale=scale_flow,
        cluster_key=cluster_key, colors=colors,
        title=f'ODE Perturbation Flow\n({perturb_str})'
    )

    # Row 1, Col 0: Inner product
    ax = axes[1, 0]
    if 'ode_perturbation_inner_product' in adata.obs:
        try:
            norm = mpl_colors.TwoSlopeNorm(vmin=-vm, vcenter=0, vmax=vm)
        except:
            norm = mpl_colors.Normalize(vmin=-vm, vmax=vm)
        sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                       c=adata.obs['ode_perturbation_inner_product'],
                       cmap='RdBu_r', norm=norm, s=15)
        plt.colorbar(sc, ax=ax, shrink=0.6)
        ax.set_title('Inner Product (ODE)', fontsize=12, fontweight='bold')
        ax.axis('off')
    else:
        ax.text(0.5, 0.5, 'No inner product', ha='center', va='center',
               transform=ax.transAxes)
        ax.axis('off')

    # Row 1, Col 1: Trajectory examples
    ax = axes[1, 1]
    genes = get_genes_used(adata)
    gene_names = adata.var.index[genes]

    cluster = list(wt_trajectories.keys())[0]
    wt = wt_trajectories[cluster]
    pert = perturbed_trajectories[cluster]

    delta_final = np.abs(pert[-1] - wt[-1])
    top_gene_idx = np.argsort(delta_final)[-3:]

    for idx in top_gene_idx:
        ax.plot(t_span, wt[:, idx], '-', label=f'{gene_names[idx]} (WT)')
        ax.plot(t_span, pert[:, idx], '--', label=f'{gene_names[idx]} (Pert)')

    ax.set_xlabel('Time')
    ax.set_ylabel('Expression')
    ax.set_title(f'Trajectory ({cluster})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 1, Col 2: Inner product by cluster
    ax = axes[1, 2]
    if 'ode_perturbation_inner_product' in adata.obs:
        df = pd.DataFrame({
            'Cluster': adata.obs[cluster_key],
            'Inner Product': adata.obs['ode_perturbation_inner_product']
        })
        cluster_order = df.groupby('Cluster')['Inner Product'].median().sort_values().index
        palette = [colors.get(c, 'gray') for c in cluster_order] if colors else None
        sns.boxplot(data=df, x='Cluster', y='Inner Product',
                   order=cluster_order, palette=palette, ax=ax)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('Inner Product by Cluster', fontsize=12, fontweight='bold')
        if len(cluster_order) > 5:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    else:
        ax.axis('off')

    fig.suptitle(f'ODE Perturbation Analysis: {perturb_str}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig
