"""
Flow visualization functions for perturbation analysis.

Inspired by CellOracle's development module visualization.
Compares reference velocity flow with perturbation-induced flow.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from typing import Optional, List, Dict, Tuple, Union
from anndata import AnnData
from scipy.sparse import issparse

from .._utils.io import get_genes_used


def calculate_perturbation_flow(
    adata: AnnData,
    basis: str = 'umap',
    n_neighbors: int = 50,
    method: str = 'knn_projection'
) -> np.ndarray:
    """
    Calculate perturbation-induced flow in embedding space.

    Projects delta_X (gene expression change) to embedding coordinates
    to visualize how perturbation affects cell state transitions.

    Parameters
    ----------
    adata : AnnData
        Annotated data with delta_X layer from perturbation simulation
    basis : str, optional (default: 'umap')
        Embedding basis to project onto
    n_neighbors : int, optional (default: 50)
        Number of neighbors for projection
    method : str, optional (default: 'knn_projection')
        Method for projection: 'knn_projection' or 'correlation'

    Returns
    -------
    np.ndarray
        Perturbation flow vectors in embedding space (n_cells, 2)
    """
    if 'delta_X' not in adata.layers:
        raise ValueError("No delta_X found. Run simulate_shift first.")

    embedding_key = f'X_{basis}'
    if embedding_key not in adata.obsm:
        raise ValueError(f"Embedding {embedding_key} not found in adata.obsm")

    embedding = adata.obsm[embedding_key]

    # Get delta_X for genes used in analysis
    genes = get_genes_used(adata)
    delta_X = adata.layers['delta_X'][:, genes]

    # Get base expression (sigmoid transformed)
    if 'sigmoid' in adata.layers:
        X = adata.layers['sigmoid'][:, genes]
    else:
        X = adata.X[:, genes]
        if issparse(X):
            X = X.toarray()

    # Simulated expression
    X_sim = X + delta_X

    # Project to embedding space using KNN
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X)

    # For each cell, find where the simulated state would be in embedding
    distances, indices = nn.kneighbors(X_sim)

    # Weight by inverse distance
    weights = 1 / (distances + 1e-6)
    weights = weights / weights.sum(axis=1, keepdims=True)

    # Predicted embedding position after perturbation
    predicted_embedding = np.zeros_like(embedding)
    for i in range(len(embedding)):
        predicted_embedding[i] = np.average(embedding[indices[i]], axis=0, weights=weights[i])

    # Flow = predicted - current
    flow = predicted_embedding - embedding

    # Store in adata
    adata.obsm[f'perturbation_flow_{basis}'] = flow

    return flow


def calculate_grid_flow(
    adata: AnnData,
    basis: str = 'umap',
    n_grid: int = 40,
    smooth: float = 0.5,
    min_mass: float = 1.0,
    flow_key: Optional[str] = None
) -> Dict:
    """
    Calculate flow vectors on a regular grid.

    Parameters
    ----------
    adata : AnnData
        Annotated data with flow vectors
    basis : str, optional (default: 'umap')
        Embedding basis
    n_grid : int, optional (default: 40)
        Number of grid points per dimension
    smooth : float, optional (default: 0.5)
        Smoothing factor for grid interpolation
    min_mass : float, optional (default: 1.0)
        Minimum cell density to show arrows
    flow_key : str, optional
        Key in obsm for flow vectors. If None, uses 'perturbation_flow_{basis}'

    Returns
    -------
    dict
        Dictionary with grid coordinates, flow vectors, and mass filter
    """
    embedding_key = f'X_{basis}'
    embedding = adata.obsm[embedding_key]

    if flow_key is None:
        flow_key = f'perturbation_flow_{basis}'

    if flow_key not in adata.obsm:
        raise ValueError(f"Flow {flow_key} not found. Run calculate_perturbation_flow first.")

    flow = adata.obsm[flow_key]

    # Create grid
    x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()

    # Add padding
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05

    gx = np.linspace(x_min - x_pad, x_max + x_pad, n_grid)
    gy = np.linspace(y_min - y_pad, y_max + y_pad, n_grid)

    grid_x, grid_y = np.meshgrid(gx, gy)
    grid_coords = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Calculate flow on grid using Gaussian kernel
    from sklearn.neighbors import KernelDensity

    # Bandwidth based on grid spacing
    bandwidth = smooth * max((x_max - x_min), (y_max - y_min)) / n_grid

    # Density estimation for mass filter
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(embedding)
    log_density = kde.score_samples(grid_coords)
    mass = np.exp(log_density)

    # Normalize mass
    mass = mass / mass.max()
    mass_filter = mass < (min_mass / 100)  # Filter low density regions

    # Interpolate flow to grid
    grid_flow = np.zeros((len(grid_coords), 2))

    for i, gc in enumerate(grid_coords):
        # Distance to all cells
        dists = np.sqrt(((embedding - gc) ** 2).sum(axis=1))
        weights = np.exp(-dists ** 2 / (2 * bandwidth ** 2))
        weights = weights / (weights.sum() + 1e-10)

        grid_flow[i] = np.average(flow, axis=0, weights=weights)

    return {
        'grid_coords': grid_coords,
        'grid_flow': grid_flow,
        'mass_filter': mass_filter,
        'mass': mass,
        'n_grid': n_grid
    }


def calculate_inner_product(
    adata: AnnData,
    basis: str = 'umap',
    velocity_key: Optional[str] = None,
    perturbation_flow_key: Optional[str] = None
) -> np.ndarray:
    """
    Calculate inner product between reference velocity and perturbation flow.

    Positive values indicate perturbation promotes the developmental direction.
    Negative values indicate perturbation opposes the developmental direction.

    Parameters
    ----------
    adata : AnnData
        Annotated data with velocity and perturbation flow
    basis : str, optional (default: 'umap')
        Embedding basis
    velocity_key : str, optional
        Key in obsm for reference velocity. If None, tries 'velocity_{basis}'
    perturbation_flow_key : str, optional
        Key in obsm for perturbation flow. If None, uses 'perturbation_flow_{basis}'

    Returns
    -------
    np.ndarray
        Inner product values for each cell
    """
    # Get reference velocity
    if velocity_key is None:
        velocity_key = f'velocity_{basis}'

    if velocity_key not in adata.obsm:
        raise ValueError(f"Velocity {velocity_key} not found. "
                        "Please compute velocity first (e.g., using scVelo).")

    ref_velocity = adata.obsm[velocity_key]

    # Get perturbation flow
    if perturbation_flow_key is None:
        perturbation_flow_key = f'perturbation_flow_{basis}'

    if perturbation_flow_key not in adata.obsm:
        raise ValueError(f"Perturbation flow {perturbation_flow_key} not found. "
                        "Run calculate_perturbation_flow first.")

    pert_flow = adata.obsm[perturbation_flow_key]

    # Normalize vectors
    ref_norm = np.linalg.norm(ref_velocity, axis=1, keepdims=True) + 1e-10
    pert_norm = np.linalg.norm(pert_flow, axis=1, keepdims=True) + 1e-10

    ref_unit = ref_velocity / ref_norm
    pert_unit = pert_flow / pert_norm

    # Inner product (dot product)
    inner_product = np.sum(ref_unit * pert_unit, axis=1)

    # Also compute magnitude-weighted version
    inner_product_weighted = np.sum(ref_velocity * pert_flow, axis=1)

    # Store in adata
    adata.obs['perturbation_inner_product'] = inner_product
    adata.obs['perturbation_inner_product_weighted'] = inner_product_weighted

    return inner_product


# =============================================================================
# Visualization Functions
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

    embedding_key = f'X_{basis}'
    embedding = adata.obsm[embedding_key]

    if velocity_key is None:
        velocity_key = f'velocity_{basis}'

    velocity = adata.obsm[velocity_key]

    # Background scatter
    if show_background:
        ax.scatter(embedding[:, 0], embedding[:, 1], c=background_color,
                  s=s, alpha=0.5, rasterized=True)

    # Quiver plot
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


def plot_perturbation_flow(
    adata: AnnData,
    basis: str = 'umap',
    flow_key: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    scale: float = 1.0,
    color: str = '#EC7063',
    alpha: float = 0.8,
    show_background: bool = True,
    cluster_key: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    s: float = 10,
    figsize: Tuple[float, float] = (8, 8),
    title: str = 'Perturbation Flow',
    **quiver_kwargs
) -> plt.Axes:
    """
    Plot perturbation-induced flow.

    Parameters
    ----------
    adata : AnnData
        Annotated data with perturbation flow
    basis : str, optional (default: 'umap')
        Embedding basis
    flow_key : str, optional
        Key for flow in obsm
    ax : plt.Axes, optional
        Axes to plot on
    scale : float, optional (default: 1.0)
        Scale factor for arrows
    color : str, optional (default: '#EC7063')
        Arrow color (ignored if colors dict provided)
    alpha : float, optional (default: 0.8)
        Arrow transparency
    show_background : bool, optional (default: True)
        Show background scatter with cluster colors
    cluster_key : str, optional
        Key for cluster labels (for coloring)
    colors : dict, optional
        Dictionary mapping cluster names to colors
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

    embedding_key = f'X_{basis}'
    embedding = adata.obsm[embedding_key]

    if flow_key is None:
        flow_key = f'perturbation_flow_{basis}'

    flow = adata.obsm[flow_key]

    # Background scatter
    if show_background:
        if cluster_key is not None and colors is not None:
            c = [colors.get(c, 'lightgray') for c in adata.obs[cluster_key]]
        else:
            c = 'lightgray'
        ax.scatter(embedding[:, 0], embedding[:, 1], c=c,
                  s=s, alpha=0.5, rasterized=True)

    # Quiver plot
    default_quiver = dict(headaxislength=4, headlength=5, headwidth=4,
                         linewidths=0.5, width=0.003)
    default_quiver.update(quiver_kwargs)

    ax.quiver(embedding[:, 0], embedding[:, 1],
             flow[:, 0], flow[:, 1],
             color=color, alpha=alpha, scale=scale,
             **default_quiver)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    ax.set_aspect('equal')

    return ax


def plot_flow_on_grid(
    adata: AnnData,
    grid_data: Dict,
    flow_type: str = 'perturbation',
    basis: str = 'umap',
    ax: Optional[plt.Axes] = None,
    scale: float = 1.0,
    color: str = 'black',
    alpha: float = 0.8,
    show_background: bool = True,
    cluster_key: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    s: float = 10,
    figsize: Tuple[float, float] = (8, 8),
    title: Optional[str] = None,
    **quiver_kwargs
) -> plt.Axes:
    """
    Plot flow vectors on a grid.

    Parameters
    ----------
    adata : AnnData
        Annotated data
    grid_data : dict
        Output from calculate_grid_flow
    flow_type : str, optional (default: 'perturbation')
        Type of flow: 'perturbation' or 'reference'
    basis : str, optional (default: 'umap')
        Embedding basis
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
    cluster_key : str, optional
        Key for cluster labels
    colors : dict, optional
        Colors for clusters
    s : float, optional (default: 10)
        Scatter point size
    figsize : tuple, optional
        Figure size
    title : str, optional
        Plot title
    **quiver_kwargs
        Additional arguments for quiver

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    embedding_key = f'X_{basis}'
    embedding = adata.obsm[embedding_key]

    # Background scatter
    if show_background:
        if cluster_key is not None and colors is not None:
            c = [colors.get(c, 'lightgray') for c in adata.obs[cluster_key]]
        else:
            c = 'lightgray'
        ax.scatter(embedding[:, 0], embedding[:, 1], c=c,
                  s=s, alpha=0.3, rasterized=True)

    # Grid flow
    grid_coords = grid_data['grid_coords']
    grid_flow = grid_data['grid_flow']
    mass_filter = grid_data['mass_filter']

    # Filter by mass
    valid = ~mass_filter

    default_quiver = dict(headaxislength=4, headlength=5, headwidth=4,
                         linewidths=0.5, width=0.004)
    default_quiver.update(quiver_kwargs)

    ax.quiver(grid_coords[valid, 0], grid_coords[valid, 1],
             grid_flow[valid, 0], grid_flow[valid, 1],
             color=color, alpha=alpha, scale=scale,
             **default_quiver)

    if title is None:
        title = f'{flow_type.capitalize()} Flow (Grid)'
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    ax.set_aspect('equal')

    return ax


def plot_inner_product_on_embedding(
    adata: AnnData,
    basis: str = 'umap',
    ax: Optional[plt.Axes] = None,
    vmin: float = -1,
    vmax: float = 1,
    cmap: str = 'RdBu_r',
    s: float = 15,
    figsize: Tuple[float, float] = (8, 8),
    title: str = 'Inner Product\n(Perturbation vs Reference)',
    show_colorbar: bool = True
) -> plt.Axes:
    """
    Plot inner product between perturbation and reference flow on embedding.

    Parameters
    ----------
    adata : AnnData
        Annotated data with inner product calculated
    basis : str, optional (default: 'umap')
        Embedding basis
    ax : plt.Axes, optional
        Axes to plot on
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
        Whether to show colorbar

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if 'perturbation_inner_product' not in adata.obs:
        raise ValueError("Inner product not found. Run calculate_inner_product first.")

    embedding_key = f'X_{basis}'
    embedding = adata.obsm[embedding_key]
    inner_product = adata.obs['perturbation_inner_product'].values

    # Diverging norm centered at 0
    try:
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    except:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

    sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=inner_product,
                   cmap=cmap, norm=norm, s=s, rasterized=True)

    if show_colorbar:
        cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
        cbar.set_label('Inner Product', fontsize=10)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    ax.set_aspect('equal')

    return ax


def plot_inner_product_by_cluster(
    adata: AnnData,
    cluster_key: str = 'cell_type',
    ax: Optional[plt.Axes] = None,
    order: Optional[List[str]] = None,
    colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[float, float] = (10, 5),
    title: str = 'Inner Product by Cluster'
) -> plt.Axes:
    """
    Boxplot of inner product values by cluster.

    Parameters
    ----------
    adata : AnnData
        Annotated data with inner product calculated
    cluster_key : str, optional (default: 'cell_type')
        Key for cluster labels
    ax : plt.Axes, optional
        Axes to plot on
    order : list, optional
        Order of clusters
    colors : dict, optional
        Colors for clusters
    figsize : tuple, optional
        Figure size
    title : str, optional
        Plot title

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if 'perturbation_inner_product' not in adata.obs:
        raise ValueError("Inner product not found. Run calculate_inner_product first.")

    df = pd.DataFrame({
        'Cluster': adata.obs[cluster_key].values,
        'Inner Product': adata.obs['perturbation_inner_product'].values
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
    ax.set_title(title, fontsize=12, fontweight='bold')

    if len(order) > 5:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    ax.grid(True, alpha=0.3, axis='y')
    sns.despine()

    return ax


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
    Create a comprehensive visualization of perturbation flow analysis.

    Similar to CellOracle's visualize_development_module_layout.

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
    # Get perturbation info for title
    perturb_str = "Perturbation"
    if 'scHopfield' in adata.uns and 'perturb_condition' in adata.uns['scHopfield']:
        perturb = adata.uns['scHopfield']['perturb_condition']
        perturb_str = ', '.join([f"{k}={'KO' if v==0 else v}" for k, v in perturb.items()])

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    embedding_key = f'X_{basis}'
    embedding = adata.obsm[embedding_key]

    # Row 0, Col 0: Clusters
    ax = axes[0, 0]
    if colors is not None:
        c = [colors.get(c, 'gray') for c in adata.obs[cluster_key]]
    else:
        c = adata.obs[cluster_key].astype('category').cat.codes
    ax.scatter(embedding[:, 0], embedding[:, 1], c=c, s=10, alpha=0.7, rasterized=True)
    ax.set_title('Clusters', fontsize=12, fontweight='bold')
    ax.axis('off')
    ax.set_aspect('equal')

    # Row 0, Col 1: Reference velocity
    ax = axes[0, 1]
    try:
        plot_reference_flow(adata, basis=basis, velocity_key=velocity_key, ax=ax,
                          scale=scale_reference, title='Reference Velocity')
    except ValueError as e:
        ax.text(0.5, 0.5, f'No velocity data\n({e})', ha='center', va='center',
               transform=ax.transAxes)
        ax.axis('off')

    # Row 0, Col 2: Perturbation flow
    ax = axes[0, 2]
    try:
        plot_perturbation_flow(adata, basis=basis, ax=ax, scale=scale_perturbation,
                             cluster_key=cluster_key, colors=colors,
                             title=f'Perturbation Flow\n({perturb_str})')
    except ValueError as e:
        ax.text(0.5, 0.5, f'No perturbation flow\n({e})', ha='center', va='center',
               transform=ax.transAxes)
        ax.axis('off')

    # Row 1, Col 0: Inner product on embedding
    ax = axes[1, 0]
    try:
        plot_inner_product_on_embedding(adata, basis=basis, ax=ax, vmin=-vm, vmax=vm,
                                       title='Inner Product\n(Perturbation \u00d7 Reference)')
    except ValueError as e:
        ax.text(0.5, 0.5, f'No inner product\n({e})', ha='center', va='center',
               transform=ax.transAxes)
        ax.axis('off')

    # Row 1, Col 1: Inner product + perturbation flow overlay
    ax = axes[1, 1]
    try:
        plot_inner_product_on_embedding(adata, basis=basis, ax=ax, vmin=-vm, vmax=vm,
                                       show_colorbar=False, s=10, title='')
        plot_perturbation_flow(adata, basis=basis, ax=ax, scale=scale_perturbation,
                             show_background=False, color='black', alpha=0.6)
        ax.set_title('Inner Product + Flow', fontsize=12, fontweight='bold')
    except ValueError:
        ax.axis('off')

    # Row 1, Col 2: Inner product by cluster
    ax = axes[1, 2]
    try:
        plot_inner_product_by_cluster(adata, cluster_key=cluster_key, ax=ax,
                                     colors=colors, title='Inner Product by Cluster')
    except ValueError as e:
        ax.text(0.5, 0.5, f'No inner product\n({e})', ha='center', va='center',
               transform=ax.transAxes)
        ax.axis('off')

    fig.suptitle(f'Perturbation Analysis: {perturb_str}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig
