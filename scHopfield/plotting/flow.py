"""
Flow visualization functions for perturbation analysis.

Inspired by CellOracle's development module visualization.
Compares reference velocity flow with perturbation-induced flow.

Supports two methods for calculating perturbation flow:
1. 'celloracle': Correlation-based projection (like CellOracle)
2. 'hopfield': Direct computation using Hopfield model dynamics

References
----------
Logic for the transition vector field is inspired by the perturbation
simulation workflow in CellOracle:
Kamimoto et al. (2023). Nature. https://doi.org/10.1038/s41586-022-05688-9
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
from .._utils.math import sigmoid


# =============================================================================
# Hopfield Model Velocity Computation
# =============================================================================

def compute_hopfield_velocity(
    adata: AnnData,
    cluster: str,
    X: Optional[np.ndarray] = None,
    spliced_key: str = 'Ms'
) -> np.ndarray:
    """
    Compute velocity using the Hopfield model dynamics.

    Calculates: velocity = W @ sigmoid(X) - gamma * X + I

    This is the instantaneous rate of change predicted by the fitted
    Hopfield network for each cell's expression state.

    Parameters
    ----------
    adata : AnnData
        Annotated data with fitted interactions (W, I, sigmoid params)
    cluster : str
        Cluster name to use for getting W and I matrices
    X : np.ndarray, optional
        Expression matrix (n_cells, n_genes). If None, uses adata.layers[spliced_key]
    spliced_key : str, optional (default: 'Ms')
        Key for expression data if X is not provided

    Returns
    -------
    np.ndarray
        Velocity matrix (n_cells, n_genes)
    """
    genes = get_genes_used(adata)
    gene_names = adata.var.index[genes]

    # Get expression matrix
    if X is None:
        if spliced_key in adata.layers:
            X = adata.layers[spliced_key][:, genes]
        else:
            X = adata.X[:, genes]
        if issparse(X):
            X = X.toarray()

    # Get model parameters for this cluster
    W_key = f'W_{cluster}'
    I_key = f'I_{cluster}'

    if W_key not in adata.varp:
        raise ValueError(f"Interaction matrix {W_key} not found. Run fit_interactions first.")

    W = adata.varp[W_key][np.ix_(genes, genes)]
    I = adata.var[I_key][genes].values if I_key in adata.var else np.zeros(genes.sum())

    # Get sigmoid parameters
    threshold = adata.var.loc[gene_names, 'sigmoid_threshold'].values
    exponent = adata.var.loc[gene_names, 'sigmoid_exponent'].values

    # Get gamma
    if 'gamma' in adata.var.columns:
        gamma = adata.var.loc[gene_names, 'gamma'].values
    else:
        gamma = np.ones(len(gene_names))

    # Compute sigmoid activation
    sig_X = sigmoid(X, threshold, exponent)

    # Compute velocity: dX/dt = W @ sigmoid(X) - gamma * X + I
    velocity = sig_X @ W.T - gamma * X + I

    return velocity


def compute_hopfield_velocity_delta(
    adata: AnnData,
    cluster_key: str = 'cell_type',
    use_cluster_specific_GRN: bool = True,
    spliced_key: str = 'Ms'
) -> np.ndarray:
    """
    Compute delta velocity (perturbed - original) using Hopfield model.

    This computes the difference between velocity at perturbed state
    and velocity at original state, using the Hopfield dynamics directly.

    Parameters
    ----------
    adata : AnnData
        Annotated data with 'simulated_X' and original expression
    cluster_key : str, optional (default: 'cell_type')
        Key for cluster labels
    use_cluster_specific_GRN : bool, optional (default: True)
        Whether to use cluster-specific interaction matrices
    spliced_key : str, optional (default: 'Ms')
        Key for original expression data

    Returns
    -------
    np.ndarray
        Delta velocity matrix (n_cells, n_genes)
    """
    if 'simulated_X' not in adata.layers:
        raise ValueError("simulated_X not found. Run simulate_shift first.")

    genes = get_genes_used(adata)

    # Get original and perturbed expression
    if spliced_key in adata.layers:
        X_original = adata.layers[spliced_key][:, genes]
    else:
        X_original = adata.X[:, genes]
    if issparse(X_original):
        X_original = X_original.toarray()

    X_perturbed = adata.layers['simulated_X'][:, genes]
    if issparse(X_perturbed):
        X_perturbed = X_perturbed.toarray()

    n_cells = adata.n_obs
    n_genes = len(genes)
    delta_velocity = np.zeros((n_cells, n_genes))

    if use_cluster_specific_GRN:
        clusters = adata.obs[cluster_key].unique()

        for cluster in clusters:
            mask = adata.obs[cluster_key] == cluster
            cell_indices = np.where(mask)[0]

            # Compute velocities for this cluster
            v_original = compute_hopfield_velocity(
                adata, cluster, X=X_original[mask], spliced_key=spliced_key
            )
            v_perturbed = compute_hopfield_velocity(
                adata, cluster, X=X_perturbed[mask], spliced_key=spliced_key
            )

            delta_velocity[cell_indices] = v_perturbed - v_original
    else:
        # Use global/all cluster
        cluster = 'all'
        if f'W_{cluster}' not in adata.varp:
            # Fall back to first cluster
            cluster = adata.obs[cluster_key].unique()[0]

        v_original = compute_hopfield_velocity(adata, cluster, X=X_original, spliced_key=spliced_key)
        v_perturbed = compute_hopfield_velocity(adata, cluster, X=X_perturbed, spliced_key=spliced_key)
        delta_velocity = v_perturbed - v_original

    return delta_velocity


def project_velocity_to_embedding(
    adata: AnnData,
    velocity: np.ndarray,
    basis: str = 'umap',
    n_neighbors: int = 30,
    n_jobs: int = 4
) -> np.ndarray:
    """
    Project gene-space velocity to embedding space using neighbor averaging.

    Uses a simple approach: for each cell, compute the weighted average
    of neighbor directions based on the dot product with the velocity vector.

    Parameters
    ----------
    adata : AnnData
        Annotated data with embedding
    velocity : np.ndarray
        Velocity in gene space (n_cells, n_genes)
    basis : str, optional (default: 'umap')
        Embedding basis
    n_neighbors : int, optional (default: 30)
        Number of neighbors for projection
    n_jobs : int, optional (default: 4)
        Number of parallel jobs

    Returns
    -------
    np.ndarray
        Velocity in embedding space (n_cells, 2)
    """
    from sklearn.neighbors import NearestNeighbors

    embedding_key = f'X_{basis}'
    embedding = adata.obsm[embedding_key]
    n_cells = embedding.shape[0]

    genes = get_genes_used(adata)
    spliced_key = adata.uns.get('scHopfield', {}).get('spliced_key', 'Ms')

    if spliced_key in adata.layers:
        X = adata.layers[spliced_key][:, genes]
    else:
        X = adata.X[:, genes]
    if issparse(X):
        X = X.toarray()

    # Build KNN in gene space
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    # Project velocity to embedding
    embedding_velocity = np.zeros((n_cells, 2))

    for i in range(n_cells):
        neighbors = indices[i, 1:]  # Exclude self

        # Direction to neighbors in gene space
        dX = X[neighbors] - X[i]

        # Compute alignment of velocity with direction to each neighbor
        # (positive = moving towards neighbor)
        alignment = (velocity[i] * dX).sum(axis=1)

        # Normalize by distance (give more weight to closer neighbors)
        dists = distances[i, 1:]
        weights = np.exp(-dists / (np.median(dists) + 1e-10))
        weights = weights * np.maximum(alignment, 0)  # Only positive alignment
        weights = weights / (weights.sum() + 1e-10)

        # Direction to neighbors in embedding space
        dE = embedding[neighbors] - embedding[i]

        # Weighted average direction
        embedding_velocity[i] = (weights[:, None] * dE).sum(axis=0)

    return embedding_velocity


def compute_hopfield_velocity_at_state(
    adata: AnnData,
    X: np.ndarray,
    cluster_key: str = 'cell_type',
    use_cluster_specific_GRN: bool = True
) -> np.ndarray:
    """
    Compute Hopfield velocity at a given expression state for all cells.

    Parameters
    ----------
    adata : AnnData
        Annotated data with fitted interactions
    X : np.ndarray
        Expression matrix (n_cells, n_genes) to compute velocity at
    cluster_key : str, optional (default: 'cell_type')
        Key for cluster labels
    use_cluster_specific_GRN : bool, optional (default: True)
        Whether to use cluster-specific W matrices

    Returns
    -------
    np.ndarray
        Velocity matrix (n_cells, n_genes)
    """
    genes = get_genes_used(adata)
    n_cells = adata.n_obs
    n_genes = genes.sum()
    velocity = np.zeros((n_cells, n_genes))

    if use_cluster_specific_GRN:
        clusters = adata.obs[cluster_key].unique()

        for cluster in clusters:
            mask = adata.obs[cluster_key] == cluster
            cell_indices = np.where(mask)[0]

            v = compute_hopfield_velocity(adata, cluster, X=X[mask])
            velocity[cell_indices] = v
    else:
        cluster = 'all'
        if f'W_{cluster}' not in adata.varp:
            cluster = adata.obs[cluster_key].unique()[0]
        velocity = compute_hopfield_velocity(adata, cluster, X=X)

    return velocity


def calculate_perturbed_velocity_flow(
    adata: AnnData,
    basis: str = 'umap',
    cluster_key: str = 'cell_type',
    use_cluster_specific_GRN: bool = True,
    n_neighbors: int = 30,
    n_jobs: int = 4,
    verbose: bool = True
) -> np.ndarray:
    """
    Calculate flow from the absolute velocity at the perturbed state.

    Unlike delta velocity (perturbed - original), this computes the actual
    velocity field at the perturbed expression state, showing where
    perturbed cells would "go" according to Hopfield dynamics.

    v_perturbed = W @ sigmoid(X_perturbed) - gamma * X_perturbed + I

    Parameters
    ----------
    adata : AnnData
        Annotated data with perturbation results (simulated_X layer)
    basis : str, optional (default: 'umap')
        Embedding basis for visualization
    cluster_key : str, optional (default: 'cell_type')
        Key for cluster labels
    use_cluster_specific_GRN : bool, optional (default: True)
        Whether to use cluster-specific W matrices
    n_neighbors : int, optional (default: 30)
        Number of neighbors for embedding projection
    n_jobs : int, optional (default: 4)
        Number of parallel jobs
    verbose : bool, optional (default: True)
        Print progress

    Returns
    -------
    np.ndarray
        Perturbed velocity flow in embedding space (n_cells, 2)
    """
    if 'simulated_X' not in adata.layers:
        raise ValueError("simulated_X not found. Run simulate_shift first.")

    genes = get_genes_used(adata)

    # Get perturbed expression
    X_perturbed = adata.layers['simulated_X'][:, genes]
    if issparse(X_perturbed):
        X_perturbed = X_perturbed.toarray()

    if verbose:
        print("Computing Hopfield velocity at perturbed state...")

    # Compute velocity at perturbed state
    velocity_perturbed = compute_hopfield_velocity_at_state(
        adata, X_perturbed,
        cluster_key=cluster_key,
        use_cluster_specific_GRN=use_cluster_specific_GRN
    )

    if verbose:
        print("Projecting to embedding space...")

    # Project to embedding
    embedding_flow = project_velocity_to_embedding(
        adata, velocity_perturbed, basis=basis,
        n_neighbors=n_neighbors, n_jobs=n_jobs
    )

    # Store results
    adata.obsm[f'perturbed_velocity_flow_{basis}'] = embedding_flow
    adata.layers['velocity_perturbed_hopfield'] = velocity_perturbed

    if verbose:
        print(f"Perturbed velocity flow stored in adata.obsm['perturbed_velocity_flow_{basis}']")

    return embedding_flow


def calculate_original_velocity_flow(
    adata: AnnData,
    basis: str = 'umap',
    cluster_key: str = 'cell_type',
    use_cluster_specific_GRN: bool = True,
    spliced_key: str = 'Ms',
    n_neighbors: int = 30,
    n_jobs: int = 4,
    verbose: bool = True
) -> np.ndarray:
    """
    Calculate flow from the Hopfield velocity at the original (unperturbed) state.

    v_original = W @ sigmoid(X_original) - gamma * X_original + I

    Parameters
    ----------
    adata : AnnData
        Annotated data with fitted interactions
    basis : str, optional (default: 'umap')
        Embedding basis for visualization
    cluster_key : str, optional (default: 'cell_type')
        Key for cluster labels
    use_cluster_specific_GRN : bool, optional (default: True)
        Whether to use cluster-specific W matrices
    spliced_key : str, optional (default: 'Ms')
        Key for original expression data
    n_neighbors : int, optional (default: 30)
        Number of neighbors for embedding projection
    n_jobs : int, optional (default: 4)
        Number of parallel jobs
    verbose : bool, optional (default: True)
        Print progress

    Returns
    -------
    np.ndarray
        Original velocity flow in embedding space (n_cells, 2)
    """
    genes = get_genes_used(adata)

    # Get original expression
    if spliced_key in adata.layers:
        X_original = adata.layers[spliced_key][:, genes]
    else:
        X_original = adata.X[:, genes]
    if issparse(X_original):
        X_original = X_original.toarray()

    if verbose:
        print("Computing Hopfield velocity at original state...")

    # Compute velocity at original state
    velocity_original = compute_hopfield_velocity_at_state(
        adata, X_original,
        cluster_key=cluster_key,
        use_cluster_specific_GRN=use_cluster_specific_GRN
    )

    if verbose:
        print("Projecting to embedding space...")

    # Project to embedding
    embedding_flow = project_velocity_to_embedding(
        adata, velocity_original, basis=basis,
        n_neighbors=n_neighbors, n_jobs=n_jobs
    )

    # Store results
    adata.obsm[f'original_velocity_flow_{basis}'] = embedding_flow
    adata.layers['velocity_original_hopfield'] = velocity_original

    if verbose:
        print(f"Original velocity flow stored in adata.obsm['original_velocity_flow_{basis}']")

    return embedding_flow


def calculate_perturbation_flow_hopfield(
    adata: AnnData,
    basis: str = 'umap',
    cluster_key: str = 'cell_type',
    use_cluster_specific_GRN: bool = True,
    velocity_type: str = 'delta',
    n_neighbors: int = 30,
    n_jobs: int = 4,
    verbose: bool = True
) -> np.ndarray:
    """
    Calculate perturbation flow using Hopfield model dynamics directly.

    Supports three velocity types:
    - 'delta': Velocity difference (v_perturbed - v_original)
    - 'perturbed': Absolute velocity at perturbed state
    - 'original': Absolute velocity at original state (for comparison)

    Parameters
    ----------
    adata : AnnData
        Annotated data with perturbation results (simulated_X layer)
    basis : str, optional (default: 'umap')
        Embedding basis for visualization
    cluster_key : str, optional (default: 'cell_type')
        Key for cluster labels
    use_cluster_specific_GRN : bool, optional (default: True)
        Whether to use cluster-specific W matrices
    velocity_type : str, optional (default: 'delta')
        Type of velocity to compute:
        - 'delta': v(X_perturbed) - v(X_original)
        - 'perturbed': v(X_perturbed) only
        - 'original': v(X_original) only
    n_neighbors : int, optional (default: 30)
        Number of neighbors for embedding projection
    n_jobs : int, optional (default: 4)
        Number of parallel jobs
    verbose : bool, optional (default: True)
        Print progress

    Returns
    -------
    np.ndarray
        Perturbation flow in embedding space (n_cells, 2)
    """
    if velocity_type == 'perturbed':
        return calculate_perturbed_velocity_flow(
            adata, basis=basis, cluster_key=cluster_key,
            use_cluster_specific_GRN=use_cluster_specific_GRN,
            n_neighbors=n_neighbors, n_jobs=n_jobs, verbose=verbose
        )
    elif velocity_type == 'original':
        return calculate_original_velocity_flow(
            adata, basis=basis, cluster_key=cluster_key,
            use_cluster_specific_GRN=use_cluster_specific_GRN,
            n_neighbors=n_neighbors, n_jobs=n_jobs, verbose=verbose
        )

    # Default: delta velocity
    if verbose:
        print("Computing Hopfield velocity difference...")

    # Compute delta velocity in gene space
    delta_velocity = compute_hopfield_velocity_delta(
        adata,
        cluster_key=cluster_key,
        use_cluster_specific_GRN=use_cluster_specific_GRN
    )

    if verbose:
        print("Projecting to embedding space...")

    # Project to embedding
    embedding_flow = project_velocity_to_embedding(
        adata, delta_velocity, basis=basis,
        n_neighbors=n_neighbors, n_jobs=n_jobs
    )

    # Store results
    adata.obsm[f'perturbation_flow_{basis}'] = embedding_flow
    adata.layers['delta_velocity_hopfield'] = delta_velocity

    adata.uns['perturbation_flow_params'] = {
        'basis': basis,
        'method': 'hopfield',
        'velocity_type': velocity_type,
        'cluster_key': cluster_key,
        'use_cluster_specific_GRN': use_cluster_specific_GRN,
        'n_neighbors': n_neighbors
    }

    if verbose:
        print(f"Perturbation flow (Hopfield) stored in adata.obsm['perturbation_flow_{basis}']")

    return embedding_flow


def plot_perturbed_velocity_flow(
    adata: AnnData,
    basis: str = 'umap',
    ax: Optional[plt.Axes] = None,
    scale: float = 1.0,
    color: str = '#27AE60',
    alpha: float = 0.8,
    show_background: bool = True,
    cluster_key: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    s: float = 10,
    figsize: Tuple[float, float] = (8, 8),
    title: str = 'Velocity at Perturbed State',
    **quiver_kwargs
) -> plt.Axes:
    """
    Plot the absolute velocity at the perturbed state.

    Parameters
    ----------
    adata : AnnData
        Annotated data with perturbed velocity flow
    basis : str, optional (default: 'umap')
        Embedding basis
    ax : plt.Axes, optional
        Axes to plot on
    scale : float, optional (default: 1.0)
        Arrow scale
    color : str, optional (default: '#27AE60')
        Arrow color (green by default)
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
    flow_key = f'perturbed_velocity_flow_{basis}'

    if flow_key not in adata.obsm:
        raise ValueError(f"Perturbed velocity flow not found. "
                        "Run calculate_perturbed_velocity_flow first.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    embedding = adata.obsm[f'X_{basis}']
    flow = adata.obsm[flow_key]

    # Background
    if show_background:
        if cluster_key is not None and colors is not None:
            c = [colors.get(cl, 'lightgray') for cl in adata.obs[cluster_key]]
        else:
            c = 'lightgray'
        ax.scatter(embedding[:, 0], embedding[:, 1], c=c,
                  s=s, alpha=0.5, rasterized=True)

    # Quiver
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


def visualize_velocity_comparison(
    adata: AnnData,
    basis: str = 'umap',
    cluster_key: str = 'cell_type',
    colors: Optional[Dict[str, str]] = None,
    use_cluster_specific_GRN: bool = True,
    n_neighbors: int = 30,
    scale: float = 1.0,
    figsize: Tuple[float, float] = (20, 6),
    n_jobs: int = 4
) -> plt.Figure:
    """
    Visualize comparison of original, perturbed, and delta velocities.

    Creates a 1x4 figure showing:
    1. Clusters
    2. Original Hopfield velocity
    3. Perturbed Hopfield velocity
    4. Delta velocity (perturbed - original)

    Parameters
    ----------
    adata : AnnData
        Annotated data with perturbation results
    basis : str, optional (default: 'umap')
        Embedding basis
    cluster_key : str, optional (default: 'cell_type')
        Cluster key
    colors : dict, optional
        Cluster colors
    use_cluster_specific_GRN : bool, optional (default: True)
        Use cluster-specific GRNs
    n_neighbors : int, optional (default: 30)
        Neighbors for projection
    scale : float, optional (default: 1.0)
        Arrow scale
    figsize : tuple, optional
        Figure size
    n_jobs : int, optional (default: 4)
        Parallel jobs

    Returns
    -------
    plt.Figure
    """
    # Calculate all three velocity types
    print("Calculating original velocity...")
    calculate_original_velocity_flow(
        adata, basis=basis, cluster_key=cluster_key,
        use_cluster_specific_GRN=use_cluster_specific_GRN,
        n_neighbors=n_neighbors, n_jobs=n_jobs, verbose=False
    )

    print("Calculating perturbed velocity...")
    calculate_perturbed_velocity_flow(
        adata, basis=basis, cluster_key=cluster_key,
        use_cluster_specific_GRN=use_cluster_specific_GRN,
        n_neighbors=n_neighbors, n_jobs=n_jobs, verbose=False
    )

    print("Calculating delta velocity...")
    calculate_perturbation_flow_hopfield(
        adata, basis=basis, cluster_key=cluster_key,
        use_cluster_specific_GRN=use_cluster_specific_GRN,
        velocity_type='delta',
        n_neighbors=n_neighbors, n_jobs=n_jobs, verbose=False
    )

    # Get perturbation info
    perturb_str = "Perturbation"
    if 'scHopfield' in adata.uns and 'perturb_condition' in adata.uns['scHopfield']:
        perturb = adata.uns['scHopfield']['perturb_condition']
        perturb_str = ', '.join([f"{k}={'KO' if v==0 else v}" for k, v in perturb.items()])

    fig, axes = plt.subplots(1, 4, figsize=figsize)
    embedding = adata.obsm[f'X_{basis}']

    # Panel 1: Clusters
    ax = axes[0]
    if colors is not None:
        c = [colors.get(cl, 'gray') for cl in adata.obs[cluster_key]]
    else:
        c = adata.obs[cluster_key].astype('category').cat.codes
    ax.scatter(embedding[:, 0], embedding[:, 1], c=c, s=10, alpha=0.7)
    ax.set_title('Clusters', fontsize=12, fontweight='bold')
    ax.axis('off')

    # Panel 2: Original velocity
    ax = axes[1]
    flow_orig = adata.obsm[f'original_velocity_flow_{basis}']
    if colors is not None:
        c = [colors.get(cl, 'lightgray') for cl in adata.obs[cluster_key]]
    else:
        c = 'lightgray'
    ax.scatter(embedding[:, 0], embedding[:, 1], c=c, s=10, alpha=0.5)
    ax.quiver(embedding[:, 0], embedding[:, 1], flow_orig[:, 0], flow_orig[:, 1],
             color='#3498DB', alpha=0.8, scale=scale,
             headaxislength=4, headlength=5, headwidth=4)
    ax.set_title('Original Hopfield Velocity', fontsize=12, fontweight='bold')
    ax.axis('off')

    # Panel 3: Perturbed velocity
    ax = axes[2]
    flow_pert = adata.obsm[f'perturbed_velocity_flow_{basis}']
    ax.scatter(embedding[:, 0], embedding[:, 1], c=c, s=10, alpha=0.5)
    ax.quiver(embedding[:, 0], embedding[:, 1], flow_pert[:, 0], flow_pert[:, 1],
             color='#27AE60', alpha=0.8, scale=scale,
             headaxislength=4, headlength=5, headwidth=4)
    ax.set_title(f'Perturbed Velocity\n({perturb_str})', fontsize=12, fontweight='bold')
    ax.axis('off')

    # Panel 4: Delta velocity
    ax = axes[3]
    flow_delta = adata.obsm[f'perturbation_flow_{basis}']
    ax.scatter(embedding[:, 0], embedding[:, 1], c=c, s=10, alpha=0.5)
    ax.quiver(embedding[:, 0], embedding[:, 1], flow_delta[:, 0], flow_delta[:, 1],
             color='#E74C3C', alpha=0.8, scale=scale,
             headaxislength=4, headlength=5, headwidth=4)
    ax.set_title('Delta Velocity\n(Perturbed - Original)', fontsize=12, fontweight='bold')
    ax.axis('off')

    fig.suptitle(f'Hopfield Velocity Analysis: {perturb_str}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def calculate_perturbation_flow(
    adata: AnnData,
    basis: str = 'umap',
    method: str = 'celloracle',
    n_neighbors: int = 200,
    sigma_corr: float = 0.05,
    correlation_mode: str = 'sampled',
    sampled_fraction: float = 0.3,
    sampling_probs: Tuple[float, float] = (0.5, 0.1),
    cluster_key: str = 'cell_type',
    use_cluster_specific_GRN: bool = True,
    random_seed: int = 42,
    n_jobs: int = 4,
    verbose: bool = True
) -> np.ndarray:
    """
    Calculate perturbation-induced flow in embedding space.

    Two methods are available:
    1. 'celloracle': Correlation-based projection (default, like CellOracle)
    2. 'hopfield': Direct computation using Hopfield model dynamics

    The CellOracle method:
    1. Build KNN graph in embedding space
    2. Compute correlation between delta_X[i] and (X[j] - X[i]) for neighbors
    3. Convert correlations to transition probabilities: exp(corr / sigma_corr)
    4. Compute embedding shift as weighted sum of unit vectors to neighbors

    The Hopfield method:
    1. Compute velocity at original state: v_orig = W @ sigmoid(X) - gamma*X + I
    2. Compute velocity at perturbed state: v_pert = W @ sigmoid(X') - gamma*X' + I
    3. Delta velocity = v_pert - v_orig
    4. Project delta velocity to embedding space

    Parameters
    ----------
    adata : AnnData
        Annotated data with delta_X layer from perturbation simulation
    basis : str, optional (default: 'umap')
        Embedding basis to project onto
    method : str, optional (default: 'celloracle')
        Method for computing flow:
        - 'celloracle': Correlation-based projection (like CellOracle/scVelo)
        - 'hopfield': Direct Hopfield model velocity computation
    n_neighbors : int, optional (default: 200)
        Number of neighbors for KNN graph
    sigma_corr : float, optional (default: 0.05)
        Kernel scaling for transition probability (CellOracle method only)
    correlation_mode : str, optional (default: 'sampled')
        How to compute correlations (CellOracle method only):
        - 'sampled': Sample a fraction of neighbors for faster computation
        - 'full': Compute full correlation matrix for all cell pairs
    sampled_fraction : float, optional (default: 0.3)
        Fraction of neighbors to sample (CellOracle method, sampled mode)
    sampling_probs : tuple, optional (default: (0.5, 0.1))
        Probability gradient for neighbor sampling (CellOracle method)
    cluster_key : str, optional (default: 'cell_type')
        Key for cluster labels (Hopfield method)
    use_cluster_specific_GRN : bool, optional (default: True)
        Use cluster-specific GRNs (Hopfield method)
    random_seed : int, optional (default: 42)
        Random seed for reproducibility
    n_jobs : int, optional (default: 4)
        Number of parallel jobs
    verbose : bool, optional (default: True)
        Show progress

    Returns
    -------
    np.ndarray
        Perturbation flow vectors in embedding space (n_cells, 2)

    References
    ----------
    CellOracle method inspired by:
    Kamimoto et al. (2023). Nature. https://doi.org/10.1038/s41586-022-05688-9
    """
    if method == 'hopfield':
        return calculate_perturbation_flow_hopfield(
            adata, basis=basis,
            cluster_key=cluster_key,
            use_cluster_specific_GRN=use_cluster_specific_GRN,
            n_neighbors=min(n_neighbors, 50),  # Fewer neighbors for Hopfield
            n_jobs=n_jobs,
            verbose=verbose
        )
    elif method != 'celloracle':
        raise ValueError(f"Unknown method: {method}. Use 'celloracle' or 'hopfield'.")
    from sklearn.neighbors import NearestNeighbors
    from scipy import sparse

    np.random.seed(random_seed)

    if 'delta_X' not in adata.layers:
        raise ValueError("No delta_X found. Run simulate_perturbation first.")

    embedding_key = f'X_{basis}'
    if embedding_key not in adata.obsm:
        raise ValueError(f"Embedding {embedding_key} not found in adata.obsm")

    embedding = adata.obsm[embedding_key]
    n_cells = embedding.shape[0]

    # Get delta_X for genes used in analysis
    genes = get_genes_used(adata)
    delta_X = adata.layers['delta_X'][:, genes]

    # Get base expression from spliced layer
    spliced_key = adata.uns.get('scHopfield', {}).get('spliced_key', 'Ms')
    if spliced_key in adata.layers:
        X = adata.layers[spliced_key][:, genes]
    else:
        X = adata.X[:, genes]
    if issparse(X):
        X = X.toarray()

    # Build KNN graph in embedding space
    if verbose:
        print("Building KNN graph in embedding space...")
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs)
    nn.fit(embedding)
    embedding_knn = nn.kneighbors_graph(mode="connectivity")

    if correlation_mode == 'sampled':
        # Sample neighbors for speedup
        neigh_ixs = embedding_knn.indices.reshape((-1, n_neighbors + 1))

        # Probability gradient for sampling (favor closer neighbors)
        p = np.linspace(sampling_probs[0], sampling_probs[1], neigh_ixs.shape[1])
        p = p / p.sum()

        n_sampled = int(sampled_fraction * (n_neighbors + 1))
        sampling_ixs = np.stack([
            np.random.choice(neigh_ixs.shape[1], size=n_sampled, replace=False, p=p)
            for _ in range(n_cells)
        ], axis=0)

        neigh_ixs = neigh_ixs[np.arange(n_cells)[:, None], sampling_ixs]

        # Calculate correlation for sampled neighbors only
        corrcoef = _calculate_neighbor_correlation_partial(X, delta_X, neigh_ixs, verbose=verbose)

        # Build sparse KNN matrix for sampled neighbors
        nonzero = n_cells * n_sampled
        embedding_knn_used = sparse.csr_matrix(
            (np.ones(nonzero), neigh_ixs.ravel(), np.arange(0, nonzero + 1, n_sampled)),
            shape=(n_cells, n_cells)
        )

    elif correlation_mode == 'full':
        # Compute full correlation matrix
        corrcoef = _calculate_correlation_full(X, delta_X, verbose=verbose)

        # Set diagonal to 0
        np.fill_diagonal(corrcoef, 0)

        # Use the full KNN graph
        embedding_knn_used = embedding_knn

    else:
        raise ValueError(f"Unknown correlation_mode: {correlation_mode}. Use 'sampled' or 'full'.")

    # Handle NaNs (can occur with identical cells)
    if np.any(np.isnan(corrcoef)):
        corrcoef[np.isnan(corrcoef)] = 1
        if verbose:
            print("Warning: NaNs in correlation matrix corrected to 1s.")

    # Calculate transition probabilities using exponential kernel
    # transition_prob = exp(corrcoef / sigma_corr) * knn_mask
    knn_array = embedding_knn_used.toarray()
    transition_prob = np.exp(corrcoef / sigma_corr) * knn_array
    transition_prob /= transition_prob.sum(axis=1, keepdims=True) + 1e-10

    # Calculate embedding shift (delta_embedding)
    delta_embedding = _calculate_embedding_shift(embedding, transition_prob, knn_array)

    # Store in adata
    adata.obsm[f'perturbation_flow_{basis}'] = delta_embedding
    adata.uns['perturbation_flow_params'] = {
        'basis': basis,
        'n_neighbors': n_neighbors,
        'sigma_corr': sigma_corr,
        'correlation_mode': correlation_mode,
        'sampled_fraction': sampled_fraction if correlation_mode == 'sampled' else None
    }

    if verbose:
        print(f"Perturbation flow calculated and stored in adata.obsm['perturbation_flow_{basis}']")

    return delta_embedding


def _calculate_embedding_shift(
    embedding: np.ndarray,
    transition_prob: np.ndarray,
    knn_array: np.ndarray
) -> np.ndarray:
    """
    Calculate embedding shift from transition probabilities.

    Follows CellOracle's calculate_embedding_shift logic:
    1. Compute unitary vectors from each cell to all others
    2. Weight by transition probabilities
    3. Subtract baseline (uniform distribution over neighbors)

    Parameters
    ----------
    embedding : np.ndarray
        Embedding coordinates (n_cells, 2)
    transition_prob : np.ndarray
        Transition probability matrix (n_cells, n_cells)
    knn_array : np.ndarray
        KNN connectivity matrix (n_cells, n_cells)

    Returns
    -------
    np.ndarray
        Embedding shift vectors (n_cells, 2)
    """
    # Unitary vectors from each cell to all other cells
    # Shape: (2, n_cells, n_cells) where [dim, from_cell, to_cell]
    unitary_vectors = embedding.T[:, None, :] - embedding.T[:, :, None]

    # Normalize to unit vectors
    with np.errstate(divide='ignore', invalid='ignore'):
        norms = np.linalg.norm(unitary_vectors, ord=2, axis=0)
        unitary_vectors = unitary_vectors / (norms + 1e-10)
        # Fix diagonal (self-to-self vectors)
        np.fill_diagonal(unitary_vectors[0], 0)
        np.fill_diagonal(unitary_vectors[1], 0)

    # Weighted sum of directions based on transition probabilities
    delta_embedding = (transition_prob * unitary_vectors).sum(axis=2)

    # Subtract baseline (expected shift under uniform distribution over neighbors)
    # This centers the vectors so that no shift means neutral
    knn_sum = knn_array.sum(axis=1, keepdims=True)
    baseline = (knn_array * unitary_vectors).sum(axis=2) / (knn_sum.T + 1e-10)
    delta_embedding = delta_embedding - baseline

    # Transpose to (n_cells, 2)
    delta_embedding = delta_embedding.T

    return delta_embedding


def _calculate_neighbor_correlation_partial(
    X: np.ndarray,
    delta_X: np.ndarray,
    neigh_ixs: np.ndarray,
    verbose: bool = True
) -> np.ndarray:
    """
    Calculate correlation between delta_X and neighbor expression differences.

    For each cell i and its sampled neighbors j in neigh_ixs[i], computes:
    corr(delta_X[i], X[j] - X[i])

    This is the 'partial' correlation computation used when sampling neighbors.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix (n_cells, n_genes)
    delta_X : np.ndarray
        Delta expression matrix (n_cells, n_genes)
    neigh_ixs : np.ndarray
        Sampled neighbor indices (n_cells, n_sampled)
    verbose : bool, optional (default: True)
        Show progress bar

    Returns
    -------
    np.ndarray
        Correlation matrix (n_cells, n_cells) - sparse in practice
    """
    from tqdm.auto import tqdm

    n_cells, n_neighbors = neigh_ixs.shape
    n_genes = X.shape[1]
    corrcoef = np.zeros((n_cells, n_cells))

    iterator = range(n_cells)
    if verbose:
        iterator = tqdm(iterator, desc="Calculating correlations (sampled)")

    for i in iterator:
        # Get neighbors
        neighbors = neigh_ixs[i]

        # Compute X[neighbors] - X[i] for all neighbors at once
        diffs = X[neighbors] - X[i]  # (n_neighbors, n_genes)

        # Compute correlations
        corrs = _pearson_correlation_rows(delta_X[i:i+1], diffs)

        # Store in matrix
        for j_idx, j in enumerate(neighbors):
            if j != i:
                corrcoef[i, j] = corrs[j_idx]

    return corrcoef


def _calculate_correlation_full(
    X: np.ndarray,
    delta_X: np.ndarray,
    verbose: bool = True
) -> np.ndarray:
    """
    Calculate full correlation matrix between delta_X and expression differences.

    For each cell pair (i, j), computes:
    corr(delta_X[i], X[j] - X[i])

    This is computationally expensive O(n_cells^2) but more accurate.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix (n_cells, n_genes)
    delta_X : np.ndarray
        Delta expression matrix (n_cells, n_genes)
    verbose : bool, optional (default: True)
        Show progress bar

    Returns
    -------
    np.ndarray
        Full correlation matrix (n_cells, n_cells)
    """
    from tqdm.auto import tqdm

    n_cells = X.shape[0]
    corrcoef = np.zeros((n_cells, n_cells))

    iterator = range(n_cells)
    if verbose:
        iterator = tqdm(iterator, desc="Calculating correlations (full)")

    for i in iterator:
        # Compute X[j] - X[i] for all j
        diffs = X - X[i]  # (n_cells, n_genes)

        # Compute correlations with delta_X[i]
        corrs = _pearson_correlation_rows(delta_X[i:i+1], diffs)

        corrcoef[i, :] = corrs

    return corrcoef


def _pearson_correlation_rows(
    a: np.ndarray,
    B: np.ndarray
) -> np.ndarray:
    """
    Compute Pearson correlation between vector a and each row of matrix B.

    Uses the same formula as CellOracle's C implementation:
    corr = sum((a - mean_a) * (b - mean_b)) / (sqrt(ss_a) * sqrt(ss_b))

    where ss_a = sum((a - mean_a)^2) and ss_b = sum((b - mean_b)^2)

    Parameters
    ----------
    a : np.ndarray
        Single row vector (1, n_features)
    B : np.ndarray
        Matrix (n_rows, n_features)

    Returns
    -------
    np.ndarray
        Correlation values (n_rows,)
    """
    # Center (subtract mean)
    a_centered = a - a.mean()
    B_centered = B - B.mean(axis=1, keepdims=True)

    # Sum of squares (not divided by N, matching C code)
    ss_a = np.sum(a_centered ** 2)
    ss_B = np.sum(B_centered ** 2, axis=1)

    if ss_a < 1e-10:
        return np.zeros(B.shape[0])

    # Dot product (sum of element-wise products)
    numerator = (a_centered @ B_centered.T).flatten()

    # Correlation: dot / (sqrt(ss_a) * sqrt(ss_b))
    denominator = np.sqrt(ss_a) * np.sqrt(ss_B)

    # Handle zero variance
    with np.errstate(divide='ignore', invalid='ignore'):
        corrs = numerator / (denominator + 1e-10)
        corrs[ss_B < 1e-10] = 0

    return corrs


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


def calculate_grid_flow_knn(
    adata: AnnData,
    basis: str = 'umap',
    n_grid: int = 40,
    n_neighbors: int = 200,
    min_mass: float = 1.0,
    flow_key: Optional[str] = None,
    n_jobs: int = 4
) -> Dict:
    """
    Calculate flow vectors on a grid using KNN-based interpolation.

    This follows CellOracle's approach using KNN to calculate probability mass
    and interpolate flow vectors to grid points.

    Parameters
    ----------
    adata : AnnData
        Annotated data with flow vectors
    basis : str, optional (default: 'umap')
        Embedding basis
    n_grid : int, optional (default: 40)
        Number of grid points per dimension
    n_neighbors : int, optional (default: 200)
        Number of neighbors for mass calculation
    min_mass : float, optional (default: 1.0)
        Minimum probability mass to show arrows (as percentage)
    flow_key : str, optional
        Key in obsm for flow vectors. If None, uses 'perturbation_flow_{basis}'
    n_jobs : int, optional (default: 4)
        Number of parallel jobs for KNN

    Returns
    -------
    dict
        Dictionary with grid coordinates, flow vectors, mass, and mass filter

    References
    ----------
    Logic inspired by CellOracle's calculate_p_mass function.
    Kamimoto et al. (2023). Nature. https://doi.org/10.1038/s41586-022-05688-9
    """
    from sklearn.neighbors import NearestNeighbors

    embedding_key = f'X_{basis}'
    embedding = adata.obsm[embedding_key]
    n_cells = embedding.shape[0]

    if flow_key is None:
        flow_key = f'perturbation_flow_{basis}'

    if flow_key not in adata.obsm:
        raise ValueError(f"Flow {flow_key} not found. Run calculate_perturbation_flow first.")

    flow = adata.obsm[flow_key]

    # Create grid
    x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()

    gx = np.linspace(x_min, x_max, n_grid)
    gy = np.linspace(y_min, y_max, n_grid)

    grid_x, grid_y = np.meshgrid(gx, gy)
    grid_coords = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    n_grid_points = len(grid_coords)

    # Calculate probability mass using KNN
    # For each grid point, count how many cells have it as a neighbor
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, n_cells), n_jobs=n_jobs)
    nn.fit(embedding)

    # Get distances and indices from grid points to cells
    distances, indices = nn.kneighbors(grid_coords)

    # Calculate mass as sum of Gaussian-weighted contributions
    # Use median distance as bandwidth
    median_dist = np.median(distances)
    sigma = median_dist * 0.5

    # Gaussian weights
    weights = np.exp(-distances ** 2 / (2 * sigma ** 2))
    mass = weights.sum(axis=1)

    # Normalize mass
    mass = mass / mass.max()
    mass_filter = mass < (min_mass / 100)

    # Interpolate flow to grid using weighted average of neighbors
    grid_flow = np.zeros((n_grid_points, 2))

    for i in range(n_grid_points):
        w = weights[i]
        w = w / (w.sum() + 1e-10)
        grid_flow[i] = np.average(flow[indices[i]], axis=0, weights=w)

    # Store in adata.uns for convenience
    adata.uns[f'grid_flow_{basis}'] = {
        'grid_coords': grid_coords,
        'grid_flow': grid_flow,
        'mass_filter': mass_filter,
        'mass': mass,
        'n_grid': n_grid
    }

    return {
        'grid_coords': grid_coords,
        'grid_flow': grid_flow,
        'mass_filter': mass_filter,
        'mass': mass,
        'n_grid': n_grid
    }


def plot_simulation_flow_on_grid(
    adata: AnnData,
    basis: str = 'umap',
    n_grid: int = 40,
    n_neighbors: int = 200,
    min_mass: float = 1.0,
    scale: float = 1.0,
    ax: Optional[plt.Axes] = None,
    show_background: bool = True,
    cluster_key: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    arrow_color: str = 'black',
    s: float = 10,
    alpha: float = 0.8,
    figsize: Tuple[float, float] = (8, 8),
    title: Optional[str] = None,
    flow_key: Optional[str] = None,
    recalculate: bool = False,
    n_jobs: int = 4,
    **quiver_kwargs
) -> plt.Axes:
    """
    Plot perturbation flow on a grid.

    Convenience function that calculates grid flow (if needed) and plots it.
    This produces cleaner visualizations than plotting arrows on every cell.

    Parameters
    ----------
    adata : AnnData
        Annotated data with perturbation flow
    basis : str, optional (default: 'umap')
        Embedding basis
    n_grid : int, optional (default: 40)
        Number of grid points per dimension
    n_neighbors : int, optional (default: 200)
        Number of neighbors for mass calculation
    min_mass : float, optional (default: 1.0)
        Minimum probability mass to show arrows (as percentage)
    scale : float, optional (default: 1.0)
        Scale factor for arrows (passed to quiver)
    ax : plt.Axes, optional
        Axes to plot on. If None, creates new figure.
    show_background : bool, optional (default: True)
        Show background scatter of cells
    cluster_key : str, optional
        Key for cluster labels (for coloring background)
    colors : dict, optional
        Dictionary mapping cluster names to colors
    arrow_color : str, optional (default: 'black')
        Color for flow arrows
    s : float, optional (default: 10)
        Scatter point size for background
    alpha : float, optional (default: 0.8)
        Arrow transparency
    figsize : tuple, optional
        Figure size if creating new figure
    title : str, optional
        Plot title. If None, uses perturbation condition info.
    flow_key : str, optional
        Key in obsm for flow vectors. If None, uses 'perturbation_flow_{basis}'
    recalculate : bool, optional (default: False)
        If True, recalculate grid flow even if already stored
    n_jobs : int, optional (default: 4)
        Number of parallel jobs for KNN
    **quiver_kwargs
        Additional arguments for matplotlib quiver

    Returns
    -------
    plt.Axes

    References
    ----------
    Logic inspired by CellOracle's _plot_simulation_flow_on_grid function.
    Kamimoto et al. (2023). Nature. https://doi.org/10.1038/s41586-022-05688-9
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    embedding_key = f'X_{basis}'
    embedding = adata.obsm[embedding_key]

    if flow_key is None:
        flow_key = f'perturbation_flow_{basis}'

    # Check if grid flow already calculated
    grid_key = f'grid_flow_{basis}'
    if grid_key in adata.uns and not recalculate:
        grid_data = adata.uns[grid_key]
    else:
        grid_data = calculate_grid_flow_knn(
            adata, basis=basis, n_grid=n_grid, n_neighbors=n_neighbors,
            min_mass=min_mass, flow_key=flow_key, n_jobs=n_jobs
        )

    # Plot background
    if show_background:
        if cluster_key is not None and colors is not None:
            c = [colors.get(cl, 'lightgray') for cl in adata.obs[cluster_key]]
        else:
            c = 'lightgray'
        ax.scatter(embedding[:, 0], embedding[:, 1], c=c,
                  s=s, alpha=0.5, rasterized=True)

    # Get grid data
    grid_coords = grid_data['grid_coords']
    grid_flow = grid_data['grid_flow']
    mass_filter = grid_data['mass_filter']

    # Filter by mass
    valid = ~mass_filter

    # Default quiver settings
    default_quiver = dict(
        headaxislength=4,
        headlength=5,
        headwidth=4,
        linewidths=0.5,
        width=0.004
    )
    default_quiver.update(quiver_kwargs)

    # Plot arrows
    ax.quiver(
        grid_coords[valid, 0], grid_coords[valid, 1],
        grid_flow[valid, 0], grid_flow[valid, 1],
        color=arrow_color, alpha=alpha, scale=scale,
        **default_quiver
    )

    # Title
    if title is None:
        if 'scHopfield' in adata.uns and 'perturb_condition' in adata.uns['scHopfield']:
            perturb = adata.uns['scHopfield']['perturb_condition']
            perturb_str = ', '.join([f"{k}={'KO' if v==0 else v}" for k, v in perturb.items()])
            title = f'Perturbation Flow: {perturb_str}'
        else:
            title = 'Perturbation Flow (Grid)'

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    ax.set_aspect('equal')

    return ax


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

    References
    ----------
    Logic for the transition vector field is inspired by the perturbation
    simulation workflow in CellOracle:
    Kamimoto et al. (2023). Nature. https://doi.org/10.1038/s41586-022-05688-9
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

    References
    ----------
    Logic for the transition vector field is inspired by the perturbation
    simulation workflow in CellOracle:
    Kamimoto et al. (2023). Nature. https://doi.org/10.1038/s41586-022-05688-9
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

    References
    ----------
    Logic for the transition vector field is inspired by the perturbation
    simulation workflow in CellOracle:
    Kamimoto et al. (2023). Nature. https://doi.org/10.1038/s41586-022-05688-9
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

    References
    ----------
    Logic for the transition vector field is inspired by the perturbation
    simulation workflow in CellOracle:
    Kamimoto et al. (2023). Nature. https://doi.org/10.1038/s41586-022-05688-9
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


# =============================================================================
# ODE Trajectory Flow Functions
# =============================================================================

def calculate_ode_trajectory_flow(
    adata: AnnData,
    wt_trajectories: Dict[str, np.ndarray],
    perturbed_trajectories: Dict[str, np.ndarray],
    cluster_key: str = 'cell_type',
    basis: str = 'umap',
    time_point: int = -1,
    method: str = 'hopfield',
    n_neighbors: int = 30,
    use_cluster_specific_GRN: bool = True,
    n_jobs: int = 4,
    verbose: bool = True
) -> np.ndarray:
    """
    Calculate perturbation flow from ODE trajectory results.

    Takes the final (or specified) time point from ODE trajectories and
    computes the flow in embedding space.

    Parameters
    ----------
    adata : AnnData
        Annotated data with cell information
    wt_trajectories : dict
        Dictionary mapping cluster -> WT trajectory (n_time, n_genes)
    perturbed_trajectories : dict
        Dictionary mapping cluster -> perturbed trajectory (n_time, n_genes)
    cluster_key : str, optional (default: 'cell_type')
        Key for cluster labels
    basis : str, optional (default: 'umap')
        Embedding basis
    time_point : int, optional (default: -1)
        Which time point to use (-1 for final)
    method : str, optional (default: 'hopfield')
        Flow calculation method:
        - 'hopfield': Use Hopfield model velocity directly
        - 'celloracle': Use correlation-based projection
        - 'difference': Simple difference in gene space projected to embedding
    n_neighbors : int, optional (default: 30)
        Number of neighbors for projection
    use_cluster_specific_GRN : bool, optional (default: True)
        Use cluster-specific W matrices
    n_jobs : int, optional (default: 4)
        Number of parallel jobs
    verbose : bool, optional (default: True)
        Print progress

    Returns
    -------
    np.ndarray
        Perturbation flow in embedding space (n_cells, 2)
    """
    genes = get_genes_used(adata)
    n_cells = adata.n_obs
    n_genes = genes.sum()

    # Initialize arrays
    delta_X = np.zeros((n_cells, n_genes))
    X_wt_final = np.zeros((n_cells, n_genes))
    X_pert_final = np.zeros((n_cells, n_genes))

    # Get final states from trajectories for each cluster
    for cluster in wt_trajectories.keys():
        if cluster not in perturbed_trajectories:
            continue

        mask = adata.obs[cluster_key] == cluster
        if not mask.any():
            continue

        # Get final time point
        wt_final = wt_trajectories[cluster][time_point]
        pert_final = perturbed_trajectories[cluster][time_point]

        # Assign to all cells in this cluster (they share the same trajectory)
        cell_indices = np.where(mask)[0]
        for idx in cell_indices:
            X_wt_final[idx] = wt_final
            X_pert_final[idx] = pert_final
            delta_X[idx] = pert_final - wt_final

    # Store delta_X temporarily
    adata.layers['delta_X_ode'] = delta_X

    if method == 'hopfield':
        # Compute velocity difference using Hopfield model
        if verbose:
            print("Computing Hopfield velocities...")

        delta_velocity = np.zeros((n_cells, n_genes))

        for cluster in wt_trajectories.keys():
            mask = adata.obs[cluster_key] == cluster
            if not mask.any():
                continue

            cell_indices = np.where(mask)[0]

            # Compute velocity at WT and perturbed states
            v_wt = compute_hopfield_velocity(
                adata, cluster, X=X_wt_final[mask]
            )
            v_pert = compute_hopfield_velocity(
                adata, cluster, X=X_pert_final[mask]
            )

            delta_velocity[cell_indices] = v_pert - v_wt

        # Project to embedding
        if verbose:
            print("Projecting to embedding...")
        embedding_flow = project_velocity_to_embedding(
            adata, delta_velocity, basis=basis,
            n_neighbors=n_neighbors, n_jobs=n_jobs
        )

    elif method == 'difference':
        # Simple projection of expression difference
        if verbose:
            print("Projecting expression difference to embedding...")
        embedding_flow = project_velocity_to_embedding(
            adata, delta_X, basis=basis,
            n_neighbors=n_neighbors, n_jobs=n_jobs
        )

    else:  # celloracle
        # Use correlation-based projection
        # Temporarily store simulated_X
        spliced_key = adata.uns.get('scHopfield', {}).get('spliced_key', 'Ms')
        original_simulated = adata.layers.get('simulated_X', None)

        # Set simulated_X to perturbed final state
        adata.layers['simulated_X'] = X_pert_final
        adata.layers['delta_X'] = delta_X

        embedding_flow = calculate_perturbation_flow(
            adata, basis=basis, method='celloracle',
            n_neighbors=n_neighbors, n_jobs=n_jobs, verbose=verbose
        )

        # Restore
        if original_simulated is not None:
            adata.layers['simulated_X'] = original_simulated

    # Store results
    adata.obsm[f'ode_perturbation_flow_{basis}'] = embedding_flow
    adata.uns['ode_perturbation_flow_params'] = {
        'basis': basis,
        'method': method,
        'time_point': time_point,
        'clusters': list(wt_trajectories.keys())
    }

    if verbose:
        print(f"ODE perturbation flow stored in adata.obsm['ode_perturbation_flow_{basis}']")

    return embedding_flow


def calculate_ode_trajectory_inner_product(
    adata: AnnData,
    basis: str = 'umap',
    velocity_key: Optional[str] = None,
    flow_key: Optional[str] = None
) -> np.ndarray:
    """
    Calculate inner product between reference velocity and ODE perturbation flow.

    Parameters
    ----------
    adata : AnnData
        Annotated data with ODE perturbation flow
    basis : str, optional (default: 'umap')
        Embedding basis
    velocity_key : str, optional
        Key for reference velocity
    flow_key : str, optional
        Key for ODE perturbation flow

    Returns
    -------
    np.ndarray
        Inner product values
    """
    if velocity_key is None:
        velocity_key = f'velocity_{basis}'

    if flow_key is None:
        flow_key = f'ode_perturbation_flow_{basis}'

    if velocity_key not in adata.obsm:
        raise ValueError(f"Velocity {velocity_key} not found")

    if flow_key not in adata.obsm:
        raise ValueError(f"ODE flow {flow_key} not found. Run calculate_ode_trajectory_flow first.")

    ref_velocity = adata.obsm[velocity_key]
    ode_flow = adata.obsm[flow_key]

    # Normalize and compute inner product
    ref_norm = np.linalg.norm(ref_velocity, axis=1, keepdims=True) + 1e-10
    ode_norm = np.linalg.norm(ode_flow, axis=1, keepdims=True) + 1e-10

    ref_unit = ref_velocity / ref_norm
    ode_unit = ode_flow / ode_norm

    inner_product = np.sum(ref_unit * ode_unit, axis=1)

    adata.obs['ode_perturbation_inner_product'] = inner_product

    return inner_product


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
        Arrow color
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
        raise ValueError(f"ODE flow {flow_key} not found. Run calculate_ode_trajectory_flow first.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    embedding_key = f'X_{basis}'
    embedding = adata.obsm[embedding_key]
    flow = adata.obsm[flow_key]

    # Background
    if show_background:
        if cluster_key is not None and colors is not None:
            c = [colors.get(cl, 'lightgray') for cl in adata.obs[cluster_key]]
        else:
            c = 'lightgray'
        ax.scatter(embedding[:, 0], embedding[:, 1], c=c,
                  s=s, alpha=0.5, rasterized=True)

    # Quiver
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
    # Calculate flow
    calculate_ode_trajectory_flow(
        adata, wt_trajectories, perturbed_trajectories,
        cluster_key=cluster_key, basis=basis, method=method
    )

    # Calculate inner product
    if velocity_key is None:
        velocity_key = f'velocity_{basis}'

    if velocity_key in adata.obsm:
        calculate_ode_trajectory_inner_product(
            adata, basis=basis, velocity_key=velocity_key
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
            norm = plt.matplotlib.colors.TwoSlopeNorm(vmin=-vm, vcenter=0, vmax=vm)
        except:
            norm = plt.matplotlib.colors.Normalize(vmin=-vm, vmax=vm)
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

    # Plot trajectory for first cluster
    cluster = list(wt_trajectories.keys())[0]
    wt = wt_trajectories[cluster]
    pert = perturbed_trajectories[cluster]

    # Find most changed gene
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
