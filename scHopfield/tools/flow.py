"""Flow computation and analysis.

This module contains functions for computing perturbation flow and related
analyses. These are non-plotting computation functions that were previously
mixed into plotting/flow.py.

References
----------
Logic for the transition vector field is inspired by the perturbation
simulation workflow in CellOracle:
Kamimoto et al. (2023). Nature. https://doi.org/10.1038/s41586-022-05688-9
"""

from typing import Optional, Dict, Tuple
import numpy as np
from anndata import AnnData
from scipy.sparse import issparse

from .velocity import compute_velocity, compute_velocity_delta
from .embedding import project_to_embedding
from .._utils.io import get_genes_used, get_matrix, to_numpy


def calculate_flow(
    adata: AnnData,
    source: str = 'delta',
    basis: str = 'umap',
    method: str = 'hopfield',
    cluster_key: str = 'cell_type',
    use_cluster_specific: bool = True,
    n_neighbors: int = 30,
    n_jobs: int = 4,
    store_key: Optional[str] = None,
    verbose: bool = True,
    # CellOracle method kwargs
    sigma_corr: float = 0.05,
    correlation_mode: str = 'sampled',
    sampled_fraction: float = 0.3,
    sampling_probs: Tuple[float, float] = (0.5, 0.1),
    random_seed: int = 42,
    # Layer keys
    perturbed_key: str = 'simulated_count',
    original_key: str = 'Ms',
) -> np.ndarray:
    """
    Unified flow calculation function.

    Supports two methods for computing perturbation flow:
    1. 'hopfield': Direct velocity computation using Hopfield model dynamics
    2. 'celloracle': Correlation-based projection (like CellOracle/scVelo)

    Parameters
    ----------
    adata : AnnData
        Annotated data with perturbation results (delta_X or simulated_count layer)
    source : str, optional (default: 'delta')
        What expression state to compute flow from:
        - 'delta': v(perturbed) - v(original)
        - 'perturbed': v(perturbed) absolute
        - 'original': v(original) absolute
        - custom layer key: uses that layer for expression
    basis : str, optional (default: 'umap')
        Embedding basis for visualization
    method : str, optional (default: 'hopfield')
        Method for computing flow:
        - 'hopfield': Direct Hopfield model velocity computation
        - 'celloracle': Correlation-based projection
    cluster_key : str, optional (default: 'cell_type')
        Key for cluster labels
    use_cluster_specific : bool, optional (default: True)
        Use cluster-specific W matrices
    n_neighbors : int, optional (default: 30)
        Number of neighbors for embedding projection
    n_jobs : int, optional (default: 4)
        Number of parallel jobs
    store_key : str, optional
        If provided, stores result in adata.obsm[store_key].
        If None, uses default key based on source and basis.
    verbose : bool, optional (default: True)
        Print progress
    sigma_corr : float, optional (default: 0.05)
        Kernel scaling for transition probability (CellOracle method only)
    correlation_mode : str, optional (default: 'sampled')
        How to compute correlations (CellOracle method only):
        - 'sampled': Sample a fraction of neighbors
        - 'full': Full correlation matrix
    sampled_fraction : float, optional (default: 0.3)
        Fraction of neighbors to sample (CellOracle sampled mode)
    sampling_probs : tuple, optional (default: (0.5, 0.1))
        Probability gradient for neighbor sampling
    random_seed : int, optional (default: 42)
        Random seed for reproducibility
    perturbed_key : str, optional (default: 'simulated_count')
        Key for perturbed expression data
    original_key : str, optional (default: 'Ms')
        Key for original expression data

    Returns
    -------
    np.ndarray
        Flow vectors in embedding space (n_cells, 2)
    """
    if method == 'hopfield':
        return _calculate_flow_hopfield(
            adata, source=source, basis=basis,
            cluster_key=cluster_key, use_cluster_specific=use_cluster_specific,
            n_neighbors=n_neighbors, n_jobs=n_jobs, store_key=store_key,
            verbose=verbose, perturbed_key=perturbed_key, original_key=original_key
        )
    elif method == 'celloracle':
        return _calculate_flow_celloracle(
            adata, basis=basis, n_neighbors=n_neighbors,
            sigma_corr=sigma_corr, correlation_mode=correlation_mode,
            sampled_fraction=sampled_fraction, sampling_probs=sampling_probs,
            random_seed=random_seed, n_jobs=n_jobs, store_key=store_key,
            verbose=verbose
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'hopfield' or 'celloracle'.")


def _calculate_flow_hopfield(
    adata: AnnData,
    source: str = 'delta',
    basis: str = 'umap',
    cluster_key: str = 'cell_type',
    use_cluster_specific: bool = True,
    n_neighbors: int = 30,
    n_jobs: int = 4,
    store_key: Optional[str] = None,
    verbose: bool = True,
    perturbed_key: str = 'simulated_count',
    original_key: str = 'Ms',
) -> np.ndarray:
    """
    Calculate flow using Hopfield model dynamics.

    Internal function called by calculate_flow with method='hopfield'.
    """
    genes = get_genes_used(adata)

    if source == 'delta':
        if verbose:
            print("Computing Hopfield velocity difference...")
        velocity = compute_velocity_delta(
            adata, perturbed_key=perturbed_key, original_key=original_key,
            cluster_key=cluster_key, use_cluster_specific=use_cluster_specific
        )
        default_key = f'perturbation_flow_{basis}'

    elif source == 'perturbed':
        if perturbed_key not in adata.layers:
            raise ValueError(f"'{perturbed_key}' not found. Run simulation first.")

        if verbose:
            print("Computing Hopfield velocity at perturbed state...")

        X_perturbed = adata.layers[perturbed_key][:, genes]
        if issparse(X_perturbed):
            X_perturbed = X_perturbed.toarray()

        velocity = compute_velocity(
            adata, X=X_perturbed, cluster_key=cluster_key,
            use_cluster_specific=use_cluster_specific
        )
        default_key = f'perturbed_velocity_flow_{basis}'

    elif source == 'original':
        if verbose:
            print("Computing Hopfield velocity at original state...")

        X_original = get_matrix(adata, original_key, genes=genes)
        X_original = to_numpy(X_original)

        velocity = compute_velocity(
            adata, X=X_original, cluster_key=cluster_key,
            use_cluster_specific=use_cluster_specific
        )
        default_key = f'original_velocity_flow_{basis}'

    else:
        # Treat source as a layer key
        if source not in adata.layers:
            raise ValueError(f"Layer '{source}' not found in adata.layers")

        if verbose:
            print(f"Computing Hopfield velocity from layer '{source}'...")

        X_custom = adata.layers[source][:, genes]
        if issparse(X_custom):
            X_custom = X_custom.toarray()

        velocity = compute_velocity(
            adata, X=X_custom, cluster_key=cluster_key,
            use_cluster_specific=use_cluster_specific
        )
        default_key = f'{source}_flow_{basis}'

    if verbose:
        print("Projecting to embedding space...")

    # Project to embedding
    embedding_flow = project_to_embedding(
        adata, velocity, basis=basis, n_neighbors=n_neighbors, n_jobs=n_jobs
    )

    # Store results
    flow_key = store_key if store_key is not None else default_key
    adata.obsm[flow_key] = embedding_flow

    # Store parameters
    adata.uns['perturbation_flow_params'] = {
        'basis': basis,
        'method': 'hopfield',
        'source': source,
        'cluster_key': cluster_key,
        'use_cluster_specific': use_cluster_specific,
        'n_neighbors': n_neighbors
    }

    if verbose:
        print(f"Flow stored in adata.obsm['{flow_key}']")

    return embedding_flow


def _calculate_flow_celloracle(
    adata: AnnData,
    basis: str = 'umap',
    n_neighbors: int = 200,
    sigma_corr: float = 0.05,
    correlation_mode: str = 'sampled',
    sampled_fraction: float = 0.3,
    sampling_probs: Tuple[float, float] = (0.5, 0.1),
    random_seed: int = 42,
    n_jobs: int = 4,
    store_key: Optional[str] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Calculate flow using CellOracle-style correlation-based projection.

    Internal function called by calculate_flow with method='celloracle'.
    """
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
    if issparse(delta_X):
        delta_X = delta_X.toarray()

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
        corrcoef = _calculate_correlation_sampled(X, delta_X, neigh_ixs, verbose=verbose)

        # Build sparse KNN matrix for sampled neighbors
        nonzero = n_cells * n_sampled
        embedding_knn_used = sparse.csr_matrix(
            (np.ones(nonzero), neigh_ixs.ravel(), np.arange(0, nonzero + 1, n_sampled)),
            shape=(n_cells, n_cells)
        )

    elif correlation_mode == 'full':
        # Compute full correlation matrix
        corrcoef = _calculate_correlation_full(X, delta_X, verbose=verbose)
        np.fill_diagonal(corrcoef, 0)
        embedding_knn_used = embedding_knn

    else:
        raise ValueError(f"Unknown correlation_mode: {correlation_mode}")

    # Handle NaNs
    if np.any(np.isnan(corrcoef)):
        corrcoef[np.isnan(corrcoef)] = 1
        if verbose:
            print("Warning: NaNs in correlation matrix corrected to 1s.")

    # Calculate transition probabilities
    knn_array = embedding_knn_used.toarray()
    transition_prob = np.exp(corrcoef / sigma_corr) * knn_array
    transition_prob /= transition_prob.sum(axis=1, keepdims=True) + 1e-10

    # Calculate embedding shift
    delta_embedding = _calculate_embedding_shift(embedding, transition_prob, knn_array)

    # Store in adata
    flow_key = store_key if store_key is not None else f'perturbation_flow_{basis}'
    adata.obsm[flow_key] = delta_embedding
    adata.uns['perturbation_flow_params'] = {
        'basis': basis,
        'method': 'celloracle',
        'n_neighbors': n_neighbors,
        'sigma_corr': sigma_corr,
        'correlation_mode': correlation_mode,
        'sampled_fraction': sampled_fraction if correlation_mode == 'sampled' else None
    }

    if verbose:
        print(f"Flow stored in adata.obsm['{flow_key}']")

    return delta_embedding


def calculate_grid_flow(
    adata: AnnData,
    flow_key: str,
    basis: str = 'umap',
    n_grid: int = 40,
    method: str = 'knn',
    n_neighbors: int = 200,
    min_mass: float = 1.0,
    smooth: float = 0.5,
    n_jobs: int = 4,
) -> Dict:
    """
    Interpolate flow vectors onto a regular grid.

    This produces cleaner visualizations than plotting arrows on every cell.

    Parameters
    ----------
    adata : AnnData
        Annotated data with flow vectors
    flow_key : str
        Key in adata.obsm for flow vectors
    basis : str, optional (default: 'umap')
        Embedding basis
    n_grid : int, optional (default: 40)
        Number of grid points per dimension
    method : str, optional (default: 'knn')
        Interpolation method:
        - 'knn': KNN-based interpolation (like CellOracle)
        - 'gaussian': Gaussian kernel density interpolation
    n_neighbors : int, optional (default: 200)
        Number of neighbors for mass calculation (KNN method)
    min_mass : float, optional (default: 1.0)
        Minimum probability mass to show arrows (as percentage)
    smooth : float, optional (default: 0.5)
        Smoothing factor for Gaussian kernel (gaussian method)
    n_jobs : int, optional (default: 4)
        Number of parallel jobs for KNN

    Returns
    -------
    dict
        Dictionary with:
        - 'grid_coords': Grid coordinates (n_grid^2, 2)
        - 'grid_flow': Flow vectors on grid (n_grid^2, 2)
        - 'mass_filter': Boolean mask for low-density regions
        - 'mass': Mass values
        - 'n_grid': Number of grid points per dimension
    """
    if method == 'knn':
        return _calculate_grid_flow_knn(
            adata, flow_key=flow_key, basis=basis, n_grid=n_grid,
            n_neighbors=n_neighbors, min_mass=min_mass, n_jobs=n_jobs
        )
    elif method == 'gaussian':
        return _calculate_grid_flow_gaussian(
            adata, flow_key=flow_key, basis=basis, n_grid=n_grid,
            smooth=smooth, min_mass=min_mass
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'knn' or 'gaussian'.")


def _calculate_grid_flow_knn(
    adata: AnnData,
    flow_key: str,
    basis: str = 'umap',
    n_grid: int = 40,
    n_neighbors: int = 200,
    min_mass: float = 1.0,
    n_jobs: int = 4,
) -> Dict:
    """
    Calculate grid flow using KNN-based interpolation.

    Internal function called by calculate_grid_flow with method='knn'.
    """
    from sklearn.neighbors import NearestNeighbors

    embedding_key = f'X_{basis}'
    embedding = adata.obsm[embedding_key]
    n_cells = embedding.shape[0]

    if flow_key not in adata.obsm:
        raise ValueError(f"Flow '{flow_key}' not found. Run calculate_flow first.")

    flow = adata.obsm[flow_key]

    # Create grid
    x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()

    gx = np.linspace(x_min, x_max, n_grid)
    gy = np.linspace(y_min, y_max, n_grid)

    grid_x, grid_y = np.meshgrid(gx, gy)
    grid_coords = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    n_grid_points = len(grid_coords)

    # KNN for mass and interpolation
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, n_cells), n_jobs=n_jobs)
    nn.fit(embedding)
    distances, indices = nn.kneighbors(grid_coords)

    # Gaussian weights
    median_dist = np.median(distances)
    sigma = median_dist * 0.5
    weights = np.exp(-distances ** 2 / (2 * sigma ** 2))
    mass = weights.sum(axis=1)

    # Normalize mass
    mass = mass / mass.max()
    mass_filter = mass < (min_mass / 100)

    # Interpolate flow
    grid_flow = np.zeros((n_grid_points, 2))
    for i in range(n_grid_points):
        w = weights[i]
        w = w / (w.sum() + 1e-10)
        grid_flow[i] = np.average(flow[indices[i]], axis=0, weights=w)

    return {
        'grid_coords': grid_coords,
        'grid_flow': grid_flow,
        'mass_filter': mass_filter,
        'mass': mass,
        'n_grid': n_grid
    }


def _calculate_grid_flow_gaussian(
    adata: AnnData,
    flow_key: str,
    basis: str = 'umap',
    n_grid: int = 40,
    smooth: float = 0.5,
    min_mass: float = 1.0,
) -> Dict:
    """
    Calculate grid flow using Gaussian kernel interpolation.

    Internal function called by calculate_grid_flow with method='gaussian'.
    """
    from sklearn.neighbors import KernelDensity

    embedding_key = f'X_{basis}'
    embedding = adata.obsm[embedding_key]

    if flow_key not in adata.obsm:
        raise ValueError(f"Flow '{flow_key}' not found. Run calculate_flow first.")

    flow = adata.obsm[flow_key]

    # Create grid with padding
    x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()

    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05

    gx = np.linspace(x_min - x_pad, x_max + x_pad, n_grid)
    gy = np.linspace(y_min - y_pad, y_max + y_pad, n_grid)

    grid_x, grid_y = np.meshgrid(gx, gy)
    grid_coords = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Bandwidth based on grid spacing
    bandwidth = smooth * max((x_max - x_min), (y_max - y_min)) / n_grid

    # Density estimation
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(embedding)
    log_density = kde.score_samples(grid_coords)
    mass = np.exp(log_density)

    # Normalize mass
    mass = mass / mass.max()
    mass_filter = mass < (min_mass / 100)

    # Interpolate flow
    grid_flow = np.zeros((len(grid_coords), 2))
    for i, gc in enumerate(grid_coords):
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
    flow_key_1: str,
    flow_key_2: str,
    normalize: bool = True,
    store: bool = True,
    store_key: str = 'perturbation_inner_product',
) -> np.ndarray:
    """
    Calculate inner product between two flow fields.

    Positive values indicate flows align (e.g., perturbation promotes development).
    Negative values indicate flows oppose (e.g., perturbation inhibits development).

    Parameters
    ----------
    adata : AnnData
        Annotated data with flow vectors
    flow_key_1 : str
        Key in adata.obsm for first flow field (e.g., reference velocity)
    flow_key_2 : str
        Key in adata.obsm for second flow field (e.g., perturbation flow)
    normalize : bool, optional (default: True)
        If True, normalize vectors to unit length before computing inner product.
        This gives cosine similarity in [-1, 1].
    store : bool, optional (default: True)
        If True, store result in adata.obs[store_key]
    store_key : str, optional (default: 'perturbation_inner_product')
        Key for storing inner product values

    Returns
    -------
    np.ndarray
        Inner product values for each cell
    """
    if flow_key_1 not in adata.obsm:
        raise ValueError(f"Flow '{flow_key_1}' not found in adata.obsm")
    if flow_key_2 not in adata.obsm:
        raise ValueError(f"Flow '{flow_key_2}' not found in adata.obsm")

    flow1 = adata.obsm[flow_key_1]
    flow2 = adata.obsm[flow_key_2]

    if normalize:
        norm1 = np.linalg.norm(flow1, axis=1, keepdims=True) + 1e-10
        norm2 = np.linalg.norm(flow2, axis=1, keepdims=True) + 1e-10
        flow1 = flow1 / norm1
        flow2 = flow2 / norm2

    inner_product = np.sum(flow1 * flow2, axis=1)

    if store:
        adata.obs[store_key] = inner_product
        # Also compute magnitude-weighted version
        if normalize:
            flow1_orig = adata.obsm[flow_key_1]
            flow2_orig = adata.obsm[flow_key_2]
            weighted = np.sum(flow1_orig * flow2_orig, axis=1)
            adata.obs[f'{store_key}_weighted'] = weighted

    return inner_product


# =============================================================================
# Private helper functions
# =============================================================================

def _calculate_embedding_shift(
    embedding: np.ndarray,
    transition_prob: np.ndarray,
    knn_array: np.ndarray
) -> np.ndarray:
    """
    Calculate embedding shift from transition probabilities.

    Follows CellOracle's calculate_embedding_shift logic.
    """
    # Unitary vectors from each cell to all other cells
    unitary_vectors = embedding.T[:, None, :] - embedding.T[:, :, None]

    # Normalize to unit vectors
    with np.errstate(divide='ignore', invalid='ignore'):
        norms = np.linalg.norm(unitary_vectors, ord=2, axis=0)
        unitary_vectors = unitary_vectors / (norms + 1e-10)
        np.fill_diagonal(unitary_vectors[0], 0)
        np.fill_diagonal(unitary_vectors[1], 0)

    # Weighted sum of directions
    delta_embedding = (transition_prob * unitary_vectors).sum(axis=2)

    # Subtract baseline
    knn_sum = knn_array.sum(axis=1, keepdims=True)
    baseline = (knn_array * unitary_vectors).sum(axis=2) / (knn_sum.T + 1e-10)
    delta_embedding = delta_embedding - baseline

    return delta_embedding.T


def _calculate_correlation_sampled(
    X: np.ndarray,
    delta_X: np.ndarray,
    neigh_ixs: np.ndarray,
    verbose: bool = True
) -> np.ndarray:
    """
    Calculate correlation between delta_X and neighbor expression differences.

    For each cell i and its sampled neighbors j, computes:
    corr(delta_X[i], X[j] - X[i])
    """
    from tqdm.auto import tqdm

    n_cells, n_neighbors = neigh_ixs.shape
    corrcoef = np.zeros((n_cells, n_cells))

    iterator = range(n_cells)
    if verbose:
        iterator = tqdm(iterator, desc="Calculating correlations (sampled)")

    for i in iterator:
        neighbors = neigh_ixs[i]
        diffs = X[neighbors] - X[i]
        corrs = _pearson_correlation_rows(delta_X[i:i+1], diffs)

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
    """
    from tqdm.auto import tqdm

    n_cells = X.shape[0]
    corrcoef = np.zeros((n_cells, n_cells))

    iterator = range(n_cells)
    if verbose:
        iterator = tqdm(iterator, desc="Calculating correlations (full)")

    for i in iterator:
        diffs = X - X[i]
        corrs = _pearson_correlation_rows(delta_X[i:i+1], diffs)
        corrcoef[i, :] = corrs

    return corrcoef


def _pearson_correlation_rows(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute Pearson correlation between vector a and each row of matrix B.
    """
    a_centered = a - a.mean()
    B_centered = B - B.mean(axis=1, keepdims=True)

    ss_a = np.sum(a_centered ** 2)
    ss_B = np.sum(B_centered ** 2, axis=1)

    if ss_a < 1e-10:
        return np.zeros(B.shape[0])

    numerator = (a_centered @ B_centered.T).flatten()
    denominator = np.sqrt(ss_a) * np.sqrt(ss_B)

    with np.errstate(divide='ignore', invalid='ignore'):
        corrs = numerator / (denominator + 1e-10)
        corrs[ss_B < 1e-10] = 0

    return corrs
