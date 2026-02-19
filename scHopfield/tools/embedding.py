"""Dimensionality reduction and energy landscape embedding."""

import numpy as np
import pickle
from typing import Optional
from anndata import AnnData

from .._utils.math import soften, sigmoid, int_sig_act_inv
from .._utils.io import get_matrix, to_numpy, get_genes_used


def compute_umap(
    adata: AnnData,
    spliced_key: str = 'Ms',
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    basis: str = 'umap',
    copy: bool = False
) -> Optional[AnnData]:
    """
    Compute UMAP embedding from gene expression data.

    Performs dimensionality reduction using UMAP on the selected gene expression
    layer. The UMAP model is stored in adata.uns['scHopfield']['embedding'] and
    the 2D coordinates are stored in adata.obsm[f'X_{basis}'].

    Parameters
    ----------
    adata : AnnData
        Annotated data object
    spliced_key : str, optional (default: 'Ms')
        Key in adata.layers for expression data to use
    n_neighbors : int, optional (default: 30)
        Number of neighbors for UMAP
    min_dist : float, optional (default: 0.1)
        Minimum distance parameter for UMAP
    basis : str, optional (default: 'umap')
        Name for the embedding basis (stored as 'X_{basis}' in obsm)
    copy : bool, optional (default: False)
        Whether to return a copy or modify in place

    Returns
    -------
    Optional[AnnData]
        Returns AnnData if copy=True, otherwise modifies in place and returns None
    """
    adata = adata.copy() if copy else adata
    
    genes = get_genes_used(adata)
    X = to_numpy(get_matrix(adata, spliced_key, genes=genes))
    
    import umap
    emb = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2)
    cells2d = emb.fit_transform(X)
    
    adata.uns['scHopfield']['embedding'] = emb
    adata.obsm[f'X_{basis}'] = cells2d
    
    return adata if copy else None


def energy_embedding(
    adata: AnnData,
    basis: str = 'umap',
    resolution: int = 50,
    cluster_key: str = 'cell_type',
    degradation_key: str = 'gamma',
    copy: bool = False
) -> Optional[AnnData]:
    """
    Compute energy landscape on 2D embedding space.

    For each cluster, creates a grid in the embedding space and computes the
    Hopfield energy at each grid point. The grid is transformed to the original
    high-dimensional gene expression space using the inverse UMAP transform,
    and energies are computed using the cluster-specific interaction matrices.

    Stores grid coordinates and energy values in adata.uns['scHopfield'] with
    keys: 'grid_X_{cluster}', 'grid_Y_{cluster}', 'grid_energy_{cluster}',
    'grid_energy_interaction_{cluster}', 'grid_energy_degradation_{cluster}',
    'grid_energy_bias_{cluster}'.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with computed UMAP embedding
    basis : str, optional (default: 'umap')
        Name of the embedding basis to use (from obsm['X_{basis}'])
    resolution : int, optional (default: 50)
        Number of grid points per dimension (creates resolution x resolution grid)
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    degradation_key : str, optional (default: 'gamma')
        Key in adata.var for degradation rates (fallback if cluster-specific not found)
    copy : bool, optional (default: False)
        Whether to return a copy or modify in place

    Returns
    -------
    Optional[AnnData]
        Returns AnnData if copy=True, otherwise modifies in place and returns None
    """
    adata = adata.copy() if copy else adata
    
    genes = get_genes_used(adata)
    
    cells2d = adata.obsm[f'X_{basis}']
    embedding = adata.uns['scHopfield']['embedding']
    
    clusters = adata.obs[cluster_key].unique()
    
    # Generate grids
    grid_X, grid_Y = {}, {}
    for cluster in clusters:
        cidx = (adata.obs[cluster_key] == cluster).values

        minx, miny = np.min(cells2d[cidx], axis=0)
        maxx, maxy = np.max(cells2d[cidx], axis=0)
        grid_X[cluster], grid_Y[cluster] = np.mgrid[minx:maxx:resolution*1j, miny:maxy:resolution*1j]
    
    # Transform to high-D space
    all_grid_points = np.vstack([np.c_[grid_X[k].ravel(), grid_Y[k].ravel()] for k in clusters])
    highD_grid = embedding.inverse_transform(all_grid_points)
    highD_grid = np.maximum(highD_grid, 0)
    
    adata.varm['highD_grid'] = highD_grid
    
    # Compute energies on grid
    threshold = adata.var['sigmoid_threshold'].values[genes]
    exponent = adata.var['sigmoid_exponent'].values[genes]
    
    for i, cluster in enumerate(clusters):
        start_idx = i * resolution**2
        end_idx = (i + 1) * resolution**2
        x_grid = highD_grid[start_idx:end_idx]
        
        sig_grid = sigmoid(x_grid, threshold[None, :], exponent[None, :])
        
        W = adata.varp[f'W_{cluster}']
        I = adata.var[f'I_{cluster}'].values[genes]
        
        gamma_key = f'gamma_{cluster}'
        g = adata.var[gamma_key].values[genes] if gamma_key in adata.var else adata.var[degradation_key].values[genes]
        
        e_int = -0.5 * np.sum((sig_grid @ W.T) * sig_grid, axis=1)
        integral = int_sig_act_inv(sig_grid, threshold, exponent)
        e_deg = np.sum(g[None, :] * integral, axis=1)
        e_bias = -np.sum(I[None, :] * sig_grid, axis=1)
        e_total = e_int + e_deg + e_bias
        
        shape = grid_X[cluster].shape
        adata.uns['scHopfield'][f'grid_X_{cluster}'] = grid_X[cluster]
        adata.uns['scHopfield'][f'grid_Y_{cluster}'] = grid_Y[cluster]
        adata.uns['scHopfield'][f'grid_energy_{cluster}'] = soften(e_total.reshape(shape))
        adata.uns['scHopfield'][f'grid_energy_interaction_{cluster}'] = soften(e_int.reshape(shape))
        adata.uns['scHopfield'][f'grid_energy_degradation_{cluster}'] = soften(e_deg.reshape(shape))
        adata.uns['scHopfield'][f'grid_energy_bias_{cluster}'] = soften(e_bias.reshape(shape))
    
    return adata if copy else None


def save_embedding(adata: AnnData, filename: str):
    """
    Save UMAP embedding and energy grid data to file.

    Saves the UMAP model, high-dimensional grid points, and grid coordinates
    to a pickle file for later loading.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with computed embedding
    filename : str
        Path to save the embedding data (will be saved as pickle file)
    """
    emb_data = {
        'embedding': adata.uns['scHopfield']['embedding'],
        'highD_grid': adata.varm['highD_grid']
    }
    
    for key in adata.uns['scHopfield'].keys():
        if key.startswith('grid_X_') or key.startswith('grid_Y_'):
            emb_data[key] = adata.uns['scHopfield'][key]
    
    with open(filename, 'wb') as f:
        pickle.dump(emb_data, f, pickle.HIGHEST_PROTOCOL)


def load_embedding(
    adata: AnnData,
    filename: str,
    basis: str = 'umap',
    copy: bool = False
) -> Optional[AnnData]:
    """
    Load UMAP embedding and energy grid data from file.

    Loads the UMAP model and grid data from a pickle file saved with
    save_embedding(). Also transforms the current expression data using
    the loaded UMAP model to get 2D coordinates.

    Parameters
    ----------
    adata : AnnData
        Annotated data object to load embedding into
    filename : str
        Path to the saved embedding pickle file
    basis : str, optional (default: 'umap')
        Name for the embedding basis (stored as 'X_{basis}' in obsm)
    copy : bool, optional (default: False)
        Whether to return a copy or modify in place

    Returns
    -------
    Optional[AnnData]
        Returns AnnData if copy=True, otherwise modifies in place and returns None
    """
    adata = adata.copy() if copy else adata
    
    with open(filename, 'rb') as f:
        emb_data = pickle.load(f)
    
    adata.uns['scHopfield']['embedding'] = emb_data['embedding']
    adata.varm['highD_grid'] = emb_data['highD_grid']
    
    for key in emb_data.keys():
        if key.startswith('grid_X_') or key.startswith('grid_Y_'):
            adata.uns['scHopfield'][key] = emb_data[key]
    
    genes = get_genes_used(adata)
    X = to_numpy(get_matrix(adata, 'Ms', genes=genes))
    cells2d = emb_data['embedding'].transform(X)
    adata.obsm[f'X_{basis}'] = cells2d

    return adata if copy else None


from typing import Tuple, Optional
import numpy as np

def project_to_embedding(
    adata: 'AnnData',
    vectors: np.ndarray,
    basis: str = 'umap',
    method: str = 'dot_product',
    n_neighbors: int = 30,
    n_jobs: int = 4,
    spliced_key: Optional[str] = None,
    # CellOracle correlation specific parameters
    sigma_corr: float = 0.05,
    correlation_mode: str = 'sampled',
    sampled_fraction: float = 0.3,
    sampling_probs: Tuple[float, float] = (0.5, 0.1),
    random_seed: int = 42,
    verbose: bool = False
) -> np.ndarray:
    """
    Project gene-space vectors to embedding space.

    Parameters
    ----------
    adata : AnnData
        Annotated data with embedding
    vectors : np.ndarray
        Gene-space vectors (n_cells, n_genes) - can be velocities, delta_X, etc.
    basis : str, optional (default: 'umap')
        Embedding to project onto (key in adata.obsm as X_{basis})
    method : str, optional (default: 'dot_product')
        Projection method:
        - 'dot_product': Gene-space KNN, weights neighbors by vector alignment.
        - 'correlation': Embedding-space KNN, CellOracle-style correlation.
    n_neighbors : int, optional (default: 30)
        Number of neighbors for projection (Note: correlation method often uses ~200)
    spliced_key : str, optional
        Key for expression data. Defaults to scHopfield's spliced_key or 'Ms'.
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy import sparse
    from scipy.sparse import issparse

    embedding_key = f'X_{basis}'
    if embedding_key not in adata.obsm:
        raise ValueError(f"Embedding {embedding_key} not found in adata.obsm")

    embedding = adata.obsm[embedding_key]
    n_cells = embedding.shape[0]

    genes = get_genes_used(adata)

    if spliced_key is None:
        spliced_key = adata.uns.get('scHopfield', {}).get('spliced_key', 'Ms')

    if spliced_key in adata.layers:
        X = adata.layers[spliced_key][:, genes]
    else:
        X = adata.X[:, genes]
        
    if issparse(X): X = X.toarray()
    if issparse(vectors): vectors = vectors.toarray()

    # -------------------------------------------------------------------
    # Method 1: Gene-Space Dot Product Alignment
    # -------------------------------------------------------------------
    if method == 'dot_product':
        if verbose: print("Projecting using gene-space dot product alignment...")
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)

        embedding_vectors = np.zeros((n_cells, 2))

        for i in range(n_cells):
            neighbors = indices[i, 1:]  # Exclude self
            dX = X[neighbors] - X[i]
            alignment = (vectors[i] * dX).sum(axis=1)

            dists = distances[i, 1:]
            weights = np.exp(-dists / (np.median(dists) + 1e-10))
            weights = weights * np.maximum(alignment, 0)
            weights = weights / (weights.sum() + 1e-10)

            dE = embedding[neighbors] - embedding[i]
            embedding_vectors[i] = (weights[:, None] * dE).sum(axis=0)

        return embedding_vectors

    # -------------------------------------------------------------------
    # Method 2: Embedding-Space Correlation (CellOracle)
    # -------------------------------------------------------------------
    elif method == 'correlation':
        if verbose: print("Projecting using embedding-space correlation (CellOracle style)...")
        np.random.seed(random_seed)
        
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs)
        nn.fit(embedding)
        embedding_knn = nn.kneighbors_graph(mode="connectivity")

        if correlation_mode == 'sampled':
            neigh_ixs = embedding_knn.indices.reshape((-1, n_neighbors + 1))
            p = np.linspace(sampling_probs[0], sampling_probs[1], neigh_ixs.shape[1])
            p = p / p.sum()

            n_sampled = int(sampled_fraction * (n_neighbors + 1))
            sampling_ixs = np.stack([
                np.random.choice(neigh_ixs.shape[1], size=n_sampled, replace=False, p=p)
                for _ in range(n_cells)
            ], axis=0)

            neigh_ixs = neigh_ixs[np.arange(n_cells)[:, None], sampling_ixs]

            # Computes correlation for sampled neighbors
            corrcoef = _calculate_correlation_sampled(X, vectors, neigh_ixs, verbose=verbose)

            nonzero = n_cells * n_sampled
            embedding_knn_used = sparse.csr_matrix(
                (np.ones(nonzero), neigh_ixs.ravel(), np.arange(0, nonzero + 1, n_sampled)),
                shape=(n_cells, n_cells)
            )

        elif correlation_mode == 'full':
            corrcoef = _calculate_correlation_full(X, vectors, verbose=verbose)
            np.fill_diagonal(corrcoef, 0)
            embedding_knn_used = embedding_knn

        else:
            raise ValueError(f"Unknown correlation_mode: {correlation_mode}")

        if np.any(np.isnan(corrcoef)):
            corrcoef[np.isnan(corrcoef)] = 1
            if verbose: print("Warning: NaNs in correlation matrix corrected to 1s.")

        knn_array = embedding_knn_used.toarray()
        transition_prob = np.exp(corrcoef / sigma_corr) * knn_array
        transition_prob /= transition_prob.sum(axis=1, keepdims=True) + 1e-10

        return _calculate_embedding_shift(embedding, transition_prob, knn_array)

    else:
        raise ValueError(f"Unknown method: '{method}'. Use 'dot_product' or 'correlation'.")

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

