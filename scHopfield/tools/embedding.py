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


def project_to_embedding(
    adata: AnnData,
    vectors: np.ndarray,
    basis: str = 'umap',
    n_neighbors: int = 30,
    n_jobs: int = 4,
    spliced_key: Optional[str] = None,
) -> np.ndarray:
    """
    Project gene-space vectors to embedding space.

    Uses neighbor-based averaging: for each cell, weight neighbors by
    alignment of vector with direction to neighbor in gene space,
    then average their directions in embedding space.

    This is useful for projecting velocity vectors, perturbation effects,
    gradients, or any gene-space direction to the 2D embedding.

    Parameters
    ----------
    adata : AnnData
        Annotated data with embedding
    vectors : np.ndarray
        Gene-space vectors (n_cells, n_genes) - can be velocities,
        perturbation effects, gradients, etc.
    basis : str, optional (default: 'umap')
        Embedding to project onto (key in adata.obsm as X_{basis})
    n_neighbors : int, optional (default: 30)
        Number of neighbors for projection
    n_jobs : int, optional (default: 4)
        Number of parallel jobs
    spliced_key : str, optional
        Key for expression data. If None, uses adata.uns['scHopfield']['spliced_key']
        or defaults to 'Ms'.

    Returns
    -------
    np.ndarray
        Vectors in embedding space (n_cells, 2)
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse import issparse

    embedding_key = f'X_{basis}'
    embedding = adata.obsm[embedding_key]
    n_cells = embedding.shape[0]

    genes = get_genes_used(adata)

    # Get spliced key
    if spliced_key is None:
        spliced_key = adata.uns.get('scHopfield', {}).get('spliced_key', 'Ms')

    # Get expression data
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

    # Project vectors to embedding
    embedding_vectors = np.zeros((n_cells, 2))

    for i in range(n_cells):
        neighbors = indices[i, 1:]  # Exclude self

        # Direction to neighbors in gene space
        dX = X[neighbors] - X[i]

        # Compute alignment of vector with direction to each neighbor
        # (positive = moving towards neighbor)
        alignment = (vectors[i] * dX).sum(axis=1)

        # Normalize by distance (give more weight to closer neighbors)
        dists = distances[i, 1:]
        weights = np.exp(-dists / (np.median(dists) + 1e-10))
        weights = weights * np.maximum(alignment, 0)  # Only positive alignment
        weights = weights / (weights.sum() + 1e-10)

        # Direction to neighbors in embedding space
        dE = embedding[neighbors] - embedding[i]

        # Weighted average direction
        embedding_vectors[i] = (weights[:, None] * dE).sum(axis=0)

    return embedding_vectors
