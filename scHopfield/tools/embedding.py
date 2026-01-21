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
    """Compute UMAP embedding."""
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
    """Compute energy landscape on embedding space."""
    adata = adata.copy() if copy else adata
    
    genes = get_genes_used(adata)
    
    cells2d = adata.obsm[f'X_{basis}']
    embedding = adata.uns['scHopfield']['embedding']
    
    clusters = adata.obs[cluster_key].unique()
    
    # Generate grids
    grid_X, grid_Y = {}, {}
    for cluster in clusters:
        cidx = adata.obs[cluster_key] == cluster
        
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
    """Save embedding and grid data to file."""
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
    """Load embedding from file."""
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
