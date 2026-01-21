"""Jacobian matrix computation and stability analysis."""

import numpy as np
import torch
from typing import Optional
from anndata import AnnData
from tqdm import tqdm

from .._utils.io import get_matrix, to_numpy, get_genes_used
from .._utils.math import sigmoid


def compute_jacobians(
    adata: AnnData,
    spliced_key: str = 'Ms',
    degradation_key: str = 'gamma',
    cluster_key: str = 'cell_type',
    compute_eigenvectors: bool = False,
    device: str = 'cpu',
    copy: bool = False
) -> Optional[AnnData]:
    """
    Compute Jacobian matrices and eigenvalues for all cells.
    
    The Jacobian is: J = W * diag(dsigmoid/dx) - diag(gamma)
    """
    adata = adata.copy() if copy else adata

    genes = get_genes_used(adata)
    n_genes = len(genes)
    n_cells = adata.n_obs
    
    device = torch.device("cuda" if (torch.cuda.is_available() and device == "cuda") else "cpu")
    
    jacobian_eigenvalues = np.zeros((n_cells, n_genes), dtype=np.complex128)
    jacobian_eigenvectors = None
    if compute_eigenvectors:
        jacobian_eigenvectors = np.zeros((n_cells, n_genes, n_genes), dtype=np.complex128)
    
    threshold = adata.var['sigmoid_threshold'].values[genes]
    exponent = adata.var['sigmoid_exponent'].values[genes]
    
    clusters = adata.obs[cluster_key].unique()

    for cluster in clusters:
        print(f"Computing Jacobians for cluster {cluster}")
        
        gamma_key = f'gamma_{cluster}'
        g = adata.var[gamma_key].values[genes] if gamma_key in adata.var else adata.var[degradation_key].values[genes]
        
        gamma = torch.diag(torch.tensor(g, device=device, dtype=torch.float32))
        W = torch.tensor(adata.varp[f'W_{cluster}'], device=device, dtype=torch.float32)
        
        cluster_indices = np.where(adata.obs[cluster_key] == cluster)[0]
        cell_data = torch.tensor(
            to_numpy(get_matrix(adata, spliced_key, genes=genes)[cluster_indices]),
            device=device,
            dtype=torch.float32
        )
        
        sigmoid_values = torch.tensor(
            sigmoid(cell_data.cpu().numpy(), threshold[None, :], exponent[None, :]),
            device=device,
            dtype=torch.float32
        )
        sigmoid_prime = (
            torch.tensor(exponent, device=device, dtype=torch.float32)
            * sigmoid_values
            * (1 - sigmoid_values)
            / torch.where(cell_data == 0, torch.ones_like(cell_data), cell_data)
        )
        
        for idx, sig_prime_val in tqdm(zip(cluster_indices, sigmoid_prime), total=len(cluster_indices)):
            jac_f = W * sig_prime_val.view(-1, 1) - gamma
            
            if compute_eigenvectors:
                eigenvalues, eigenvectors = torch.linalg.eig(jac_f)
                jacobian_eigenvectors[idx] = eigenvectors.cpu().numpy()
            else:
                eigenvalues = torch.linalg.eigvals(jac_f)
            
            jacobian_eigenvalues[idx] = eigenvalues.cpu().numpy()
    
    adata.obsm['jacobian_eigenvalues'] = jacobian_eigenvalues
    
    if compute_eigenvectors:
        adata.uns['jacobian_eigenvectors_temp'] = {'data': jacobian_eigenvectors}
        print("Warning: Eigenvectors stored temporarily. Use sch.tl.save_jacobians() to save externally.")
    
    return adata if copy else None


def save_jacobians(adata: AnnData, filename: str, cluster_key: str = 'cell_type', compression: str = 'gzip'):
    """Save Jacobian eigenvalues and eigenvectors to HDF5 file."""
    import h5py

    with h5py.File(filename, 'w') as f:
        if 'jacobian_eigenvalues' in adata.obsm:
            evals = adata.obsm['jacobian_eigenvalues']
            f.create_dataset('eigenvalues_real', data=evals.real, compression=compression)
            f.create_dataset('eigenvalues_imag', data=evals.imag, compression=compression)

        if 'jacobian_eigenvectors_temp' in adata.uns:
            evecs = adata.uns['jacobian_eigenvectors_temp']['data']
            f.create_dataset('eigenvectors_real', data=evecs.real, compression=compression)
            f.create_dataset('eigenvectors_imag', data=evecs.imag, compression=compression)
            del adata.uns['jacobian_eigenvectors_temp']
            print("Eigenvectors saved to file and removed from memory.")

        f.attrs['n_cells'] = adata.n_obs
        f.attrs['n_genes'] = len(get_genes_used(adata))
        f.attrs['cluster_key'] = cluster_key


def load_jacobians(
    adata: AnnData,
    filename: str,
    load_eigenvectors: bool = False
) -> Optional[np.ndarray]:
    """Load Jacobian eigenvalues and optionally eigenvectors from HDF5 file."""
    import h5py
    
    with h5py.File(filename, 'r') as f:
        evals_real = f['eigenvalues_real'][:]
        evals_imag = f['eigenvalues_imag'][:]
        adata.obsm['jacobian_eigenvalues'] = evals_real + 1j * evals_imag
        
        if load_eigenvectors and 'eigenvectors_real' in f:
            evecs_real = f['eigenvectors_real'][:]
            evecs_imag = f['eigenvectors_imag'][:]
            eigenvectors = evecs_real + 1j * evecs_imag
            print(f"Loaded eigenvectors with shape {eigenvectors.shape}")
            return eigenvectors
    
    return None
