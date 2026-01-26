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

    For each cell, computes the Jacobian matrix of the Hopfield dynamics:
    J = W * diag(dsigmoid/dx) - diag(gamma)
    where W is the interaction matrix, dsigmoid/dx is the derivative of the
    sigmoid activation, and gamma is the degradation rate.

    Eigenvalues are stored in adata.obsm['jacobian_eigenvalues']. Eigenvectors
    can optionally be computed but are stored temporarily and should be saved
    to disk using save_jacobians() to avoid memory issues.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with fitted parameters
    spliced_key : str, optional (default: 'Ms')
        Layer key for spliced counts
    degradation_key : str, optional (default: 'gamma')
        Base key for degradation rates (cluster-specific rates used if available)
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    compute_eigenvectors : bool, optional (default: False)
        Whether to compute eigenvectors (requires more memory and time)
    device : str, optional (default: 'cpu')
        Device for computation: 'cpu' or 'cuda'
    copy : bool, optional (default: False)
        If True, return a copy instead of modifying in-place

    Returns
    -------
    Optional[AnnData]
        Returns adata if copy=True, otherwise modifies in place and returns None.
        Adds to adata.obsm:
        - 'jacobian_eigenvalues': Complex eigenvalues array of shape (n_cells, n_genes)
        If compute_eigenvectors=True, temporarily stores in adata.uns['jacobian_eigenvectors_temp']
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
    """
    Save Jacobian eigenvalues and eigenvectors to HDF5 file.

    Saves eigenvalues and eigenvectors (if computed) to an HDF5 file with
    compression. Complex arrays are stored as separate real and imaginary
    parts. Removes temporary eigenvector storage from adata after saving.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with computed Jacobian data
    filename : str
        Path to save HDF5 file
    cluster_key : str, optional (default: 'cell_type')
        Cluster key to store as metadata attribute
    compression : str, optional (default: 'gzip')
        HDF5 compression algorithm ('gzip', 'lzf', etc.)
    """
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
    """
    Load Jacobian eigenvalues and optionally eigenvectors from HDF5 file.

    Loads eigenvalues into adata.obsm['jacobian_eigenvalues']. Optionally
    loads and returns eigenvectors as a numpy array (not stored in adata
    to save memory).

    Parameters
    ----------
    adata : AnnData
        Annotated data object to load data into
    filename : str
        Path to HDF5 file with saved Jacobian data
    load_eigenvectors : bool, optional (default: False)
        Whether to load eigenvectors (returns them if True)

    Returns
    -------
    Optional[np.ndarray]
        If load_eigenvectors=True, returns eigenvectors array of shape
        (n_cells, n_genes, n_genes). Otherwise returns None.
        Always loads eigenvalues into adata.obsm['jacobian_eigenvalues']
    """
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


def compute_jacobian_stats(
    adata: AnnData,
    filename: Optional[str] = None,
    store_in_obs: bool = True,
    copy: bool = False
) -> Optional[AnnData]:
    """
    Compute summary statistics from Jacobian eigenvalues.

    Computes and stores:
    - First eigenvalue (real and imaginary parts)
    - Number of positive/negative real eigenvalues
    - Jacobian trace (sum of eigenvalues)
    - Rotational part magnitude (if full Jacobians available)

    Parameters
    ----------
    adata : AnnData
        Annotated data object with jacobian_eigenvalues in obsm or filename specified
    filename : str, optional
        Path to HDF5 file with saved Jacobians. If None, uses adata.obsm['jacobian_eigenvalues']
    store_in_obs : bool, optional (default: True)
        Whether to store results in adata.obs
    copy : bool, optional (default: False)
        If True, return a copy instead of modifying in-place

    Returns
    -------
    AnnData or None
        Returns adata if copy=True, otherwise None.
        Adds to adata.obs:
        - 'jacobian_eig1_real': Real part of first eigenvalue
        - 'jacobian_eig1_imag': Imaginary part of first eigenvalue
        - 'jacobian_positive_evals': Count of positive real eigenvalues
        - 'jacobian_negative_evals': Count of negative real eigenvalues
        - 'jacobian_trace': Trace of Jacobian (sum of eigenvalues)
        - 'jacobian_rotational': Rotational part magnitude (if available)
    """
    adata = adata.copy() if copy else adata

    # Load eigenvalues
    if 'jacobian_eigenvalues' in adata.obsm:
        eigenvalues = adata.obsm['jacobian_eigenvalues']
    elif filename is not None:
        import h5py
        with h5py.File(filename, 'r') as f:
            evals_real = f['eigenvalues_real'][:]
            evals_imag = f['eigenvalues_imag'][:]
            eigenvalues = evals_real + 1j * evals_imag
    else:
        raise ValueError(
            "No Jacobian eigenvalues found. Either compute them first with "
            "sch.tl.compute_jacobians() or provide filename to load from."
        )

    if store_in_obs:
        # First eigenvalue
        adata.obs['jacobian_eig1_real'] = eigenvalues[:, 0].real
        adata.obs['jacobian_eig1_imag'] = eigenvalues[:, 0].imag

        # Count positive and negative eigenvalues
        adata.obs['jacobian_positive_evals'] = np.sum(eigenvalues.real > 0, axis=1)
        adata.obs['jacobian_negative_evals'] = np.sum(eigenvalues.real < 0, axis=1)

        # Trace (sum of eigenvalues)
        adata.obs['jacobian_trace'] = np.sum(eigenvalues.real, axis=1)

        # Try to compute rotational part if full Jacobians available
        if filename is not None:
            try:
                import h5py
                with h5py.File(filename, 'r') as f:
                    if 'eigenvectors_real' in f:
                        # Load full Jacobians if needed
                        print("Note: Full Jacobian matrices needed for rotational part.")
                        print("This requires computing Jacobians on-the-fly or storing them.")
            except:
                pass

    return adata if copy else None


def compute_jacobian_elements(
    adata: AnnData,
    gene_pairs: list,
    spliced_key: str = 'Ms',
    degradation_key: str = 'gamma',
    cluster_key: str = 'cell_type',
    device: str = 'cpu',
    store_in_obs: bool = True,
    copy: bool = False
) -> Optional[AnnData]:
    """
    Compute specific Jacobian matrix elements (partial derivatives).

    Computes df_i/dx_j for specified gene pairs, where f_i is the
    dynamics function for gene i and x_j is the expression of gene j.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with fitted parameters
    gene_pairs : list of tuples
        List of (gene_i, gene_j) pairs to compute df_i/dx_j.
        Example: [('GATA1', 'GATA2'), ('FLI1', 'KLF1')]
    spliced_key : str, optional (default: 'Ms')
        Layer key for spliced counts
    degradation_key : str, optional (default: 'gamma')
        Base key for degradation rates
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    device : str, optional (default: 'cpu')
        Device for computation: 'cpu' or 'cuda'
    store_in_obs : bool, optional (default: True)
        Whether to store results in adata.obs with formatted names
    copy : bool, optional (default: False)
        If True, return a copy instead of modifying in-place

    Returns
    -------
    AnnData or None
        Returns adata if copy=True, otherwise None.
        Adds to adata.obs columns like:
        'jacobian_df_GATA1_dx_GATA2': Partial derivative of GATA1 w.r.t. GATA2

    Notes
    -----
    The Jacobian element is: df_i/dx_j = W_ij * dsigmoid/dx_j - gamma_i * delta_ij
    where delta_ij is the Kronecker delta (1 if i==j, 0 otherwise).
    """
    adata = adata.copy() if copy else adata

    genes = get_genes_used(adata)
    gene_names = adata.var.index[genes]

    # Get gene indices
    gene_indices = {}
    for gene_i, gene_j in gene_pairs:
        if gene_i not in gene_indices:
            idx_i = np.where(gene_names == gene_i)[0]
            if len(idx_i) == 0:
                raise ValueError(f"Gene '{gene_i}' not found in dataset")
            gene_indices[gene_i] = idx_i[0]

        if gene_j not in gene_indices:
            idx_j = np.where(gene_names == gene_j)[0]
            if len(idx_j) == 0:
                raise ValueError(f"Gene '{gene_j}' not found in dataset")
            gene_indices[gene_j] = idx_j[0]

    threshold = adata.var['sigmoid_threshold'].values[genes]
    exponent = adata.var['sigmoid_exponent'].values[genes]
    n_cells = adata.n_obs

    device_obj = torch.device("cuda" if (torch.cuda.is_available() and device == "cuda") else "cpu")

    # Initialize result storage
    results = {f"df_{gi}_dx_{gj}": np.zeros(n_cells) for gi, gj in gene_pairs}

    clusters = adata.obs[cluster_key].unique()

    for cluster in clusters:
        print(f"Computing Jacobian elements for cluster {cluster}")

        # Get parameters
        gamma_key = f'gamma_{cluster}'
        g = adata.var[gamma_key].values[genes] if gamma_key in adata.var else adata.var[degradation_key].values[genes]

        gamma = torch.diag(torch.tensor(g, device=device_obj, dtype=torch.float32))
        W = torch.tensor(adata.varp[f'W_{cluster}'], device=device_obj, dtype=torch.float32)

        cluster_indices = np.where(adata.obs[cluster_key] == cluster)[0]
        cell_data = torch.tensor(
            to_numpy(get_matrix(adata, spliced_key, genes=genes)[cluster_indices]),
            device=device_obj,
            dtype=torch.float32
        )

        # Compute sigmoid derivative
        sigmoid_values = torch.tensor(
            sigmoid(cell_data.cpu().numpy(), threshold[None, :], exponent[None, :]),
            device=device_obj,
            dtype=torch.float32
        )
        sigmoid_prime = (
            torch.tensor(exponent, device=device_obj, dtype=torch.float32)
            * sigmoid_values
            * (1 - sigmoid_values)
            / torch.where(cell_data == 0, torch.ones_like(cell_data), cell_data)
        )

        # Compute Jacobian elements for each cell
        for local_idx, global_idx in enumerate(tqdm(cluster_indices, desc=f"Cluster {cluster}")):
            sig_prime_val = sigmoid_prime[local_idx]
            jac_f = W * sig_prime_val.view(-1, 1) - gamma
            jac_np = jac_f.cpu().numpy()

            # Extract requested elements
            for gene_i, gene_j in gene_pairs:
                idx_i = gene_indices[gene_i]
                idx_j = gene_indices[gene_j]
                results[f"df_{gene_i}_dx_{gene_j}"][global_idx] = jac_np[idx_i, idx_j]

    # Store results
    if store_in_obs:
        for key, values in results.items():
            adata.obs[f'jacobian_{key}'] = values

    return adata if copy else None


def compute_rotational_part(
    adata: AnnData,
    spliced_key: str = 'Ms',
    degradation_key: str = 'gamma',
    cluster_key: str = 'cell_type',
    device: str = 'cpu',
    copy: bool = False
) -> Optional[AnnData]:
    """
    Compute the rotational (antisymmetric) part of the Jacobian.

    The rotational part is: A = 0.5 * (J - J^T)
    Its Frobenius norm indicates local rotation strength.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with fitted parameters
    spliced_key : str, optional (default: 'Ms')
        Layer key for spliced counts
    degradation_key : str, optional (default: 'gamma')
        Base key for degradation rates
    cluster_key : str, optional (default: 'cell_type')
        Key in adata.obs for cluster labels
    device : str, optional (default: 'cpu')
        Device for computation: 'cpu' or 'cuda'
    copy : bool, optional (default: False)
        If True, return a copy instead of modifying in-place

    Returns
    -------
    AnnData or None
        Returns adata if copy=True, otherwise None.
        Adds to adata.obs:
        - 'jacobian_rotational': Frobenius norm of rotational part
    """
    adata = adata.copy() if copy else adata

    genes = get_genes_used(adata)
    n_cells = adata.n_obs
    threshold = adata.var['sigmoid_threshold'].values[genes]
    exponent = adata.var['sigmoid_exponent'].values[genes]

    device_obj = torch.device("cuda" if (torch.cuda.is_available() and device == "cuda") else "cpu")

    rotational_norms = np.zeros(n_cells)
    clusters = adata.obs[cluster_key].unique()

    for cluster in clusters:
        print(f"Computing rotational part for cluster {cluster}")

        gamma_key = f'gamma_{cluster}'
        g = adata.var[gamma_key].values[genes] if gamma_key in adata.var else adata.var[degradation_key].values[genes]

        gamma = torch.diag(torch.tensor(g, device=device_obj, dtype=torch.float32))
        W = torch.tensor(adata.varp[f'W_{cluster}'], device=device_obj, dtype=torch.float32)

        cluster_indices = np.where(adata.obs[cluster_key] == cluster)[0]
        cell_data = torch.tensor(
            to_numpy(get_matrix(adata, spliced_key, genes=genes)[cluster_indices]),
            device=device_obj,
            dtype=torch.float32
        )

        sigmoid_values = torch.tensor(
            sigmoid(cell_data.cpu().numpy(), threshold[None, :], exponent[None, :]),
            device=device_obj,
            dtype=torch.float32
        )
        sigmoid_prime = (
            torch.tensor(exponent, device=device_obj, dtype=torch.float32)
            * sigmoid_values
            * (1 - sigmoid_values)
            / torch.where(cell_data == 0, torch.ones_like(cell_data), cell_data)
        )

        for local_idx, global_idx in enumerate(tqdm(cluster_indices, desc=f"Cluster {cluster}")):
            sig_prime_val = sigmoid_prime[local_idx]
            jac_f = W * sig_prime_val.view(-1, 1) - gamma

            # Compute antisymmetric part: A = 0.5 * (J - J^T)
            A = 0.5 * (jac_f - jac_f.T)

            # Frobenius norm
            rotational_norms[global_idx] = torch.norm(A).cpu().item()

    adata.obs['jacobian_rotational'] = rotational_norms

    return adata if copy else None
