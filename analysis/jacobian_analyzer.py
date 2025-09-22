"""
Jacobian analysis functionality for scHopfield package.
Contains the JacobianAnalyzer class for computing and analyzing Jacobian matrices.
"""

import numpy as np
import pandas as pd
import torch
import scipy.linalg
from typing import Optional, Tuple, Dict, Any
from tqdm import tqdm

from ..core.base_models import BaseAnalyzer
from ..utils.utilities import to_numpy, sigmoid


class JacobianAnalyzer(BaseAnalyzer):
    """
    Analyzer for computing and analyzing Jacobian matrices of the dynamical system.

    This class provides methods for computing Jacobian matrices at each cell's state,
    calculating eigenvalues and eigenvectors, and analyzing stability properties.
    """

    def __init__(self, landscape_analyzer):
        """
        Initialize the JacobianAnalyzer.

        Args:
            landscape_analyzer: A fitted LandscapeAnalyzer instance
        """
        super().__init__(landscape_analyzer.adata)
        self.analyzer = landscape_analyzer

        # Initialize storage for results
        self.jacobians = None
        self.eigenvalues = None
        self.eigenvectors = None

    def compute(self, **kwargs) -> Dict[str, Any]:
        """
        Compute method required by BaseAnalyzer.

        Returns:
            Dictionary containing computation results
        """
        self.compute_jacobians(**kwargs)
        return {'jacobians_computed': True}

    def compute_jacobians(self, save_to_disk: bool = False, save_dir: Optional[str] = None,
                         compute_eigenvectors: bool = False, device: str = 'cpu',
                         use_chunking: bool = True, chunk_size: int = 100) -> None:
        """
        Compute Jacobian matrices for all cells in the dataset.

        Args:
            save_to_disk: Whether to save results to disk using memory mapping
            save_dir: Directory to save results (required if save_to_disk=True)
            compute_eigenvectors: Whether to compute eigenvectors in addition to eigenvalues
            device: Device to use for computation ('cpu' or 'cuda')
            use_chunking: Whether to use chunked processing for efficiency
            chunk_size: Size of chunks when use_chunking=True
        """
        if use_chunking:
            self._compute_jacobians_chunked(compute_eigenvectors=compute_eigenvectors,
                                           device=device, chunk_size=chunk_size)
        else:
            self._compute_jacobians_original(save_to_disk=save_to_disk, save_dir=save_dir,
                                           compute_eigenvectors=compute_eigenvectors, device=device)

    def _compute_jacobians_chunked(self, compute_eigenvectors: bool = False,
                                  device: str = 'cpu', chunk_size: int = 100) -> None:
        """
        Compute Jacobian matrices using chunked processing for efficiency.
        """
        print("Computing Jacobian matrices in chunks...")

        expression_data = self.analyzer.get_matrix(self.analyzer.spliced_matrix_key, genes=self.analyzer.genes)
        expression_data = torch.tensor(expression_data, dtype=torch.float32)
        n_cells, n_genes = expression_data.shape

        clusters = np.array(self.adata.obs[self.analyzer.cluster_key])
        unique_clusters = np.unique(clusters)

        # Prepare output arrays on CPU or disk: small buffers sufficient since batch-wise processing
        self.jacobians = np.zeros((n_cells, n_genes, n_genes), dtype=np.float64)
        self.eigenvalues = np.zeros((n_cells, n_genes), dtype=np.complex128)
        if compute_eigenvectors:
            self.eigenvectors = np.zeros((n_cells, n_genes, n_genes), dtype=np.complex128)

        exponent = self.analyzer.exponent
        threshold = self.analyzer.threshold

        for cluster in unique_clusters:
            W = torch.tensor(self.analyzer.W[cluster], dtype=torch.float32).to(device)
            if self.analyzer.refit_gamma:
                gamma = torch.tensor(self.analyzer.gamma[cluster], dtype=torch.float32).to(device)
            else:
                gamma = torch.tensor(self.analyzer.adata.var[self.analyzer.gamma_key][self.analyzer.genes].values, dtype=torch.float32).to(device)

            cluster_idx = np.where(clusters == cluster)[0]
            expr_cluster = expression_data[cluster_idx].to(device)

            # Compute sigmoid and derivative in batch
            sig = 1 / (1 + torch.exp(-exponent * (expr_cluster - threshold)))
            dsig_dx = sig * (1 - sig) * exponent  # shape (num_cells_in_cluster, n_genes)

            num_cells_cluster = len(cluster_idx)

            for start in tqdm(range(0, num_cells_cluster, chunk_size), desc=f"Cluster {cluster}"):
                end = min(start + chunk_size, num_cells_cluster)
                batch_idx = slice(start, end)

                dsig_chunk = dsig_dx[batch_idx]  # (chunk_size, n_genes)

                # Compute batch Jacobians: W * diag(dsig_dx) - diag(gamma)
                # Broadcasting: (chunk_size, n_genes, n_genes)
                jac_chunk = W.unsqueeze(0) * dsig_chunk.unsqueeze(2) - torch.diag(gamma).unsqueeze(0)

                # Transfer to CPU as numpy for storage and eig computation if CPU only
                jac_chunk_cpu = jac_chunk.cpu().numpy()

                # Store Jacobians
                self.jacobians[cluster_idx[start:end], :, :] = jac_chunk_cpu

                # Compute eigenvalues (and vectors optionally)
                if compute_eigenvectors:
                    for i, jac in enumerate(jac_chunk_cpu):
                        vals, vecs = scipy.linalg.eig(jac)
                        self.eigenvalues[cluster_idx[start + i]] = vals
                        self.eigenvectors[cluster_idx[start + i]] = vecs
                else:
                    for i, jac in enumerate(jac_chunk_cpu):
                        vals = scipy.linalg.eigvals(jac)
                        self.eigenvalues[cluster_idx[start + i]] = vals

                # Free GPU cache
                if device == 'cuda':
                    torch.cuda.empty_cache()

        print(f"Completed computation for {n_cells} cells.")

    def _compute_jacobians_original(self, save_to_disk: bool = False, save_dir: Optional[str] = None,
                                   compute_eigenvectors: bool = False, device: str = 'cpu') -> None:
        """
        Original implementation - compute Jacobian matrices for all cells in the dataset.

        Args:
            save_to_disk: Whether to save results to disk using memory mapping
            save_dir: Directory to save results (required if save_to_disk=True)
            compute_eigenvectors: Whether to compute eigenvectors in addition to eigenvalues
            device: Device to use for computation ('cpu' or 'cuda')
        """
        print("Computing Jacobian matrices for all cells...")

        n_genes = len(self.analyzer.genes)
        n_cells = self.adata.n_obs

        if save_to_disk:
            if save_dir is None:
                raise ValueError("save_dir must be provided when save_to_disk=True")
            # Use np.memmap for disk-based storage
            self.jacobians = np.memmap(f'{save_dir}/jacobians.dat', dtype=np.float64,
                                     mode='w+', shape=(n_cells, n_genes, n_genes))
            self.eigenvalues = np.memmap(f'{save_dir}/jacobian_eigenvalues.dat', dtype=np.complex128,
                                       mode='w+', shape=(n_cells, n_genes))
            if compute_eigenvectors:
                self.eigenvectors = np.memmap(f'{save_dir}/jacobian_eigenvectors.dat', dtype=np.complex128,
                                            mode='w+', shape=(n_cells, n_genes, n_genes))
        else:
            # Use regular numpy arrays in memory
            self.jacobians = np.zeros((n_cells, n_genes, n_genes), dtype=np.float64)
            self.eigenvalues = np.zeros((n_cells, n_genes), dtype=np.complex128)
            if compute_eigenvectors:
                self.eigenvectors = np.zeros((n_cells, n_genes, n_genes), dtype=np.complex128)

        # Get expression data
        expression_data = to_numpy(self.analyzer.get_matrix(self.analyzer.spliced_matrix_key, genes=self.analyzer.genes))

        # Compute Jacobians for each cell
        for cell_idx in tqdm(range(n_cells), desc="Computing Jacobians"):
            cell_state = expression_data[cell_idx, :]
            jacobian = self.jacobian_for_cell(cell_state, cell_idx)

            self.jacobians[cell_idx] = jacobian

            # Compute eigenvalues (and eigenvectors if requested)
            if compute_eigenvectors:
                eigenvals, eigenvecs = np.linalg.eig(jacobian)
                self.eigenvalues[cell_idx] = eigenvals
                self.eigenvectors[cell_idx] = eigenvecs
            else:
                eigenvals = np.linalg.eigvals(jacobian)
                self.eigenvalues[cell_idx] = eigenvals

        print(f"Computed Jacobians and eigenvalues for {n_cells} cells")

    def jacobian_for_cell(self, cell_state: np.ndarray, cell_idx: int) -> np.ndarray:
        """
        Compute the Jacobian matrix for a specific cell.

        Args:
            cell_state: Expression state of the cell
            cell_idx: Index of the cell (used to determine cluster)

        Returns:
            Jacobian matrix for the cell
        """
        # Determine which cluster this cell belongs to
        cluster = self.adata.obs[self.analyzer.cluster_key].iloc[cell_idx]

        # Get model parameters for this cluster
        W = self.analyzer.W[cluster]
        gamma = (self.analyzer.adata.var[self.analyzer.gamma_key][self.analyzer.genes].values
                if not self.analyzer.refit_gamma else self.analyzer.gamma[cluster])

        # Compute sigmoid activation and its derivative
        sig = sigmoid(cell_state, self.analyzer.threshold, self.analyzer.exponent)

        # Compute derivative of sigmoid: d(sigmoid)/dx = sig * (1 - sig) * exponent
        dsig_dx = sig * (1 - sig) * self.analyzer.exponent

        # Jacobian = W * diag(dsig_dx) - diag(gamma)
        jacobian = W * dsig_dx[None, :] - np.diag(gamma)

        return jacobian

    def get_stability_stats(self) -> pd.DataFrame:
        """
        Compute stability statistics for each cluster.

        Returns:
            DataFrame with stability statistics per cluster
        """
        if self.eigenvalues is None:
            raise ValueError("Eigenvalues not computed yet. Run compute_jacobians() first.")

        stats = []
        clusters = self.adata.obs[self.analyzer.cluster_key].unique()

        for cluster in clusters:
            cluster_mask = self.adata.obs[self.analyzer.cluster_key] == cluster
            cluster_eigenvals = self.eigenvalues[cluster_mask]

            # Compute statistics
            n_cells = cluster_mask.sum()
            real_parts = np.real(cluster_eigenvals)
            imag_parts = np.imag(cluster_eigenvals)

            stats.append({
                'cluster': cluster,
                'n_cells': n_cells,
                'mean_real_eigenval': np.mean(real_parts),
                'std_real_eigenval': np.std(real_parts),
                'mean_positive_eigenvals': np.mean(np.sum(real_parts > 0, axis=1)),
                'mean_negative_eigenvals': np.mean(np.sum(real_parts < 0, axis=1)),
                'mean_abs_imag': np.mean(np.abs(imag_parts)),
                'stable_cells_fraction': np.mean(np.all(real_parts < 0, axis=1)),
                'unstable_cells_fraction': np.mean(np.any(real_parts > 0, axis=1))
            })

        return pd.DataFrame(stats)