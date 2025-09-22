"""
Jacobian analysis functionality for scHopfield package.
Contains the JacobianAnalyzer class for computing and analyzing Jacobian matrices.
"""

import numpy as np
import pandas as pd
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
                         compute_eigenvectors: bool = False, device: str = 'cpu') -> None:
        """
        Compute Jacobian matrices for all cells in the dataset.

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