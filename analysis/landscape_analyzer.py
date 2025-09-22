"""
Landscape analysis functionality for scHopfield package.
Contains the main LandscapeAnalyzer class that implements energy landscape analysis.
"""

import anndata
from collections import defaultdict
import itertools
import numpy as np
import pandas as pd
import pickle
from scipy.spatial.distance import cdist
import torch
from tqdm import tqdm
from typing import Union, List, Optional, Tuple, Dict, Any

from ..core.base_models import BaseAnalyzer, ValidationMixin
from ..utils.utilities import to_numpy, sigmoid, fit_sigmoid, soften, int_sig_act_inv
from .energy_calculator import EnergyCalculator


class LandscapeAnalyzer(BaseAnalyzer, ValidationMixin):
    """
    Main class for energy landscape analysis using Hopfield-like dynamics.

    This class implements comprehensive energy landscape analysis for single-cell
    trajectory data, including sigmoid fitting, interaction matrix inference,
    and energy calculations.
    """

    def __init__(self,
                 data: anndata.AnnData,
                 spliced_matrix_key: str = 'Ms',
                 velocity_key: str = 'velocity_S',
                 degradation_key: str = 'gamma',
                 genes: Union[None, List[str], List[bool], List[int]] = None,
                 cluster_key: Union[None, str] = None,
                 w_threshold: float = 1e-5,
                 w_scaffold: Union[None, np.ndarray] = None,
                 scaffold_regularization: float = 1.0,
                 only_TFs: bool = False,
                 infer_I: bool = False,
                 refit_gamma: bool = False,
                 pre_initialize_W: bool = False,
                 criterion: str = 'L2',
                 batch_size: int = 64,
                 n_epochs: int = 1000,
                 device: str = 'cpu',
                 use_scheduler: bool = False,
                 scheduler_kws: dict = {},
                 get_plots: bool = False,
                 skip_all: bool = False,
                 manual_fit: bool = False):

        super().__init__(data)

        # Store configuration parameters
        self.adata = data
        self.spliced_matrix_key = spliced_matrix_key
        self.velocity_key = velocity_key
        self.gamma_key = degradation_key
        self.genes = self.gene_parser(genes)
        self.gene_names = self.adata.var.index[self.genes]
        self.cluster_key = cluster_key
        self.clusters = self.adata.obs[self.cluster_key].unique() if self.cluster_key else []

        # Store fitting parameters for later use
        self.w_threshold = w_threshold
        self.w_scaffold = w_scaffold
        self.scaffold_regularization = scaffold_regularization
        self.only_TFs = only_TFs
        self.infer_I = infer_I
        self.refit_gamma = refit_gamma
        self.pre_initialize_W = pre_initialize_W
        self.criterion = criterion
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device
        self.use_scheduler = use_scheduler
        self.scheduler_kws = scheduler_kws
        self.get_plots = get_plots
        self.skip_all = skip_all

        # Initialize empty results containers
        self.threshold = None
        self.exponent = None
        self.offset = None
        self.W = {}
        self.I = {}
        self.E = {}
        self.E_int = {}
        self.E_deg = {}
        self.E_bias = {}

        # Initialize energy calculator
        self.energy_calculator = EnergyCalculator(self)

        print(f"LandscapeAnalyzer initialized with {len(self.genes)} genes and {len(self.clusters)} clusters")
        print("Call fit_sigmoids() to start the analysis pipeline")

    def get_matrix(self, key: str, genes: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Retrieve a specific matrix from the AnnData object based on the given key.

        Parameters:
            key: Key for the desired matrix in the AnnData layers.
            genes: Array of gene indices to subset the matrix.

        Returns:
            The requested matrix, optionally subset by genes.
        """
        if genes is None:
            return self.adata.layers[key]
        else:
            return self.adata.layers[key][:, genes]

    def write_property(self, key: str, value: np.ndarray) -> None:
        """
        Write a property (value) to the AnnData object under the specified key.

        Parameters:
            key: Key under which to store the value.
            value: The value to be stored.
        """
        shape = np.shape(value)

        # Scalar or 1D array
        if len(shape) == 1:
            if shape[0] == self.adata.n_obs:
                self.adata.obs[key] = value
            elif shape[0] == self.adata.n_vars:
                self.adata.var[key] = value
            else:
                self.adata.uns[key] = value

        # 2D array
        elif len(shape) == 2:
            if shape[0] == self.adata.n_vars:
                if shape[1] == self.adata.n_vars:
                    self.adata.varp[key] = value
                else:
                    self.adata.varm[key] = value
            elif shape[0] == self.adata.n_obs:
                if shape[1] == self.adata.n_vars:
                    self.adata.layers[key] = value
                elif shape[1] == self.adata.n_obs:
                    self.adata.obsp[key] = value
                else:
                    self.adata.obsm[key] = value
            else:
                self.adata.uns[key] = value

        # Other
        else:
            self.adata.uns[key] = value

    def fit_sigmoids(self, min_th: float = 0.05) -> None:
        """
        Fit sigmoid functions to gene expression data for all genes.

        Args:
            min_th: Threshold for zero expression in percentage of maximum expression.
        """
        print("Fitting sigmoid functions to gene expression data...")

        # Retrieve expression data for all genes
        x = to_numpy(self.get_matrix(self.spliced_matrix_key, genes=self.genes).T)

        # Apply the fitting function to each gene's expression data
        results = np.array([fit_sigmoid(g, min_th=min_th) for g in x])

        # Unpack fitting results into separate attributes
        self.threshold, self.exponent, self.offset, self.sigmoid_mse = results.T

        print(f"Successfully fitted sigmoids for {len(self.genes)} genes")

    def write_sigmoids(self) -> None:
        """Write computed sigmoid values to the AnnData object."""
        sig = self.get_sigmoid()
        sigmoids = np.zeros(list(self.adata.layers.values())[0].shape, dtype=sig.dtype)
        sigmoids[:, self.genes] = sig
        self.write_property('sigmoid', sigmoids)
        self.sigmoids = self.adata.layers['sigmoid']

    def get_sigmoid(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute the sigmoid activation for the given input x or for the entire spliced matrix.

        Args:
            x: Input data for which to compute the sigmoid function.

        Returns:
            The sigmoid activation applied to the input data.
        """
        # Use the entire spliced matrix if x is not provided
        if x is None:
            x = to_numpy(self.get_matrix(self.spliced_matrix_key, genes=self.genes))

        # Compute the sigmoid function of x using the class's threshold and exponent parameters
        sigmoid_output = np.nan_to_num(sigmoid(x, self.threshold[None, :], self.exponent[None, :]))

        return sigmoid_output

    def gene_parser(self, genes: Union[None, List[str], List[bool], List[int]]) -> np.ndarray:
        """
        Parses the given gene list into indices that indicate which genes to use.

        Args:
            genes: List of gene names, indices, or Boolean values, or None.

        Returns:
            Array of indices indicating which genes to use.
        """
        if genes is None:
            # If genes is None, use all genes
            return np.arange(self.adata.n_vars)

        if isinstance(genes[0], str):
            # If the first element is a string, assume genes is a list of gene names
            gene_indices = self.adata.var.index.get_indexer_for(genes)
            # Check for -1 in gene_indices which indicates gene name not found
            if np.any(gene_indices == -1):
                missing_genes = np.array(genes)[gene_indices == -1]
                raise ValueError(f"Gene names not found in adata.var.index: {missing_genes}")
            return gene_indices
        elif isinstance(genes[0], (int, np.int64, np.int32, np.int16, np.int8)):
            # If the first element is an int, assume genes is a list of gene indices
            return np.array(genes)
        elif isinstance(genes[0], (bool, np.bool_)):
            # If the first element is a bool, assume genes is a Boolean mask
            if len(genes) != self.adata.n_vars:
                raise ValueError("Boolean gene list must have the same length as the number of genes in the dataset.")
            return np.where(genes)[0]
        else:
            raise ValueError("Genes argument must be None, a list of gene names, indices, or a Boolean mask.")

    def compute(self, **kwargs) -> Dict[str, Any]:
        """
        Compute the complete energy landscape analysis.

        This method implements the abstract method from BaseAnalyzer and runs
        the complete analysis pipeline.

        Returns:
            Dictionary containing analysis results
        """
        # Fit sigmoid functions to expression data
        self.fit_sigmoids()

        # Write sigmoids to the adata object
        self.write_sigmoids()

        # Return basic results (this is a placeholder implementation)
        return {
            'fitted_sigmoids': True,
            'analysis_complete': True
        }

    def fit_interactions(self, **kwargs) -> None:
        """
        Fit interaction matrices W and bias vectors I.

        This is a placeholder method that should be implemented with the
        actual interaction fitting logic.
        """
        print("Fitting interaction matrices...")

        # Initialize interaction matrices and bias vectors for each cluster
        for cluster in self.clusters:
            self.W[cluster] = np.random.randn(len(self.genes), len(self.genes)) * 0.1
            self.I[cluster] = np.random.randn(len(self.genes)) * 0.1

        print(f"Initialized interaction matrices for {len(self.clusters)} clusters")

    def get_energies(self) -> None:
        """
        Compute energy values for each cell and cluster.
        """
        print("Computing energies using EnergyCalculator...")

        # Use the energy calculator to compute all energy components
        self.energy_calculator.get_energies()

        # Update E_int, E_deg, E_bias to match the expected attribute names
        self.E_int = self.E_interaction
        self.E_deg = self.E_degradation
        # E_bias is already correct from the energy calculator

        print(f"Computed energies for {len(self.clusters)} clusters")

    def write_energies(self) -> None:
        """
        Write computed energy values to the AnnData object.
        """
        if not self.E:
            print("No energies computed yet. Run get_energies() first.")
            return

        print("Writing energies to adata.obs...")

        # Use the energy calculator to write energies
        self.energy_calculator.write_energies()

        print("Energy values written to adata.obs")

    def energy_genes_correlation(self) -> None:
        """
        Compute correlations between energy values and gene expressions.
        """
        print("Computing energy-gene correlations...")

        # Initialize an array to hold energies for all observations
        energies = np.zeros((4, self.adata.n_obs))

        # Initialize correlation dictionaries
        self.correlation = {}
        self.correlation_interaction = {}
        self.correlation_degradation = {}
        self.correlation_bias = {}

        # Loop through each cluster
        for k in self.W.keys():
            if k == 'all':
                continue
            else:
                cells = self.adata.obs[self.cluster_key] == k

            # Assign computed energies for the current cluster to the energies array
            energies[0, cells] = self.E.get(k, np.zeros(sum(cells)))
            energies[1, cells] = self.E_interaction.get(k, np.zeros(sum(cells)))
            energies[2, cells] = self.E_degradation.get(k, np.zeros(sum(cells)))
            energies[3, cells] = self.E_bias.get(k, np.zeros(sum(cells)))

            # Extract expression data for the current cluster
            X = to_numpy(self.adata.layers[self.spliced_matrix_key][cells][:, self.genes].T)

            # Compute correlations between energies and gene expression
            correlations = np.nan_to_num(np.corrcoef(np.vstack((energies[:, cells], X)))[:4, 4:])

            # Store computed correlations in their respective dictionaries
            self.correlation[k], self.correlation_interaction[k], self.correlation_degradation[k], self.correlation_bias[k] = correlations

        # Compute correlations for all cells
        X = to_numpy(self.adata.layers[self.spliced_matrix_key][:, self.genes].T)
        correlations = np.nan_to_num(np.corrcoef(np.vstack((energies, X)))[:4, 4:])
        self.correlation['all'], self.correlation_interaction['all'], self.correlation_degradation['all'], self.correlation_bias['all'] = correlations

        print(f"Computed correlations for {len(self.clusters)} clusters")

    def celltype_correlation(self, modified: bool = True, all_genes: bool = False) -> None:
        """
        Compute correlations between cell types based on their gene expression profiles.

        Args:
            modified: Whether to use modified correlation (currently uses standard correlation)
            all_genes: If True, considers all genes; if False, only considers self.genes
        """
        print("Computing cell type correlations...")

        # Retrieve unique cell types from the data
        keys = self.adata.obs[self.cluster_key].unique()

        # Initialize a DataFrame to hold the correlation coefficients
        rv = pd.DataFrame(index=keys, columns=keys, data=1.0)

        # Determine the set of genes to consider
        genes_to_consider = None if all_genes else self.genes

        # Retrieve expression data for the chosen genes
        counts = self.get_matrix(self.spliced_matrix_key, genes=genes_to_consider)

        # Compute pairwise correlations between cell types
        for k1, k2 in itertools.combinations(keys, 2):
            # Get cells belonging to each cluster
            cells1 = self.adata.obs[self.cluster_key] == k1
            cells2 = self.adata.obs[self.cluster_key] == k2

            # Extract expression data for each cluster
            X1 = to_numpy(counts[cells1].T)  # genes x cells
            X2 = to_numpy(counts[cells2].T)  # genes x cells

            # Compute mean expression profiles
            mean1 = np.mean(X1, axis=1)
            mean2 = np.mean(X2, axis=1)

            # Compute correlation between mean expression profiles
            if len(mean1) > 1 and len(mean2) > 1:
                corr = np.corrcoef(mean1, mean2)[0, 1]
                corr = np.nan_to_num(corr, nan=0.0)
            else:
                corr = 0.0

            # Store symmetric correlation values
            rv.loc[k1, k2] = rv.loc[k2, k1] = corr

        self.cells_correlation = rv
        print("Cell type correlations computed")

    def network_correlations(self) -> None:
        """
        Compute various correlations and distances between the interaction networks of different cell types.
        The interaction networks are represented by the interaction matrices W for each cell type.

        Updates:
        - self.jaccard: DataFrame containing Jaccard indices between cell types.
        - self.hamming: DataFrame containing Hamming distances between cell types.
        - self.euclidean: DataFrame containing Euclidean distances between cell types.
        - self.pearson: DataFrame containing Pearson correlations between cell types.
        - self.pearson_bin: DataFrame containing Pearson correlations between binary representations of cell types.
        - self.mean_col_corr: DataFrame containing mean column-wise Pearson correlations between cell types.
        - self.singular: DataFrame containing distances based on singular values between cell types.
        """
        print("Computing network correlations...")

        # Retrieve unique cell types from the data
        keys = self.adata.obs[self.cluster_key].unique()

        # Initialize DataFrames to hold the computed metrics
        jaccard, hamming, pearson, pearson_bin, euclidean, mean_col, singular = \
            [pd.DataFrame(index=keys, columns=keys, data=d) for d in [1., 0., 1., 1., 0., 1., 0.]]

        # Compute singular values for each interaction matrix
        svs = {k: np.linalg.svd(self.W[k], compute_uv=False) for k in keys}

        # Compute pairwise metrics between cell types
        for k1, k2 in itertools.combinations(keys, 2):
            w1, w2 = self.W[k1], self.W[k2]
            bw1, bw2 = np.sign(w1), np.sign(w2)

            # Pearson correlation
            pearson.loc[k1, k2] = pearson.loc[k2, k1] = np.corrcoef(w1.ravel(), w2.ravel())[0, 1]

            # Pearson correlation for binary representations
            pearson_bin.loc[k1, k2] = pearson_bin.loc[k2, k1] = np.corrcoef(bw1.ravel(), bw2.ravel())[0, 1]

            # Euclidean distance
            euclidean.loc[k1, k2] = euclidean.loc[k2, k1] = np.linalg.norm(w1 - w2)

            # Hamming distance
            hamming.loc[k1, k2] = hamming.loc[k2, k1] = np.count_nonzero(bw1 != bw2)

            # Jaccard index
            intersection = np.logical_and(bw1, bw2)
            union = np.logical_or(bw1, bw2)
            jaccard.loc[k1, k2] = jaccard.loc[k2, k1] = intersection.sum() / union.sum() if union.sum() > 0 else 0.0

            # Mean column-wise Pearson correlation
            mean_col_corr = np.mean(np.diag(np.corrcoef(w1, w2, rowvar=False)[:w1.shape[0], :w1.shape[0]]))
            mean_col.loc[k1, k2] = mean_col.loc[k2, k1] = np.nan_to_num(mean_col_corr, nan=0.0)

            # Distance based on singular values
            sv1, sv2 = svs[k1], svs[k2]
            min_len = min(len(sv1), len(sv2))
            sv_distance = np.linalg.norm(sv1[:min_len] - sv2[:min_len])
            singular.loc[k1, k2] = singular.loc[k2, k1] = sv_distance

        # Clean up NaN values and store results
        self.jaccard = jaccard.fillna(1.0)
        self.hamming = hamming.fillna(0.0)
        self.euclidean = euclidean.fillna(0.0)
        self.pearson = pearson.fillna(1.0)
        self.pearson_bin = pearson_bin.fillna(1.0)
        self.mean_col_corr = mean_col.fillna(1.0)
        self.singular = singular.fillna(0.0)

        print("Network correlations computed")

    def plot_high_correlation_genes(self, top_n: int = 10, energy: str = 'total',
                                   cluster: str = 'all', absolute: bool = False,
                                   basis: str = 'umap', figsize: tuple = (15, 10)) -> None:
        """
        Plot genes with high correlation to energy values.

        Args:
            top_n: Number of top genes to plot
            energy: Type of energy correlation to use
            cluster: Cluster to analyze
            absolute: Whether to use absolute correlation values
            basis: Embedding basis for plotting
            figsize: Figure size
        """
        print(f"Plotting top {top_n} correlated genes for {energy} energy...")

        # This is a placeholder - would need actual plotting implementation
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f'Top {top_n} {energy} energy correlated genes\n(placeholder plot)',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'High Correlation Genes - {energy.title()} Energy')
        plt.show()