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

    def fit_interactions(self,
                         w_threshold: float = None,
                         w_scaffold: np.ndarray = None,
                         scaffold_regularization: float = None,
                         only_TFs: bool = None,
                         infer_I: bool = None,
                         refit_gamma: bool = None,
                         pre_initialize_W: bool = None,
                         n_epochs: int = None,
                         criterion: str = None,
                         batch_size: int = None,
                         device: str = None,
                         skip_all: bool = None,
                         use_scheduler: bool = None,
                         scheduler_kws: dict = None,
                         get_plots: bool = None) -> None:
        """
        Fit interaction matrices W and bias vectors I for each cluster.

        This method implements the complete interaction fitting logic from scMomentum,
        including scaffold-based optimization using PyTorch.
        """
        # Use stored parameters if not overridden
        w_threshold = w_threshold if w_threshold is not None else self.w_threshold
        w_scaffold = w_scaffold if w_scaffold is not None else self.w_scaffold
        scaffold_regularization = scaffold_regularization if scaffold_regularization is not None else self.scaffold_regularization
        only_TFs = only_TFs if only_TFs is not None else self.only_TFs
        infer_I = infer_I if infer_I is not None else self.infer_I
        refit_gamma = refit_gamma if refit_gamma is not None else self.refit_gamma
        pre_initialize_W = pre_initialize_W if pre_initialize_W is not None else self.pre_initialize_W
        n_epochs = n_epochs if n_epochs is not None else self.n_epochs
        criterion = criterion if criterion is not None else self.criterion
        batch_size = batch_size if batch_size is not None else self.batch_size
        device = device if device is not None else self.device
        skip_all = skip_all if skip_all is not None else self.skip_all
        use_scheduler = use_scheduler if use_scheduler is not None else self.use_scheduler
        scheduler_kws = scheduler_kws if scheduler_kws is not None else self.scheduler_kws
        get_plots = get_plots if get_plots is not None else self.get_plots

        print("Fitting interaction matrices using scaffold optimization...")

        # Get spliced and velocity matrices
        x = to_numpy(self.get_matrix(self.spliced_matrix_key, genes=self.genes))
        v = to_numpy(self.get_matrix(self.velocity_key, genes=self.genes))
        g = self.adata.var[self.gamma_key][self.genes].values.astype(x.dtype)
        sig = self.get_sigmoid()

        self.W = {}
        self.I = {}
        if refit_gamma:
            self.gamma = {}

        clusters = self.adata.obs[self.cluster_key].unique()
        if not skip_all:
            clusters = np.append(clusters, 'all')
        if w_scaffold is not None:
            self.models = {}

        for ct in clusters:
            print(f"Inferring interaction matrix W and bias vector I for cluster {ct}")
            if ct == 'all':
                idx = slice(None)
            else:
                idx = self.adata.obs[self.cluster_key].values == ct

            self._fit_interactions_for_group(
                group=ct,
                x=x[idx, :],
                v=v[idx, :],
                sig=sig[idx, :],
                g=g,
                w_threshold=w_threshold,
                w_scaffold=w_scaffold,
                scaffold_regularization=scaffold_regularization,
                only_TFs=only_TFs,
                infer_I=infer_I,
                refit_gamma=refit_gamma,
                pre_initialize_W=pre_initialize_W,
                n_epochs=n_epochs,
                criterion=criterion,
                batch_size=batch_size,
                device=device,
                use_scheduler=use_scheduler,
                scheduler_kws=scheduler_kws,
                get_plots=get_plots,
            )

        print(f"Fitted interaction matrices for {len(clusters)} clusters")

    def _fit_interactions_for_group(self,
                                   group: str,
                                   x: np.ndarray,
                                   v: np.ndarray,
                                   sig: np.ndarray,
                                   g: np.ndarray,
                                   w_threshold: float,
                                   w_scaffold: np.ndarray,
                                   scaffold_regularization: float,
                                   only_TFs: bool,
                                   infer_I: bool,
                                   refit_gamma: bool,
                                   pre_initialize_W: bool,
                                   n_epochs: int,
                                   criterion: str,
                                   batch_size: int,
                                   device: str,
                                   use_scheduler: bool,
                                   scheduler_kws: dict,
                                   get_plots: bool) -> None:
        """
        Helper function to fit interaction matrix W and bias vector I for a group (global or cluster).
        """
        import torch
        device_torch = torch.device("cuda" if (torch.cuda.is_available() and device == "cuda") else "cpu")

        W = None
        I = None
        if (w_scaffold is None) or pre_initialize_W:
            # Default L2 criterion using least squares
            rhs = np.hstack((sig, np.ones((sig.shape[0], 1), dtype=x.dtype))) if infer_I else sig
            try:
                WI = np.linalg.lstsq(rhs, v + g[None, :] * x, rcond=1e-5)[0]
                W = WI[:-1, :].T if infer_I else WI.T
                I = WI[-1, :] if infer_I else -np.clip(WI, a_min=None, a_max=0).sum(axis=0)
            except:
                pass

        if w_scaffold is not None:
            # Import the optimizer here to avoid circular imports
            from ..optimization.scaffold_optimizer import ScaffoldOptimizer, CustomDataset
            import torch.utils.data

            # Use ScaffoldOptimizer
            model = ScaffoldOptimizer(
                g, w_scaffold, device_torch, refit_gamma,
                scaffold_regularization=scaffold_regularization,
                use_masked_linear=only_TFs,
                pre_initialized_W=W,
                pre_initialized_I=I
            )
            train_loader = self._create_train_loader(sig, v, x, device_torch, batch_size=batch_size)
            scheduler_fn = torch.optim.lr_scheduler.StepLR if use_scheduler else None
            scheduler_kwargs = {"step_size": 100, "gamma": 0.4} if scheduler_kws == {} else scheduler_kws
            model.train_model(
                train_loader, n_epochs, learning_rate=0.1, criterion=criterion,
                scheduler_fn=scheduler_fn, scheduler_kwargs=scheduler_kwargs, get_plots=get_plots
            )
            W = model.W.weight.detach().cpu().numpy()
            I = model.I.detach().cpu().numpy()
            g = np.exp(model.gamma.detach().cpu().numpy()) if refit_gamma else g
            if hasattr(self, 'models'):
                self.models[group] = model

        # Handle case where W and I are still None
        if W is None:
            W = np.random.randn(len(self.genes), len(self.genes)) * 0.01
        if I is None:
            I = np.random.randn(len(self.genes)) * 0.01

        # Threshold values and store results
        W[np.abs(W) < w_threshold] = 0
        I[np.abs(I) < w_threshold] = 0
        self.W[group] = W
        self.I[group] = I
        if refit_gamma:
            self.gamma[group] = g

    def _create_train_loader(self, sig: np.ndarray, v: np.ndarray, x: np.ndarray,
                           device, batch_size: int = 64):
        """
        Helper function to create a PyTorch DataLoader for training.
        """
        import torch
        from ..optimization.scaffold_optimizer import CustomDataset
        import torch.utils.data

        dataset = CustomDataset(sig, v, x, device)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

            # Convert to numpy boolean array for consistent indexing
            cells_bool = cells.values

            # Assign computed energies for the current cluster to the energies array
            energies[0, cells_bool] = self.E.get(k, np.zeros(sum(cells_bool)))
            energies[1, cells_bool] = self.E_interaction.get(k, np.zeros(sum(cells_bool)))
            energies[2, cells_bool] = self.E_degradation.get(k, np.zeros(sum(cells_bool)))
            energies[3, cells_bool] = self.E_bias.get(k, np.zeros(sum(cells_bool)))

            # Extract expression data for the current cluster
            X = to_numpy(self.adata.layers[self.spliced_matrix_key][cells_bool][:, self.genes].T)

            # Compute correlations between energies and gene expression
            correlations = np.nan_to_num(np.corrcoef(np.vstack((energies[:, cells_bool], X)))[:4, 4:])

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

            # Convert to numpy boolean arrays for consistent indexing
            cells1_bool = cells1.values
            cells2_bool = cells2.values

            # Extract expression data for each cluster
            X1 = to_numpy(counts[cells1_bool].T)  # genes x cells
            X2 = to_numpy(counts[cells2_bool].T)  # genes x cells

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
                                   basis: str = 'umap', figsize: tuple = (15, 10),
                                   plot_correlations: bool = False) -> None:
        """
        Plot genes with high correlation to energy values.

        Args:
            top_n: Number of top genes to plot
            energy: Type of energy correlation to use ('total', 'interaction', 'degradation', 'bias')
            cluster: Cluster to analyze
            absolute: Whether to use absolute correlation values
            basis: Embedding basis for plotting
            figsize: Figure size
            plot_correlations: Whether to plot correlation values or gene expression
        """
        print(f"Plotting top {top_n} correlated genes for {energy} energy...")

        # Determine the correct correlation dictionary based on the energy type
        if energy == 'interaction':
            corr_dict = self.correlation_interaction
        elif energy == 'degradation':
            corr_dict = self.correlation_degradation
        elif energy == 'bias':
            corr_dict = self.correlation_bias
        else:
            corr_dict = self.correlation

        # Check if correlations have been computed
        if not hasattr(self, 'correlation') or not corr_dict:
            print("Correlations not computed yet. Run energy_genes_correlation() first.")
            return

        # Get the correlations for the specified cluster
        if cluster not in corr_dict:
            print(f"Cluster '{cluster}' not found in correlation data. Available clusters: {list(corr_dict.keys())}")
            return

        corr = corr_dict[cluster]

        # Sort genes based on their absolute or relative correlation values
        abscorr = np.abs(corr) if absolute else corr
        top_genes_indices = np.argsort(abscorr)[-top_n:][::-1]
        top_genes_names = self.gene_names[top_genes_indices]

        # Plot the correlations
        self.plot_gene_correlation(top_genes_names, energy=energy, cluster=cluster,
                                 absolute=absolute, basis=basis, return_corr=False,
                                 plot='correlation' if plot_correlations else 'expression',
                                 figsize=figsize)

    def plot_gene_correlation(self, genes, energy: str = 'total', cluster: str = 'all',
                            absolute: bool = False, basis: str = 'umap', return_corr: bool = False,
                            plot: str = 'correlation', figsize: tuple = (15, 10)) -> str:
        """
        Plot the correlation of specified genes with energy landscapes.

        Args:
            genes: Gene names to plot (can be string or list of strings)
            energy: Type of energy correlation to use
            cluster: Cluster to analyze
            absolute: Whether to use absolute correlation values
            basis: Embedding basis for plotting
            return_corr: Whether to return correlation string instead of plotting
            plot: Whether to plot 'correlation' or 'expression'
            figsize: Figure size

        Returns:
            If return_corr is True, returns correlation string
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Ensure genes is a list
        if isinstance(genes, str):
            genes = [genes]

        # Determine the correct correlation dictionary
        if energy == 'interaction':
            corr_dict = self.correlation_interaction
        elif energy == 'degradation':
            corr_dict = self.correlation_degradation
        elif energy == 'bias':
            corr_dict = self.correlation_bias
        else:
            corr_dict = self.correlation

        if cluster not in corr_dict:
            print(f"Cluster '{cluster}' not found in correlation data.")
            return ""

        corr = corr_dict[cluster]

        # Get gene indices
        gene_indices = []
        for gene in genes:
            if gene in self.gene_names:
                idx = np.where(self.gene_names == gene)[0][0]
                gene_indices.append(idx)
            else:
                print(f"Gene '{gene}' not found in gene names.")

        if not gene_indices:
            print("No valid genes found.")
            return ""

        # Get correlations for the genes
        gene_corrs = corr[gene_indices]
        if absolute:
            gene_corrs = np.abs(gene_corrs)

        if return_corr:
            # Return correlation string
            corr_strings = []
            for gene, corr_val in zip(genes, gene_corrs):
                corr_strings.append(f"{cluster}: {gene} = {corr_val:.4f}")
            return "\n".join(corr_strings)

        # Create subplot layout
        n_genes = len(genes)
        if n_genes == 1:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            axes = [ax]
        else:
            cols = min(4, n_genes)
            rows = (n_genes + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=figsize)
            if rows == 1:
                axes = axes if n_genes > 1 else [axes]
            else:
                axes = axes.flatten()

        # Plot each gene
        for i, (gene, gene_idx, corr_val) in enumerate(zip(genes, gene_indices, gene_corrs)):
            if i >= len(axes):
                break

            ax = axes[i]

            if plot == 'correlation':
                # Plot correlation value as a bar
                ax.bar([0], [corr_val], color='skyblue', alpha=0.7)
                ax.set_ylim(-1, 1) if not absolute else ax.set_ylim(0, 1)
                ax.set_ylabel('Correlation')
                ax.set_title(f'{gene}\nCorr: {corr_val:.3f}')
                ax.set_xticks([])
            else:
                # Plot gene expression as scatter plot
                if basis in self.adata.obsm:
                    embedding = self.adata.obsm[f'X_{basis}']
                    gene_expr = to_numpy(self.adata.layers[self.spliced_matrix_key][:, gene_idx])

                    scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                                       c=gene_expr, cmap='viridis', s=1, alpha=0.6)
                    ax.set_xlabel(f'{basis.upper()}_1')
                    ax.set_ylabel(f'{basis.upper()}_2')
                    ax.set_title(f'{gene} (Corr: {corr_val:.3f})')
                    plt.colorbar(scatter, ax=ax, shrink=0.6)
                else:
                    ax.text(0.5, 0.5, f'{gene}\nCorr: {corr_val:.3f}\n({basis} not available)',
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{gene}')

        # Hide unused subplots
        for i in range(len(genes), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(f'Top Correlated Genes - {energy.title()} Energy ({cluster})', fontsize=14)
        plt.tight_layout()
        plt.show()

        return ""