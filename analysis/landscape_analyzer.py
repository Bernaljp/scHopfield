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

        self.adata = data
        self.spliced_matrix_key = spliced_matrix_key
        self.velocity_key = velocity_key
        self.gamma_key = degradation_key
        self.genes = self.gene_parser(genes)
        self.gene_names = self.adata.var.index[self.genes]
        self.cluster_key = cluster_key
        self.clusters = self.adata.obs[self.cluster_key].unique() if self.cluster_key else []
        self.scaffold = w_scaffold
        self.refit_gamma = refit_gamma

        if not manual_fit:
            # Fit sigmoids and heavysides for all genes
            self.fit_all_sigmoids()
            self.write_sigmoids()
            # Fit interactions
            self.fit_interactions(w_threshold=w_threshold,
                                  w_scaffold=w_scaffold,
                                  scaffold_regularization=scaffold_regularization,
                                  only_TFs=only_TFs,
                                  infer_I=infer_I,
                                  refit_gamma=refit_gamma,
                                  pre_initialize_W=pre_initialize_W,
                                  n_epochs=n_epochs,
                                  device=device,
                                  skip_all=skip_all,
                                  criterion=criterion,
                                  batch_size=batch_size,
                                  use_scheduler=use_scheduler,
                                  scheduler_kws=scheduler_kws,
                                  get_plots=get_plots)

            # Compute energies and their correlations with gene expressions
            self.get_energies()
            self.energy_genes_correlation()

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

    def fit_all_sigmoids(self, min_th: float = 0.05) -> None:
        """
        Fit sigmoid functions to gene expression data for all genes.

        Args:
            min_th: Threshold for zero expression in percentage of maximum expression.
        """
        # Retrieve expression data for all genes
        x = to_numpy(self.get_matrix(self.spliced_matrix_key, genes=self.genes).T)

        # Apply the fitting function to each gene's expression data
        results = np.array([fit_sigmoid(g, min_th=min_th) for g in x])

        # Unpack fitting results into separate attributes
        self.threshold, self.exponent, self.offset, self.sigmoid_mse = results.T

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
        self.fit_all_sigmoids()

        # Write sigmoids to the adata object
        self.write_sigmoids()

        # Return basic results (this is a placeholder implementation)
        return {
            'fitted_sigmoids': True,
            'analysis_complete': True
        }