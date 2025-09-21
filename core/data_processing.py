"""
Data processing functionality for scHopfield package.
Extracted from original scMomentum Landscape class.
"""

import numpy as np
import anndata
from typing import Union, List
from ..utils.utilities import to_numpy, sigmoid, fit_sigmoid


class DataProcessor:
    """
    Handles all data processing operations including gene parsing,
    matrix operations, and data validation.
    """

    def __init__(self, adata: anndata.AnnData,
                 spliced_matrix_key: str = 'Ms',
                 velocity_key: str = 'velocity_S',
                 degradation_key: str = 'gamma',
                 genes: Union[None, List[str], List[bool], List[int]] = None):
        """
        Initialize data processor.

        Parameters:
            adata: AnnData object containing the data
            spliced_matrix_key: Key for spliced matrix in adata.layers
            velocity_key: Key for velocity matrix in adata.layers
            degradation_key: Key for degradation rates in adata.var
            genes: Gene selection (names, indices, or boolean mask)
        """
        self.adata = adata
        self.spliced_matrix_key = spliced_matrix_key
        self.velocity_key = velocity_key
        self.gamma_key = degradation_key
        self.genes = self.gene_parser(genes)
        self.gene_names = self.adata.var.index[self.genes]

    def gene_parser(self, genes: Union[None, List[str], List[bool], List[int]]) -> np.ndarray:
        """
        Parses the given gene list into indices that indicate which genes to use.

        Args:
            genes: List of gene names, indices, or Boolean values, or None.

        Returns:
            Array of indices indicating which genes to use.
        """
        if genes is None:
            return np.arange(self.adata.n_vars)

        if isinstance(genes[0], str):
            gene_indices = self.adata.var.index.get_indexer_for(genes)
            if np.any(gene_indices == -1):
                missing_genes = np.array(genes)[gene_indices == -1]
                raise ValueError(f"Gene names not found in adata.var.index: {missing_genes}")
            return gene_indices
        elif isinstance(genes[0], (int, np.int64, np.int32, np.int16, np.int8)):
            return np.array(genes)
        elif isinstance(genes[0], (bool, np.bool_)):
            if len(genes) != self.adata.n_vars:
                raise ValueError("Boolean gene list must have the same length as the number of genes in the dataset.")
            return np.where(genes)[0]
        else:
            raise ValueError("Genes argument must be None, a list of gene names, indices, or a Boolean mask.")

    def get_matrix(self, key: str, genes=None) -> np.ndarray:
        """
        Retrieve a specific matrix from the AnnData object based on the given key.

        Parameters:
            key: Key for the desired matrix in the AnnData layers.
            genes: List of gene names or indices to subset the matrix. Defaults to None.

        Returns:
            The requested matrix, optionally subset by genes.
        """
        if genes is None:
            return self.adata.layers[key]
        else:
            return self.adata.layers[key][:, genes]

    def write_property(self, key: str, value: np.ndarray):
        """
        Write a property (value) to the AnnData object under the specified key,
        determining the appropriate location based on the shape of the value.

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

    def fit_all_sigmoids(self, min_th: float = 0.05):
        """
        Fit sigmoid functions to gene expression data for all genes.

        Args:
            min_th: Threshold for zero expression in percentage of maximum expression of each gene.
        """
        # Retrieve expression data for all genes
        x = to_numpy(self.get_matrix(self.spliced_matrix_key, genes=self.genes).T)

        # Apply the fitting function to each gene's expression data
        results = np.array([fit_sigmoid(g, min_th=min_th) for g in x])

        # Unpack fitting results into separate attributes
        self.threshold, self.exponent, self.offset, self.sigmoid_mse = results.T

    def write_sigmoids(self):
        """Write sigmoid values to the AnnData object."""
        sig = self.get_sigmoid()
        sigmoids = np.zeros(list(self.adata.layers.values())[0].shape, dtype=sig.dtype)
        sigmoids[:, self.genes] = sig
        self.write_property('sigmoid', sigmoids)
        self.sigmoids = self.adata.layers['sigmoid']

    def get_sigmoid(self, x=None) -> np.ndarray:
        """
        Compute the sigmoid activation for the given input x or for the entire spliced matrix.

        Args:
            x: Input data for which to compute the sigmoid function.
               If None, the method uses the spliced matrix from the AnnData object.

        Returns:
            The sigmoid activation applied to the input data.
        """
        # Use the entire spliced matrix if x is not provided
        if x is None:
            x = to_numpy(self.get_matrix(self.spliced_matrix_key, genes=self.genes))

        # Compute the sigmoid function of x using the class's threshold and exponent parameters
        sigmoid_output = np.nan_to_num(sigmoid(x, self.threshold[None, :], self.exponent[None, :]))

        return sigmoid_output