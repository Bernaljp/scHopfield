import numpy as np
import pandas as pd
import anndata as ad
import umap
from typing import Union, List, Optional
from scipy.spatial import KDTree
from scipy.sparse import issparse
import logging
from schopfield.utils.math import compute_sigmoid
from schopfield.utils.data import get_matrix

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Landscape:
    
    """Core class for managing single-cell data and Hopfield model computations in scHopfield.

    This class serves as the primary interface for data storage (via AnnData) and core model
    computations, similar to AnnData in anndata. Analysis, visualization, and other tasks are
    handled by separate modules in schopfield.tools, schopfield.plotting, etc.

    The Hopfield model computes dynamics as ẋ = σ(x)Wᵀ - γx + I, where σ is the sigmoid
    activation, W is the interaction matrix, γ is the degradation rate, and I is the bias.

    Attributes:
        adata (ad.AnnData): Single-cell data object storing expression, metadata, and results.
        spliced_matrix_key (str): Key in adata.layers for spliced expression matrix.
        velocity_key (str): Key in adata.layers for velocity data.
        gamma_key (str): Key in adata.var for degradation rates.
        genes (np.ndarray): Indices of selected genes for analysis.
        gene_names (pd.Index): Names of selected genes.
        cluster_key (str): Key in adata.obs for cluster labels.
        scaffold (Optional[np.ndarray]): Scaffold matrix for regularization.
        refit_gamma (bool): Whether to refit degradation rates.
        bias_regularization (float): Regularization strength for bias terms.
        threshold (np.ndarray): Sigmoid threshold parameters from fitting.
        exponent (np.ndarray): Sigmoid exponent parameters from fitting.
        W (dict): Interaction matrices for each cluster and 'all'.
        I (dict): Bias vectors for each cluster and 'all'.
        gamma (dict): Degradation rates for each cluster and 'all'.
        embedding (Optional[umap.UMAP]): UMAP embedding model for transformations.
    """

    def __init__(
        self,
        adata: ad.AnnData,
        spliced_matrix_key: str = "Ms",
        velocity_key: str = "velocity_S",
        gamma_key: str = "gamma",
        genes: Union[None, List[str], List[bool], List[int]] = None,
        cluster_key: Optional[str] = None,
        scaffold: Optional[np.ndarray] = None,
        refit_gamma: bool = False,
        bias_regularization: float = 0.0,
    ) -> None:
        """Initialize the Landscape class with core data and parameters.

        Args:
            adata: AnnData object containing single-cell data.
            spliced_matrix_key: Key in adata.layers for spliced expression (default: 'Ms').
            velocity_key: Key in adata.layers for velocity data (default: 'velocity_S').
            gamma_key: Key in adata.var for degradation rates (default: 'gamma').
            genes: Subset of genes to analyze (names, indices, or boolean mask).
            cluster_key: Key in adata.obs for cluster labels (optional).
            scaffold: Scaffold matrix for regularization in interaction fitting (optional).
            refit_gamma: Whether to refit degradation rates during interaction fitting.
            bias_regularization: Regularization strength for bias terms.

        Raises:
            TypeError: If adata is not an AnnData object.
            ValueError: If keys are not found in adata, scaffold has incorrect, or bias_regularization is negative.
        """
        logger.info("Initializing Landscape object")
        
        # Validate inputs
        if not isinstance(adata, ad.AnnData):
            raise TypeError("adata must be an AnnData object")
        if spliced_matrix_key not in adata.layers:
            raise ValueError(f"spliced_matrix_key '{spliced_matrix_key}' not found in adata.layers")
        if velocity_key not in adata.layers:
            raise ValueError(f"velocity_key '{velocity_key}' not found in adata.layers")
        if gamma_key not in adata.var:
            raise KeyError(f"gamma_key '{gamma_key}' not found in adata.var")
        if cluster_key is not None and cluster_key not in adata.obs:
            raise ValueError(f"cluster_key '{cluster_key}' not found in adata.obs")
        if bias_regularization < 0:
            raise ValueError("bias_regularization must be non-negative")

        # Core data storage
        self._adata = adata.copy()  # Ensure a copy to avoid modifying input
        self.spliced_matrix_key = spliced_matrix_key
        self.velocity_key = velocity_key
        self.gamma_key = gamma_key
        # Gene selection
        self.genes = self.gene_parser(genes) if genes is not None else np.arange(adata.n_vars)
        self.gene_names = adata.var.index[self.genes]
        # Validate scaffold shape
        if scaffold is not None:
            n_genes = len(self.genes)
            if scaffold.shape != (n_genes, n_genes):
                raise ValueError(f"scaffold shape {scaffold.shape} must be ({n_genes}, {n_genes})")
        self.scaffold = scaffold
        # Cluster configuration
        self.cluster_key = cluster_key
        # Model parameters
        self.refit_gamma = refit_gamma
        self.bias_regularization = bias_regularization
        # Sigmoid parameters (set by schopfield.tools.fitting.fit_sigmoids)
        self.threshold = None
        self.exponent = None
        # Interaction and bias parameters (set by schopfield.tools.fitting.fit_interactions)
        self.W = {}
        self.I = {}
        self.gamma = {}
        # Embedding model (set by
        # schopfield.preprocessing.embedding.compute_embedding)
        self.embedding = None

    @property
    def adata(self) -> ad.AnnData:
        """Get the AnnData object."""
        return self._adata

    @adata.setter
    def adata(self, value: ad.AnnData) -> None:
        """Set the AnnData object with validation."""
        if not isinstance(value, ad.AnnData):
            raise TypeError("adata must be an AnnData object")
        self._adata = value

    def gene_parser(self, genes: Union[None, List[str], List[bool], List[int]]) -> np.ndarray:
        """Parse gene identifiers into indices.

        Args:
            genes: Gene names, boolean mask, indices, or None to select all genes.

        Returns:
            np.ndarray: Indices of selected genes.

        Raises:
            ValueError: If genes is empty, contains invalid types, or names are not in adata.var.index.
        """
        logger.info("Parsing gene identifiers")
        
        if genes is None:
            return np.arange(self.adata.n_vars)
        
        if len(genes)==0:
            raise ValueError("Genes list cannot be empty")

        if isinstance(genes[0], str):
            if not all(isinstance(g, str) for g in genes):
                raise ValueError("All elements in genes must be strings if gene names are provided")
            gene_indices = pd.Index(self.adata.var.index).get_indexer(genes)
            if np.any(gene_indices == -1):
                missing_genes = np.array(genes)[gene_indices == -1]
                raise ValueError(f"Gene names not found in adata.var.index: {missing_genes}")
            if len(set(genes)) != len(genes):
                raise ValueError("Duplicate gene names provided")
            return gene_indices
        elif type(genes[0]) in (int, np.integer):
            gene_indices = np.array(genes)
            if np.any((gene_indices < 0) | (gene_indices >= self.adata.n_vars)):
                raise ValueError("Gene indices out of bounds")
            return gene_indices
        elif type(genes[0]) in (bool, np.bool_):
            if len(genes) != self.adata.n_vars:
                raise ValueError("Boolean gene list must match number of genes in adata")
            return np.where(genes)[0]
        else:
            raise ValueError("Genes must be None, list of gene names, indices, or boolean mask")

    def hopfield_model(self, x: Optional[np.ndarray] = None, cluster: str = "all") -> np.ndarray:
        """Compute the Hopfield model output for given expression data.

        The model computes dynamics as ẋ = σ(x)Wᵀ - γx + I, where σ is the sigmoid activation,
        W is the interaction matrix, γ is the degradation rate, and I is the bias vector.

        Args:
            x: Gene expression data (n_cells, n_genes). If None, uses spliced matrix from adata.
            cluster: Cluster label or 'all' for model parameters (default: 'all').

        Returns:
            np.ndarray: Hopfield model output (n_cells, n_genes).

        Raises:
            ValueError: If cluster is not found, model parameters are not initialized, or x has incorrect shape.

        Notes:
            Requires fitted parameters from schopfield.tools.fitting.fit_interactions.
            Sigmoid activations are computed via schopfield.utils.math.compute_sigmoid.
            If x is None, uses adata.layers[spliced_matrix_key] filtered by cluster if cluster_key is set.
        """
        logger.info(f"Computing Hopfield model for cluster '{cluster}'")
        
        # Validate cluster and model parameters
        if cluster not in self.W:
            raise ValueError(f"Cluster '{cluster}' not found in fitted interaction matrices")
        if not all(k in self.W for k in [cluster]) or not all(k in self.I for k in [cluster]):
            raise ValueError(f"Model parameters (W, I) for cluster '{cluster}' not initialized")

        # Get sigmoid activations
        idx = self.adata.obs[self.cluster_key] == cluster if x is None and self.cluster_key is not None else slice(None)
        if x is None:
            if 'sigmoid' not in self.adata.layers:
                raise ValueError("Sigmoid layer not found in adata.layers; run schopfield.tools.fitting.fit_sigmoids")
            sig = get_matrix(self.adata, 'sigmoid', genes=self.genes)[idx]
        else:
            if self.threshold is None or self.exponent is None:
                raise ValueError("Sigmoid parameters not initialized; run schopfield.tools.fitting.fit_sigmoids")
            sig = compute_sigmoid(x, self.threshold, self.exponent)

        # Get model parameters
        W = self.W[cluster]
        I = self.I[cluster]
        gamma = (self.adata.var[self.gamma_key][self.genes].values
                 if not self.refit_gamma else self.gamma[cluster])

        # Get input data
        if x is None:
            x = np.asarray(get_matrix(self.adata, self.spliced_matrix_key, genes=self.genes)[idx])
        else:
            if x.ndim == 1:
                x = x[None, :]
            if x.shape[1] != len(self.genes):
                raise ValueError(f"Input x must have {len(self.genes)} genes, got {x.shape[1]}")

        # Compute dynamics
        xdot = sig @ W.T - gamma[None, :] * x + I
        return xdot

    def jacobian_for_cell(self, x: np.ndarray, cluster: str = "all") -> List[np.ndarray]:
        """Compute the Jacobian matrix for each point in x based on the model parameters.

        The Jacobian is computed as ∂ẋ/∂x = W * dσ/dx - diag(γ), where dσ/dx is the derivative
        of the sigmoid function. For each point, the cluster is determined by the closest cell
        in the dataset (based on spliced expression) if cluster_key is provided.

        Args:
            x: Array of points (n_points, n_genes) or (n_genes,) at which to compute the Jacobian.
            cluster: Cluster label or 'all' for model parameters (default: 'all').

        Returns:
            List[np.ndarray]: List of Jacobian matrices (n_genes, n_genes) for each point.

        Raises:
            ValueError: If cluster is not found, model parameters are not initialized, or x has incorrect shape.

        Notes:
            Uses KDTree for efficient nearest neighbor search to assign clusters.
            Requires fitted parameters from schopfield.tools.fitting.fit_interactions and fit_sigmoids.
            Handles sparse matrices in adata.layers[spliced_matrix_key] efficiently.
        """
        logger.info(f"Computing Jacobians for {x.shape[0] if x.ndim > 1 else 1} points, cluster '{cluster}'")
        
        # Validate inputs
        if cluster not in self.W:
            raise ValueError(f"Cluster '{cluster}' not found in fitted interaction matrices")
        if self.threshold is None or self.exponent is None:
            raise ValueError("Sigmoid parameters not initialized; run schopfield.tools.fitting.fit_sigmoids")
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[1] != len(self.genes):
            raise ValueError(f"Input x must have {len(self.genes)} genes, got {x.shape[1]}")

        # Adjust sigmoid parameters for broadcasting
        ex = self.exponent[None, :] if x.shape[0] > 1 else self.exponent
        th = self.threshold[None, :] if x.shape[0] > 1 else self.threshold

        # Compute sigmoid values
        sig = compute_sigmoid(x, th, ex)

        # Get spliced matrix and handle sparsity
        matrix = get_matrix(self.adata, self.spliced_matrix_key, genes=self.genes)
        matrix = matrix.toarray() if issparse(matrix) else matrix

        # Find closest cells using KDTree
        tree = KDTree(matrix)
        _, minidx = tree.query(x, k=1)
        minidx = minidx.flatten()
        
        # Assign clusters
        celltype = (self.adata.obs[self.cluster_key].values[minidx]
                    if self.cluster_key else np.array([cluster] * len(x)))

        # Compute Jacobians
        jacobians = []
        gamma_base = self.adata.var[self.gamma_key][self.genes].values
        for i in range(len(x)):
            # Compute sigmoid derivative with numerical stability
            dsig_dx = np.nan_to_num(ex[:, 0] * sig[i] * (1 - sig[i]) / (x[i] + 1e-10))
            
            # Get cluster-specific parameters
            ct = celltype[i]
            if ct not in self.W:
                logger.warning(f"Cluster '{ct}' not found; using cluster '{cluster}' parameters")
                ct = cluster
            W = self.W[ct]
            gamma = gamma_base if not self.refit_gamma else self.gamma[ct]
            
            # Compute Jacobian
            jacobian_matrix = W * dsig_dx[:, None] - np.diag(gamma)
            jacobians.append(jacobian_matrix)

        return jacobians