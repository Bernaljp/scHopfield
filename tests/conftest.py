import pytest
import numpy as np
import anndata as ad
import pandas as pd
from schopfield._core.landscape import Landscape

@pytest.fixture
def sample_adata():
    """Create a sample AnnData object for testing."""
    n_cells = 100
    n_genes = 10
    np.random.seed(42)
    
    # Expression and velocity matrices
    X = np.random.rand(n_cells, n_genes)
    Ms = X + 0.1 * np.random.randn(n_cells, n_genes)  # Spliced
    velocity = 0.1 * np.random.randn(n_cells, n_genes)
    
    # Gene metadata
    var = pd.DataFrame(
        {
            "gamma": np.ones(n_genes) * 0.1,
            "sigmoid_threshold": np.ones(n_genes),
            "sigmoid_exponent": np.ones(n_genes) * 2.0,
            "sigmoid_offset": np.zeros(n_genes),
            "sigmoid_mse": np.zeros(n_genes),
        },
        index=[f"Gene_{i}" for i in range(n_genes)]
    )
    
    # Cell metadata
    obs = pd.DataFrame(
        {"cluster": np.random.choice(["A", "B"], n_cells)},
        index=[f"Cell_{i}" for i in range(n_cells)]
    )
    
    # Create AnnData
    adata = ad.AnnData(X, obs=obs, var=var)
    adata.layers["Ms"] = Ms
    adata.layers["velocity_S"] = velocity
    adata.layers["sigmoid"] = 1 / (1 + np.exp(-Ms))  # Mock sigmoid layer
    
    return adata

@pytest.fixture
def sample_landscape(sample_adata):
    """Create a sample Landscape object for testing."""
    landscape = Landscape(sample_adata, cluster_key='cluster')
    landscape.W = {"all": np.eye(10)}
    landscape.I = {"all": np.zeros(10)}
    landscape.gamma = {"all": np.ones(10)}

    for cluster in sample_adata.obs['cluster'].unique():
        landscape.W[cluster] = np.eye(10) * 0.1
        landscape.I[cluster] = np.zeros(10)
        landscape.gamma[cluster] = np.ones(10) * 0.1
        

    landscape.threshold = np.ones(10)
    landscape.exponent = np.ones(10) * 2.0
    
    return landscape
