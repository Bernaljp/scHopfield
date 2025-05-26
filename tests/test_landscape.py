import pytest
import numpy as np
import anndata as ad
from schopfield._core.landscape import Landscape

def test_landscape_init(sample_adata):
    """Test Landscape initialization."""
    landscape = Landscape(
        sample_adata,
        spliced_matrix_key="Ms",
        velocity_key="velocity_S",
        gamma_key="gamma",
        cluster_key="cluster",
        genes=[0, 1, 2]
    )
    assert isinstance(landscape.adata, ad.AnnData)
    assert landscape.genes.shape == (3,)
    assert landscape.cluster_key == "cluster"
    assert landscape.scaffold is None

def test_landscape_init_invalid(sample_adata):
    """Test Landscape initialization with invalid inputs."""
    with pytest.raises(TypeError):
        Landscape("not_anndata")
    with pytest.raises(ValueError):
        Landscape(sample_adata, spliced_matrix_key="invalid_key")
    with pytest.raises(ValueError):
        Landscape(sample_adata, cluster_key="invalid_cluster")
    with pytest.raises(ValueError):
        Landscape(sample_adata, bias_regularization=-1.0)

def test_gene_parser(sample_adata):
    """Test gene_parser method."""
    landscape = Landscape(sample_adata)
    
    # Test gene names
    genes = ["Gene_0", "Gene_1"]
    indices = landscape.gene_parser(genes)
    assert np.array_equal(indices, [0, 1])
    
    # Test indices
    indices = landscape.gene_parser([2, 3])
    assert np.array_equal(indices, [2, 3])
    
    # Test boolean mask
    mask = [True, False] + [False] * 8
    indices = landscape.gene_parser(mask)
    assert np.array_equal(indices, [0])
    
    # Test invalid inputs
    with pytest.raises(ValueError):
        landscape.gene_parser(["Invalid_Gene"])
    with pytest.raises(ValueError):
        landscape.gene_parser([10])
    with pytest.raises(ValueError):
        landscape.gene_parser([True] * 5)

def test_hopfield_model(sample_adata):
    """Test hopfield_model method."""
    landscape = Landscape(sample_adata)
    landscape.W = {"all": np.eye(10)}
    landscape.I = {"all": np.zeros(10)}
    landscape.gamma = {"all": np.ones(10)}
    landscape.threshold = np.ones(10)
    landscape.exponent = np.ones(10) * 2.0
    
    xdot = landscape.hopfield_model(cluster="all")
    assert xdot.shape == (100, 10)
    
    x = np.random.rand(5, 10)
    xdot = landscape.hopfield_model(x, cluster="all")
    assert xdot.shape == (5, 10)

def test_hopfield_model_invalid(sample_adata):
    """Test hopfield_model with invalid inputs."""
    landscape = Landscape(sample_adata)
    with pytest.raises(ValueError):
        landscape.hopfield_model(cluster="invalid")
    landscape.W = {"all": np.eye(10)}
    with pytest.raises(ValueError):
        landscape.hopfield_model()  # Missing sigmoid layer
    landscape.threshold = None
    with pytest.raises(ValueError):
        landscape.hopfield_model(np.ones((5, 10)))

def test_jacobian_for_cell(sample_adata):
    """Test jacobian_for_cell method."""
    landscape = Landscape(sample_adata)
    landscape.W = {"all": np.eye(10)}
    landscape.I = {"all": np.zeros(10)}
    landscape.gamma = {"all": np.ones(10)}
    landscape.threshold = np.ones(10)
    landscape.exponent = np.ones(10) * 2.0
    
    x = np.random.rand(5, 10)
    jacobians = landscape.jacobian_for_cell(x, cluster="all")
    assert len(jacobians) == 5
    assert jacobians[0].shape == (10, 10)