import pytest
import numpy as np
import anndata as ad
from scipy.sparse import csr_matrix
from schopfield.utils.data import to_numpy, get_matrix, write_property, write_sigmoids, write_energies
from schopfield._core.landscape import Landscape

def test_to_numpy(sample_adata):
    """Test to_numpy function."""
    matrix = sample_adata.layers["Ms"]
    result = to_numpy(matrix)
    assert isinstance(result, np.ndarray)
    
    sparse_matrix = csr_matrix(matrix)
    result = to_numpy(sparse_matrix)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, matrix)

def test_get_matrix(sample_adata):
    """Test get_matrix function."""
    landscape = Landscape(sample_adata)
    matrix = get_matrix(sample_adata, "Ms", genes=[0, 1])
    assert matrix.shape == (100, 2)
    
    with pytest.raises(KeyError):
        get_matrix(sample_adata, "invalid_key")

def test_write_property(sample_adata):
    """Test write_property function."""
    landscape = Landscape(sample_adata)
    values = np.ones(10)
    write_property(sample_adata, "test_prop", values)
    np.testing.assert_array_equal(sample_adata.var["test_prop"], values)
    
    with pytest.raises(ValueError):
        write_property(sample_adata, "test_prop", np.ones(5))

def test_write_sigmoids(sample_adata):
    """Test write_sigmoids function."""
    landscape = Landscape(sample_adata)
    landscape.threshold = np.ones(10)
    landscape.exponent = np.ones(10) * 2.0
    write_sigmoids(landscape)
    assert "sigmoid" in sample_adata.layers
    assert sample_adata.layers["sigmoid"].shape == (100, 10)

def test_write_energies(sample_adata):
    """Test write_energies function."""
    landscape = Landscape(sample_adata)
    energies = {
        "total": {"all": np.random.rand(100)},
        "interaction": {"all": np.random.rand(100)},
        "degradation": {"all": np.random.rand(100)},
        "bias": {"all": np.random.rand(100)}
    }
    write_energies(landscape, energies)
    assert "Total_energy" in sample_adata.obs
    assert sample_adata.obs["Total_energy"].shape == (100,)