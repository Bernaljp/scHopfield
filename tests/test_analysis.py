import pytest
import numpy as np
from schopfield.tools.analysis import compute_energies
from schopfield._core.landscape import Landscape

def test_compute_energies(sample_adata):
    """Test compute_energies function."""
    landscape = Landscape(sample_adata)
    landscape.W = {"all": np.eye(10)}
    landscape.I = {"all": np.zeros(10)}
    landscape.gamma = {"all": np.ones(10)}
    landscape.threshold = np.ones(10)
    landscape.exponent = np.ones(10) * 2.0
    
    energies = compute_energies(landscape)
    assert set(energies.keys()) == {"total", "interaction", "degradation", "bias"}
    assert "all" in energies["total"]
    assert energies["total"]["all"].shape == (100,)
    
    x = np.random.rand(5, 10)
    energies_x = compute_energies(landscape, x)
    assert energies_x["total"]["all"].shape == (5,)

def test_compute_energies_invalid(sample_adata):
    """Test compute_energies with invalid inputs."""
    landscape = Landscape(sample_adata)
    with pytest.raises(ValueError):
        compute_energies(landscape)  # Missing W, I, gamma
    landscape.W = {"all": np.eye(10)}
    with pytest.raises(ValueError):
        compute_energies(landscape, np.ones((5, 10)))  # Missing threshold