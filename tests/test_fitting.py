import pytest
import numpy as np
from unittest.mock import patch
from schopfield.tools.fitting import fit_sigmoids, fit_interactions
from schopfield._core.landscape import Landscape

def test_fit_sigmoids(sample_adata):
    """Test fit_sigmoids function."""
    landscape = Landscape(sample_adata)
    fit_sigmoids(landscape, min_th=0.05)
    assert "sigmoid_threshold" in sample_adata.var
    assert "sigmoid_exponent" in sample_adata.var
    assert "sigmoid_offset" in sample_adata.var
    assert "sigmoid_mse" in sample_adata.var
    assert landscape.threshold is not None
    assert "sigmoid" in sample_adata.layers

def test_fit_interactions(sample_adata):
    """Test fit_interactions function with mocked optimizer."""
    landscape = Landscape(sample_adata)
    landscape.threshold = np.ones(10)
    landscape.exponent = np.ones(10) * 2.0
    
    with patch("schopfield.tools.fitting.ScaffoldOptimizer") as mock_optimizer:
        mock_model = mock_optimizer.return_value
        mock_model.W.weight.detach.return_value.cpu.return_value.numpy.return_value = np.eye(10)
        mock_model.I.detach.return_value.cpu.return_value.numpy.return_value = np.zeros(10)
        mock_model.gamma.detach.return_value.cpu.return_value.numpy.return_value = np.ones(10)
        
        fit_interactions(landscape, w_scaffold=np.eye(10), skip_all=True)
    
    assert "A" in landscape.W
    assert "B" in landscape.W
    assert landscape.W["A"].shape == (10, 10)