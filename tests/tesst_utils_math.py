import pytest
import numpy as np
from schopfield.utils.math import compute_sigmoid, int_sig_act_inv

def test_compute_sigmoid():
    """Test compute_sigmoid function."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    threshold = np.array([1.0, 1.0])
    exponent = np.array([2.0, 2.0])
    
    sig = compute_sigmoid(x, threshold, exponent)
    expected = x**2 / (x**2 + 1**2)
    np.testing.assert_almost_equal(sig, expected, decimal=6)

def test_int_sig_act_inv():
    """Test int_sig_act_inv function."""
    x = np.array([[0.5, 0.7], [0.3, 0.9]])
    threshold = np.array([1.0, 1.0])
    exponent = np.array([2.0, 2.0])
    
    result = int_sig_act_inv(x, threshold, exponent)
    assert result.shape == (2, 2)
    assert np.all(np.isfinite(result))
    
    # Test verbose mode
    result_verbose = int_sig_act_inv(x, threshold, exponent, verbose=True)
    np.testing.assert_array_equal(result, result_verbose)