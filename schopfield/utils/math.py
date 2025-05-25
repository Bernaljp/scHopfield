import numpy as np
from typing import Union

def compute_sigmoid(x: np.ndarray, threshold: Union[np.ndarray, float], exponent: Union[np.ndarray, float]) -> np.ndarray:
    """Compute the sigmoid function for given input x, threshold, and exponent.

    The sigmoid is computed as x^n / (x^n + s^n), where s is the threshold and n is the exponent.

    Args:
        x: Input array (n_cells, n_genes).
        threshold: Threshold parameter(s) controlling the transition point.
        exponent: Exponent parameter(s) controlling the steepness.

    Returns:
        np.ndarray: Sigmoid function applied to x (n_cells, n_genes).

    Notes:
        Inputs are converted to NumPy arrays for element-wise operations.
        Threshold and exponent can be scalars or arrays matching x's gene dimension.
    """
    x = np.asarray(x)
    threshold = np.asarray(threshold)
    exponent = np.asarray(exponent)
    return x**exponent / (x**exponent + threshold**exponent)