import numpy as np
from scipy.special import hyp2f1
from typing import Union, Optional
from scipy.signal import convolve2d
import logging
logger = logging.getLogger(__name__)

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

def int_sig_act_inv(x: np.ndarray, threshold: np.ndarray, exponent: np.ndarray, verbose: bool = False) -> np.ndarray:
    """Compute the integral of the inverse sigmoid activation function.

    Args:
        x: Input sigmoid activation values (n_cells, n_genes).
        threshold: Sigmoid threshold parameters (n_genes,).
        exponent: Sigmoid exponent parameters (n_genes,).
        verbose: If True, logs intermediate computation results.

    Returns:
        np.ndarray: Integral of the inverse sigmoid (n_cells, n_genes).

    Notes:
        Uses the hypergeometric function (hyp2f1) from scipy.special.
        Assumes x is the sigmoid output, typically from compute_sigmoid.
    """
    if verbose:
        logger.info("Computing int_sig_act_inv with verbose output")
    
    # Ensure inputs are arrays and broadcastable
    x = np.asarray(x)
    threshold = np.asarray(threshold)
    exponent = np.asarray(exponent)
    
    # Compute hypergeometric term
    hyper_term = hyp2f1(-1 / exponent, (exponent - 1) / exponent, (2 * exponent - 1) / exponent, 1)
    z = -(exponent / (exponent - 1)) * threshold * hyper_term
    z = z[None, :]
    
    # Compute second term
    z1 = (-exponent * threshold * (1 - x) ** ((exponent - 1) / exponent) *
          hyp2f1(-1 / exponent, (exponent - 1) / exponent, (2 * exponent - 1) / exponent, 1 - x) /
          (exponent - 1))
    
    if verbose:
        logger.debug(f"z: {z[0]}")
        logger.debug(f"z1: {z1}")
    
    return z1 - z

def soften(z: np.ndarray, n_filt: int = 5) -> np.ndarray:
    """
    Apply a softening filter to a 2D array, using a uniform filter with a zeroed center.

    Args:
        z: Input 2D array to be softened.
        n_filt: Size of the square filter (must be odd). Defaults to 5.

    Returns:
        np.ndarray: Softened 2D array.

    Raises:
        ValueError: If n_filt is even or z is not 2D.
    """
    logger.info(f"Applying softening filter with n_filt={n_filt}")

    if n_filt % 2 == 0:
        raise ValueError("n_filt must be an odd number")
    if z.ndim != 2:
        raise ValueError("Input array z must be 2D")

    # Create uniform filter
    filt = np.ones((n_filt, n_filt)) / (n_filt**2 - 1)
    filt[n_filt // 2, n_filt // 2] = 0

    # Apply convolution
    softened_z = convolve2d(z, filt, mode='same')
    return softened_z