"""Mathematical utility functions for scHopfield."""

import numpy as np
from scipy.interpolate import griddata
from scipy.special import hyp2f1 as hyper
from scipy.signal import convolve2d
from scipy.optimize import least_squares


def sigmoid(x, s, n):
    """
    Compute the sigmoid (Hill) function for given input x, threshold s, and exponent n.

    This is the Hill function phi(x) = x^n / (x^n + s^n) referred to as the Hill
    activation in the Methods; "sigmoid" and "Hill" are used interchangeably in the
    code. ``s`` is the half-maximal threshold (k) and ``n`` is the Hill coefficient.

    Args:
        x (np.ndarray): Input array for which to compute the sigmoid function.
        s (float or np.ndarray): Threshold parameter of the sigmoid. The point at which the sigmoid
                                 transitions from its minimum value to its maximum value.
        n (float): Exponent parameter controlling the steepness of the sigmoid curve.

    Returns:
        np.ndarray: The sigmoid function applied to each element of x.
    """
    # Ensure inputs are numpy arrays for element-wise operations
    x = np.asarray(x)
    s = np.asarray(s)

    # Compute the sigmoid function
    return x**n / (x**n + s**n)


def d_sigmoid(x, s, n):
    """
    Compute the derivative of the sigmoid function with respect to x.

    Args:
        x (np.ndarray): Input array for which to compute the derivative.
        s (float): Threshold parameter of the sigmoid.
        n (float): Exponent parameter controlling the steepness of the sigmoid curve.

    Returns:
        np.ndarray: The derivative of the sigmoid function applied to each element of x.

    Notes:
        For the Hill function phi(x) = x^n / (x^n + s^n), the exact derivative is
        phi'(x) = n * phi(x) * (1 - phi(x)) / x. The factor ``n`` must be included
        (it is omitted in Methods Eq. 4/21 as written, a typographical error; the
        Jacobian code in tools/jacobian.py includes it correctly). x = 0 is guarded.
    """
    x = np.asarray(x, dtype=float)
    n = np.asarray(n, dtype=float)
    sig = sigmoid(x, s, n)
    x_safe = np.where(x == 0, 1.0, x)
    return n * sig * (1 - sig) / x_safe


def fit_k(g):
    """
    Fit a threshold parameter k for a heavyside function based on data g by aligning
    it with the most rapid increase in the empirical cumulative distribution function (ECDF) of g.

    Args:
        g (np.ndarray): Input data array from which to compute the threshold k.

    Returns:
        float: Optimized threshold parameter k.
    """
    # Ensure g is a numpy array and remove any NaN values
    g = np.asarray(g)
    g = g[~np.isnan(g)]

    # Sort g to compute the ECDF
    sorted_g = np.sort(g)
    if sorted_g.size == 0:
        return 0.0
    if sorted_g.size < 2 or np.ptp(sorted_g) == 0:
        # Degenerate (single/constant value): the "threshold" is that value.
        return float(sorted_g[0])

    # Compute ECDF - each point in sorted_g corresponds to a step in the ECDF
    ecdf = np.arange(1, len(sorted_g) + 1) / len(sorted_g)

    # Compute the gradient of the ECDF to find the most rapid increase. Ties in
    # sorted_g give zero spacing (divide-by-zero); those become non-finite and are
    # ignored so the argmax lands on a real, finite maximum.
    with np.errstate(divide="ignore", invalid="ignore"):
        gradient_ecdf = np.gradient(ecdf, sorted_g)
    gradient_ecdf = np.where(np.isfinite(gradient_ecdf), gradient_ecdf, -np.inf)

    # Find the index of the maximum gradient, which corresponds to the optimal k
    max_gradient_index = np.argmax(gradient_ecdf)

    # The optimal k is the value in sorted_g at the index of the maximum gradient
    k = sorted_g[max_gradient_index]

    return k


def fit_sigmoid(g, min_th=0.05, n_min=1.0, n_max=8.0, refine=True):
    """
    Fit a Hill/sigmoid CDF ``phi(x) = x^n / (x^n + k^n)`` to a gene's expression.

    A fast closed-form estimate (linear regression of the log-logit ECDF) initializes
    ``(k, n)``; it is then constrained to biologically plausible bounds
    (``n in [n_min, n_max]``, ``0 < k <= max(x)``) and, if ``refine``, polished by a
    bounded nonlinear least-squares fit to the ECDF (kept only when it lowers the MSE).
    Degenerate genes (all-zero, constant, or too few expressed cells) are handled
    explicitly instead of returning NaN or out-of-range parameters, which the raw
    closed-form fit could previously do (negative ``n``, ``k`` exploding via
    ``exp(-b/n)`` when ``n -> 0``).

    Parameters
    ----------
    g : np.ndarray
        Expression values of one gene across cells.
    min_th : float, optional (default: 0.05)
        Cells below ``min_th * max(g)`` are treated as "off" and excluded from the CDF
        fit; their fraction is returned as ``offset``.
    n_min, n_max : float, optional (default: 1.0, 8.0)
        Bounds on the Hill exponent (matches ``HillScaffoldOptimizer``). ``n >= 1`` keeps
        the activation monotone-sigmoidal; the upper cap avoids near-step fits to noise.
    refine : bool, optional (default: True)
        Refine the clamped closed-form estimate with a bounded nonlinear least-squares
        fit; the refined ``(k, n)`` is kept only when it does not worsen the MSE.

    Returns
    -------
    (k, n, offset, mse) : tuple of float
        Half-max threshold ``k``, Hill exponent ``n``, off-fraction, and CDF-fit MSE.
    """
    g = np.asarray(g, dtype=float)
    g = g[np.isfinite(g)]
    n_default = float(np.clip(2.0, n_min, n_max))

    # Degenerate: no data or non-positive dynamic range -> gene is effectively "off".
    if g.size == 0 or not (np.max(g) > 0):
        return 1.0, n_default, 1.0, 0.0

    gmax = float(np.max(g))
    thr = min_th * gmax
    offset = float(np.mean(g < thr))
    valid = np.sort(g[g > thr])

    # Too few expressed cells for a shape fit -> robust threshold, default exponent.
    if valid.size < 5:
        k = float(np.clip(np.median(valid) if valid.size else gmax, 1e-6, gmax))
        x_cdf = valid if valid.size else np.array([gmax])
        y_cdf = np.linspace(0.0, 1.0, x_cdf.size)
        mse = float(np.mean((sigmoid(x_cdf, k, n_default) - y_cdf) ** 2))
        return k, n_default, offset, mse

    x_cdf = valid
    y_cdf = np.linspace(0.0, 1.0, valid.size)

    def _mse(k, n):
        return float(np.mean((sigmoid(x_cdf, k, n) - y_cdf) ** 2))

    # Fast closed-form estimate: regress logit(y) on log(x). Endpoints y in {0,1} give
    # +/-inf and are dropped by the finite mask (expected; warnings silenced).
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.log(x_cdf)
        ty = np.log(y_cdf / (1.0 - y_cdf))
    m = np.isfinite(tx) & np.isfinite(ty)
    n_hat, b_hat = np.nan, np.nan
    if m.sum() >= 2:
        A = np.vstack([tx[m], np.ones(int(m.sum()))]).T
        n_hat, b_hat = np.linalg.lstsq(A, ty[m], rcond=None)[0]

    # Derive (k, n) from the closed-form fit, clamped to valid ranges.
    if np.isfinite(n_hat) and n_hat > 0:
        n0 = float(np.clip(n_hat, n_min, n_max))
        k0 = float(np.exp(-b_hat / n_hat)) if np.isfinite(b_hat) else fit_k(valid)
    else:
        n0 = n_default
        k0 = fit_k(valid)
    k0 = float(np.clip(k0, 1e-6, gmax))

    k_best, n_best, mse_best = k0, n0, _mse(k0, n0)

    # Bounded nonlinear refinement on the ECDF, from multiple starts. Genes whose CDF is
    # a "double sigmoid" (two expression regimes with different k/n) have a poor single
    # optimum near the closed-form init; also starting from ~2x the closed-form (k,n)
    # lets the optimizer settle on the steeper/higher regime. Keep whichever start gives
    # the lowest MSE.
    if refine:
        starts = [(k0, n0),
                  (float(np.clip(2.0 * k0, 1e-6, gmax)), float(np.clip(2.0 * n0, n_min, n_max)))]
        for x0k, x0n in starts:
            try:
                res = least_squares(
                    lambda p: sigmoid(x_cdf, p[0], p[1]) - y_cdf,
                    x0=[x0k, x0n],
                    bounds=([1e-6, n_min], [gmax, n_max]),
                    max_nfev=300,
                )
                k_r, n_r = float(res.x[0]), float(res.x[1])
                mse_r = _mse(k_r, n_r)
                if np.isfinite(mse_r) and mse_r <= mse_best:
                    k_best, n_best, mse_best = k_r, n_r, mse_r
            except Exception:
                pass

    return k_best, n_best, offset, mse_best


def fit_sigmoid_bimodal(g, min_th=0.05, n_min=1.0, n_max=20.0, margin=0.85):
    """Fit a two-component Hill CDF ``a*H(x;k1,n1) + (1-a)*H(x;k2,n2)`` to a gene.

    Genes with two expression regimes (a "double sigmoid" CDF) are poorly captured by a
    single Hill; this fits a bimodal mixture. It is accepted over the single fit only when
    it improves the MSE by at least ``(1 - margin)`` (default 15%), so the extra
    parameters do not just overfit noise.

    Returns
    -------
    (k1, n1, k2, n2, a, offset, mse, is_bimodal)
        Component thresholds/exponents, mixing weight ``a`` on component 1, off-fraction,
        MSE, and whether the bimodal fit was accepted. When not accepted, both components
        equal the single fit and ``a = 1``.
    """
    k1, n1, offset, mse1 = fit_sigmoid(g, min_th=min_th, n_min=n_min, n_max=n_max)
    g = np.asarray(g, dtype=float); g = g[np.isfinite(g)]
    if g.size == 0 or not (np.max(g) > 0):
        return k1, n1, k1, n1, 1.0, offset, mse1, False
    gmax = float(np.max(g)); thr = min_th * gmax
    valid = np.sort(g[g > thr])
    if valid.size < 8:
        return k1, n1, k1, n1, 1.0, offset, mse1, False
    x = valid; y = np.linspace(0.0, 1.0, valid.size)

    def _mix(k1_, n1_, k2_, n2_, a_):
        return a_ * (x ** n1_ / (x ** n1_ + k1_ ** n1_)) + (1 - a_) * (x ** n2_ / (x ** n2_ + k2_ ** n2_))

    try:
        res = least_squares(
            lambda p: _mix(*p) - y,
            x0=[k1, n1, min(2 * k1, gmax), n1, 0.5],
            bounds=([1e-6, n_min, 1e-6, n_min, 0.0], [gmax, n_max, gmax, n_max, 1.0]),
            max_nfev=600,
        )
        k1b, n1b, k2b, n2b, ab = [float(v) for v in res.x]
        mse2 = float(np.mean((_mix(k1b, n1b, k2b, n2b, ab) - y) ** 2))
        if np.isfinite(mse2) and mse2 < margin * mse1:
            return k1b, n1b, k2b, n2b, ab, offset, mse2, True
    except Exception:
        pass
    return k1, n1, k1, n1, 1.0, offset, mse1, False


def int_sig_act_inv(x, s, n, verbose=False):
    """
    Compute the integral of the inverse sigmoid activation function.

    Args:
        x (np.ndarray): Input data array.
        s (float): Sigmoid threshold parameter.
        n (float): Sigmoid steepness parameter.
        verbose (bool): If True, prints intermediate computation results.

    Returns:
        np.ndarray: Integral of the inverse sigmoid activation.
    """
    z = -(n / (n - 1)) * s * hyper(-1 / n, (n - 1) / n, (2 * n - 1) / n, 1)
    z = z[None, :]
    n = n[None, :]
    s = s[None, :]
    z1 = -n * s * (1 - x) ** ((n - 1) / n) * hyper(-1 / n, (n - 1) / n, (2 * n - 1) / n, 1 - x) / (n - 1)

    if verbose:
        print(z[0])
        print(z1)

    return z1 - z


def soften(z, n_filt=5):
    """
    Apply a softening filter to an array z, with a specified filter size n_filt.
    The filter applied is a uniform filter, except for the center, which is set to 0.

    Args:
        z (np.ndarray): Input 2D array to be softened.
        n_filt (int): Size of the square filter, must be an odd number to have a center.

    Returns:
        np.ndarray: The softened 2D array.
    """
    # Create a uniform filter of size n_filt x n_filt
    filt = np.ones((n_filt, n_filt)) / (n_filt**2 - 1)

    # Set the center of the filter to 0
    filt[n_filt // 2, n_filt // 2] = 0

    # Apply the filter to the input array z using 2D convolution
    softened_z = convolve2d(z, filt, mode='same')

    return softened_z


def rezet(gridX, gridY, energySurface, points):
    """
    Interpolate energy values from a grid onto specific points.

    Args:
        gridX (np.ndarray): The grid's X coordinates.
        gridY (np.ndarray): The grid's Y coordinates.
        energySurface (np.ndarray): The energy values at each point on the grid.
        points (np.ndarray): The points at which to interpolate energy values, shape (N, 2).

    Returns:
        np.ndarray: The interpolated energy values at the given points.
    """
    points = np.array(points)
    grid_points = np.array([gridX.ravel(), gridY.ravel()]).T
    energy_values = griddata(grid_points, energySurface.ravel(), points, method='cubic')
    return energy_values


def ordinal(n: int):
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + suffix
