"""Maximum-likelihood fit of one- and two-component Hill activations to single-cell expression.

The Hill function ``x^n / (x^n + k^n)`` is the cumulative distribution function of the
log-logistic distribution with scale ``k`` and shape ``n``, so a gene's expression can be modeled
directly as a sample from a log-logistic distribution, or from a mixture of two, and fitted by
maximum likelihood. On the log scale a log-logistic variable is logistic with location ``log k``
and scale ``1/n``, whose standard deviation is ``pi / (sqrt(3) n)``; a two-component Gaussian
mixture fitted to log expression therefore gives a natural starting point for both components.

All genes are fitted at once as one batched problem, with each gene's parameters independent of
every other gene's, so the result equals fitting them one at a time.
"""
from typing import List, Optional, Sequence, Tuple

import numpy as np

_TINY = 1e-12


def active_values(g: np.ndarray, min_th: float = 0.05) -> Tuple[np.ndarray, float]:
    """Sorted expression values above ``min_th * max(g)``, and the fraction of cells below it."""
    g = np.asarray(g, dtype=float)
    g = g[np.isfinite(g)]
    if g.size == 0 or not (np.max(g) > 0):
        return np.array([], dtype=float), 1.0
    thr = min_th * float(np.max(g))
    return np.sort(g[g > thr]), float(np.mean(g < thr))


def loglogistic_logpdf_np(x, k, n):
    """Log-density of the log-logistic distribution, the derivative of the Hill function."""
    logx = np.log(np.maximum(np.asarray(x, dtype=float), _TINY))
    logk = np.log(k)
    return np.log(n) + n * logk + (n - 1.0) * logx - 2.0 * np.logaddexp(n * logk, n * logx)


def log_moment_init(v: np.ndarray, n_min: float, n_max: float) -> Tuple[float, float]:
    """Single-component start: ``k`` from the mean and ``n`` from the spread of log expression."""
    lv = np.log(v)
    sd = float(np.std(lv))
    n = np.pi / (np.sqrt(3.0) * sd) if sd > 0 else n_max
    return float(np.exp(np.mean(lv))), float(np.clip(n, n_min, n_max))


def gmm_log_init(v: np.ndarray, n_min: float, n_max: float,
                 seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-component start from a Gaussian mixture on log expression, components ordered by ``k``."""
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(n_components=2, covariance_type="spherical", random_state=seed)
    gm.fit(np.log(v).reshape(-1, 1))
    mu = gm.means_.ravel()
    sd = np.sqrt(gm.covariances_.ravel())
    order = np.argsort(mu)
    k = np.exp(mu[order])
    n = np.clip(np.pi / (np.sqrt(3.0) * np.maximum(sd[order], _TINY)), n_min, n_max)
    return k, n, gm.weights_[order]


def fit_loglogistic_mle(actives: Sequence[np.ndarray], k0: np.ndarray, n0: np.ndarray,
                        w0: np.ndarray, n_min: float = 1.0, n_max: float = 20.0,
                        steps: int = 1500, lr: float = 0.05, device: Optional[str] = None):
    """Batched maximum-likelihood fit of a ``C``-component log-logistic mixture per gene.

    Parameters
    ----------
    actives : sequence of 1-D arrays
        One array of positive expression values per gene.
    k0, n0, w0 : ndarray, shape (G, C)
        Starting scales, shapes and mixture weights (each row of ``w0`` sums to one).
    n_min, n_max : float
        Bounds on every shape parameter. The lower bound keeps the Hill derivative finite at the
        origin; the upper bound keeps the likelihood bounded, since a log-logistic whose shape
        grows without limit at one observed value has unbounded density there.
    steps, lr : int, float
        Adam iterations and learning rate on the unconstrained parameters.

    Returns
    -------
    k, n, w : ndarray, shape (G, C)
        Fitted parameters, components ordered by increasing ``k``. The scale is bounded to
        ``(0, max(values)]`` for each gene.
    nll : ndarray, shape (G,)
        Negative log-likelihood of each gene's values at the fit.
    """
    import torch
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    G = len(actives)
    C = k0.shape[1]
    L = max(1, max(len(a) for a in actives))
    X = np.ones((G, L))
    M = np.zeros((G, L))
    xmax = np.ones(G)
    for i, a in enumerate(actives):
        if len(a):
            X[i, :len(a)] = a
            M[i, :len(a)] = 1.0
            xmax[i] = float(np.max(a))

    def t(z):
        return torch.as_tensor(np.asarray(z, dtype=float), dtype=torch.float64, device=dev)

    def logit(u):
        u = np.clip(u, 1e-6, 1 - 1e-6)
        return np.log(u / (1 - u))

    logx = torch.log(t(X)).unsqueeze(-1)                          # (G, L, 1)
    mask = t(M)
    xm = t(xmax)[:, None]
    a = torch.nn.Parameter(t(logit(k0 / xmax[:, None])))
    b = torch.nn.Parameter(t(logit((n0 - n_min) / (n_max - n_min))))
    c = torch.nn.Parameter(t(np.log(np.clip(w0, 1e-6, None))))
    opt = torch.optim.Adam([a, b, c], lr=lr)

    def per_gene_nll():
        k = xm * torch.sigmoid(a)
        n = n_min + (n_max - n_min) * torch.sigmoid(b)
        logw = torch.log_softmax(c, dim=1)
        logk = torch.log(k)[:, None, :]
        nn_ = n[:, None, :]
        lp = (torch.log(nn_) + nn_ * logk + (nn_ - 1.0) * logx
              - 2.0 * torch.logaddexp(nn_ * logk, nn_ * logx)) + logw[:, None, :]
        return -(torch.logsumexp(lp, dim=2) * mask).sum(1)

    for _ in range(steps):
        opt.zero_grad()
        loss = per_gene_nll().sum()
        loss.backward()
        opt.step()

    with torch.no_grad():
        nll = per_gene_nll().cpu().numpy()
        k = (xm * torch.sigmoid(a)).cpu().numpy()
        n = (n_min + (n_max - n_min) * torch.sigmoid(b)).cpu().numpy()
        w = torch.softmax(c, dim=1).cpu().numpy()
    order = np.argsort(k, axis=1)
    take = lambda z: np.take_along_axis(z, order, axis=1)
    return take(k), take(n), take(w), nll


def posterior_regime(x, k1, n1, k2, n2, a, tau=None):
    """Maximum-posterior component of each value under the fitted mixture (1 selects component 2).

    ``P(z = 2 | x)`` is proportional to ``(1 - a) f2(x)`` against ``a f1(x)``, with ``f`` the
    log-logistic density. Broadcasts over (cells, genes). Genes with ``a = 1`` are single-Hill and
    always return 0.

    ``tau``, the gene's activity threshold, assigns every value at or below it to component 1. The
    mixture is fitted to the values above the threshold only, so below it the posterior is an
    extrapolation of the density tails, dominated by ``(n - 1) log x``: a high-threshold component
    with the smaller exponent would otherwise claim the cells that do not express the gene.
    """
    x = np.asarray(x, dtype=float)
    a = np.asarray(a, dtype=float)
    single = a >= 1 - 1e-9
    with np.errstate(divide="ignore", invalid="ignore"):
        l1 = np.log(np.where(single, 1.0, a)) + loglogistic_logpdf_np(x, k1, n1)
        l2 = np.log(np.where(single, 1.0, 1 - a)) + loglogistic_logpdf_np(x, k2, n2)
    reg = (l2 > l1) & ~np.broadcast_to(single, np.broadcast(x, a).shape)
    if tau is not None:
        reg = reg & ~(x <= np.asarray(tau, dtype=float))
    return reg.astype(np.int8)


def fit_hill_mle(values: List[np.ndarray], min_th: float = 0.05, n_min: float = 1.0,
                 n_max: float = 20.0, min_cells: int = 8, steps: int = 1500, lr: float = 0.05,
                 seed: int = 0, device: Optional[str] = None) -> dict:
    """Fit a single Hill and a two-component Hill mixture to every gene by maximum likelihood.

    Returns a dict of per-gene arrays: the single fit (``k``, ``n``, ``nll1``), the mixture
    (``k1``, ``n1``, ``k2``, ``n2``, ``a``, ``nll2``, components ordered by threshold), the number
    of values each was fitted to (``m``), the off fraction (``offset``), and ``valid2``, whether
    the gene had enough values for a two-component fit.
    """
    G = len(values)
    act, off = zip(*[active_values(g, min_th) for g in values])
    m = np.array([len(v) for v in act])
    k0 = np.ones((G, 1)); n0 = np.full((G, 1), 2.0)
    K0 = np.ones((G, 2)); N0 = np.full((G, 2), 2.0); W0 = np.full((G, 2), 0.5)
    valid2 = m >= min_cells
    for i, v in enumerate(act):
        if len(v) >= 2:
            k0[i, 0], n0[i, 0] = log_moment_init(v, n_min, n_max)
        if valid2[i]:
            try:
                K0[i], N0[i], W0[i] = gmm_log_init(v, n_min, n_max, seed=seed)
            except Exception:
                valid2[i] = False
    safe = [v if len(v) else np.array([1.0]) for v in act]
    k, n, _, nll1 = fit_loglogistic_mle(safe, k0, n0, np.ones((G, 1)), n_min, n_max, steps, lr, device)
    K, N, W, nll2 = fit_loglogistic_mle(safe, K0, N0, W0, n_min, n_max, steps, lr, device)
    return dict(k=k[:, 0], n=n[:, 0], nll1=nll1, k1=K[:, 0], n1=N[:, 0], k2=K[:, 1], n2=N[:, 1],
                a=W[:, 0], nll2=nll2, m=m, offset=np.array(off), valid2=valid2,
                init=dict(k1=K0[:, 0], n1=N0[:, 0], k2=K0[:, 1], n2=N0[:, 1], a=W0[:, 0]))


def bimodality_coefficient(v: np.ndarray) -> float:
    """Sarle's bimodality coefficient of a sample; above 0.555 (the uniform value) suggests bimodality."""
    from scipy.stats import kurtosis, skew
    m = len(v)
    if m < 4:
        return float("nan")
    sk = float(skew(v, bias=False))
    ku = float(kurtosis(v, fisher=True, bias=False))
    return (sk ** 2 + 1.0) / (ku + 3.0 * (m - 1) ** 2 / ((m - 2) * (m - 3)))


def fit_hill_mle_gated(values: List[np.ndarray], min_th: float = 0.05, n_min: float = 1.0,
                       n_max: float = 20.0, bimodal: bool = True, bimodality_min: float = 0.555,
                       min_k_ratio: float = 2.0, min_weight: float = 0.1, min_cells: int = 8,
                       bimodality_scale: str = 'log',
                       steps: int = 1500, lr: float = 0.05, seed: int = 0,
                       device: Optional[str] = None) -> dict:
    """Maximum-likelihood Hill fit per gene, with a two-component mixture where the data are bimodal.

    Each fit is run from two starts and the one with the higher likelihood is kept: the least-squares
    fit to the empirical CDF (as the second component, the same shape at twice the threshold), and a
    moment start (for two components, a Gaussian mixture on log expression). A gene takes two
    components only when all four hold: its active values have a bimodality coefficient above
    ``bimodality_min``; the two thresholds differ at least ``min_k_ratio``-fold; the smaller weight is
    at least ``min_weight``; and the mixture has the lower Bayesian information criterion. The
    coefficient is computed on ``log`` expression by default, the scale on which each Hill component
    is a symmetric logistic; on the ``raw`` scale a wide, right-skewed upper component lowers the
    coefficient of a clearly two-moded sample. Otherwise
    component 2 equals component 1 and the weight is 1. Genes with fewer than ``min_cells`` active
    values keep the least-squares single fit.

    Returns per-gene arrays ``k1, n1, k2, n2, a, offset, nll`` (per active value), ``mse`` and ``ks``
    (of the chosen CDF against the empirical CDF), ``bc``, ``m``, ``is_bimodal`` and ``tau``, the
    activity threshold the fit was restricted to.
    """
    from .math import fit_sigmoid, sigmoid
    G = len(values)
    act, off = zip(*[active_values(g, min_th) for g in values])
    m = np.array([len(v) for v in act])
    ok = m >= min_cells
    safe = [v if len(v) else np.array([1.0]) for v in act]
    xmax = np.array([float(v.max()) if len(v) else 1.0 for v in act])
    ls = np.array([fit_sigmoid(g, min_th=min_th, n_min=n_min, n_max=n_max)[:2] for g in values])
    ones = np.ones((G, 1))

    mom = np.array([log_moment_init(v, n_min, n_max) if len(v) >= 2 else (1.0, 2.0) for v in act])
    kA, nA, _, nllA = fit_loglogistic_mle(safe, mom[:, [0]], mom[:, [1]], ones, n_min, n_max, steps, lr, device)
    kB, nB, _, nllB = fit_loglogistic_mle(safe, ls[:, [0]], ls[:, [1]], ones, n_min, n_max, steps, lr, device)
    pickB = nllB < nllA
    k = np.where(ok, np.where(pickB, kB[:, 0], kA[:, 0]), ls[:, 0])
    n = np.where(ok, np.where(pickB, nB[:, 0], nA[:, 0]), ls[:, 1])
    nll1 = np.minimum(nllA, nllB)

    k1, n1, k2, n2, a = k.copy(), n.copy(), k.copy(), n.copy(), np.ones(G)
    nll = nll1.copy()
    if bimodality_scale not in ('log', 'raw'):
        raise ValueError(f"bimodality_scale must be 'log' or 'raw', not {bimodality_scale!r}")
    bc = np.array([bimodality_coefficient(np.log(v) if bimodality_scale == 'log' else v)
                   for v in act])
    accept = np.zeros(G, dtype=bool)
    if bimodal:
        K0 = np.stack([ls[:, 0], np.minimum(2 * ls[:, 0], xmax)], 1)
        N0 = np.stack([ls[:, 1], ls[:, 1]], 1)
        W0 = np.full((G, 2), 0.5)
        Kg, Ng, Wg = K0.copy(), N0.copy(), W0.copy()
        for i, v in enumerate(act):
            if ok[i]:
                try:
                    Kg[i], Ng[i], Wg[i] = gmm_log_init(v, n_min, n_max, seed=seed)
                except Exception:
                    pass
        Ka, Na, Wa, nll_a = fit_loglogistic_mle(safe, K0, N0, W0, n_min, n_max, steps, lr, device)
        Kb, Nb, Wb, nll_b = fit_loglogistic_mle(safe, Kg, Ng, Wg, n_min, n_max, steps, lr, device)
        pick = (nll_b < nll_a)[:, None]
        K = np.where(pick, Kb, Ka); N = np.where(pick, Nb, Na); W = np.where(pick, Wb, Wa)
        nll2 = np.minimum(nll_a, nll_b)
        logm = np.log(np.maximum(m, 1))
        dbic = (2 * nll1 + 2 * logm) - (2 * nll2 + 5 * logm)
        with np.errstate(divide="ignore", invalid="ignore"):
            accept = (ok & (bc > bimodality_min) & (K[:, 1] / K[:, 0] >= min_k_ratio)
                      & (np.minimum(W[:, 0], W[:, 1]) >= min_weight) & (dbic > 0))
        k1 = np.where(accept, K[:, 0], k1); n1 = np.where(accept, N[:, 0], n1)
        k2 = np.where(accept, K[:, 1], k2); n2 = np.where(accept, N[:, 1], n2)
        a = np.where(accept, W[:, 0], 1.0)
        nll = np.where(accept, nll2, nll1)

    mse = np.full(G, np.nan); ks = np.full(G, np.nan)
    for i, v in enumerate(act):
        if len(v) < 2:
            continue
        cdf = a[i] * sigmoid(v, k1[i], n1[i]) + (1 - a[i]) * sigmoid(v, k2[i], n2[i])
        mm = len(v)
        mse[i] = float(np.mean((cdf - np.linspace(0, 1, mm)) ** 2))
        ks[i] = float(max(np.max(np.abs(cdf - np.arange(1, mm + 1) / mm)),
                          np.max(np.abs(cdf - np.arange(0, mm) / mm))))
    per_value = np.where(ok, nll / np.maximum(m, 1), np.nan)
    tau = np.array([min_th * float(np.nanmax(g)) if np.size(g) and np.nanmax(g) > 0 else 0.0
                    for g in values])
    return dict(k1=k1, n1=n1, k2=k2, n2=n2, a=a, offset=np.array(off), nll=per_value, mse=mse,
                ks=ks, bc=bc, m=m, is_bimodal=accept, tau=tau)
