"""Velocity estimation utilities."""

import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from typing import Optional, Union


def estimate_velocity_from_pseudotime(
    adata: AnnData,
    pseudotime_key: str = 'Pseudotime',
    spliced_key: str = 'Ms',
    connectivity_key: str = 'connectivities',
    mode: str = 'forward',
    scale: float = 1.0,
    store_key: str = 'velocity_S',
    copy: bool = False,
) -> Union[AnnData, np.ndarray]:
    """
    Estimate RNA velocity from pseudotime ordering.

    For each cell, the velocity is approximated as the weighted mean expression
    change towards pseudotime-future neighbours divided by the corresponding
    mean pseudotime advance:

    .. math::

        v_i = \\frac{\\sum_j p_{ij}(x_j - x_i)}{\\sum_j p_{ij}(t_j - t_i) + \\varepsilon}

    where :math:`p_{ij}` are row-normalised weights derived from the graph
    connectivity restricted to forward neighbours (:math:`t_j > t_i`).

    Useful when splicing-based velocity is unavailable, e.g. when working with
    datasets that only provide pseudotime and a neighbour graph.

    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    pseudotime_key : str, optional (default: ``'Pseudotime'``)
        Key in ``adata.obs`` containing pseudotime values.
    spliced_key : str, optional (default: ``'Ms'``)
        Key in ``adata.layers`` for the expression matrix used to compute
        velocity (typically moment-smoothed spliced counts).
    connectivity_key : str, optional (default: ``'connectivities'``)
        Key in ``adata.obsp`` for the cell-cell connectivity/adjacency matrix.
    mode : str, optional (default: ``'forward'``)
        ``'forward'`` — only neighbours with :math:`t_j > t_i` contribute
        (directed along pseudotime).
        ``'central'`` — all non-zero neighbours contribute (signed velocity).
    scale : float, optional (default: ``1.0``)
        Multiplicative factor applied to the pseudotime values before computing
        finite differences.  Larger values increase the overall velocity
        magnitude.
    store_key : str, optional (default: ``'velocity_S'``)
        Key under which the estimated velocity matrix is stored in
        ``adata.layers``.  Only used when ``copy=False``.
    copy : bool, optional (default: ``False``)
        If ``True``, return the velocity matrix as a NumPy array without
        modifying *adata*.

    Returns
    -------
    numpy.ndarray or None
        If ``copy=True``, returns the velocity matrix of shape
        ``(n_cells, n_genes)``.  Otherwise stores it in
        ``adata.layers[store_key]`` and returns ``None``.
    """
    X = adata.layers[spliced_key]
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    t = np.asarray(adata.obs[pseudotime_key].values, dtype=np.float32) * scale
    A = adata.obsp[connectivity_key]

    rows, cols = A.nonzero()
    delta_t = t[cols] - t[rows]

    Delta_T = sp.csr_matrix((delta_t, (rows, cols)), shape=A.shape)

    if mode == 'forward':
        mask = Delta_T.data > 0
    else:
        mask = Delta_T.data != 0

    P = sp.csr_matrix(
        (np.asarray(A[rows[mask], cols[mask]]).ravel(), (rows[mask], cols[mask])),
        shape=A.shape,
    )

    row_sums = np.asarray(P.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    P_norm = sp.diags(1.0 / row_sums).dot(P)

    E_x = np.asarray(P_norm.dot(X))
    E_t = np.asarray(P_norm.dot(t))

    dX = E_x - X
    dT = E_t - t
    # Floor |dT| relative to the pseudotime scale (sign preserved). A fixed 1e-6 floor
    # let cells whose forward neighbours have near-equal pseudotime (dT ~ 0) produce
    # exploding velocities (outliers hundreds of times the median), which corrupt both
    # the GRN fit and the flow plots.
    nz = np.abs(dT[dT != 0])
    dt_floor = 1e-2 * float(np.median(nz)) if nz.size else 1e-6
    sign = np.sign(dT); sign[sign == 0] = 1.0
    denom = sign * np.maximum(np.abs(dT), dt_floor)
    V = dX / denom[:, None]
    V[dT == 0] = 0.0
    # Winsorize extreme per-cell velocity magnitudes at the 99th percentile (direction
    # preserved) so a few residual outliers do not dominate the fit or the field.
    mag = np.linalg.norm(V, axis=1)
    pos = mag > 0
    if pos.any():
        cap = float(np.percentile(mag[pos], 99))
        if cap > 0:
            V = V * np.minimum(1.0, cap / (mag + 1e-12))[:, None]

    if copy:
        return V

    adata.layers[store_key] = V
    return None


def prepare_dataset(
    adata: AnnData,
    n_top_genes: int = 2000,
    velocity_mode: str = 'steady_state',
    spliced_key: str = 'Ms',
    velocity_key: str = 'velocity_S',
    degradation_key: str = 'gamma',
    used_key: str = 'scHopfield_used',
    n_pcs: int = 30,
    n_neighbors: int = 30,
    min_shared_counts: int = 20,
    fit_sigmoids: bool = True,
    copy: bool = False,
) -> Optional[AnnData]:
    """Preprocess a raw dataset into a scHopfield-ready AnnData.

    A single entry point for the "make my data scHopfield-ready" boilerplate that
    was previously copy-pasted per dataset. It runs (as needed) filtering,
    normalization, HVG selection, moment smoothing, steady-state RNA velocity,
    keeps genes with a finite positive degradation rate, and fits the per-gene
    sigmoid activations. The result carries everything downstream steps expect:

    - ``layers[spliced_key]`` (``'Ms'``): moment-smoothed expression
    - ``layers[velocity_key]`` (``'velocity_S'``): RNA velocity
    - ``layers['sigmoid']``: sigmoid-activated expression
    - ``var[degradation_key]`` (``'gamma'``): degradation rates
    - ``var[used_key]`` (``'scHopfield_used'``): modelled-gene mask
    - ``obsp['connectivities']``: neighbour graph

    Datasets that already carry moments (``layers['Ms']``) skip re-filtering; only
    the missing pieces (neighbours, velocity, sigmoids) are added.

    Parameters
    ----------
    adata
        Raw or partially processed annotated data. Needs spliced/unspliced counts
        (for velocity) unless ``layers['velocity_S']`` is already present.
    n_top_genes
        Number of highly variable genes to keep when starting from raw counts.
    velocity_mode
        scVelo velocity mode (``'steady_state'``, ``'stochastic'``, ...).
    spliced_key, velocity_key, degradation_key, used_key
        Output layer/column names (defaults match the rest of scHopfield).
    n_pcs, n_neighbors
        Moment/neighbour-graph parameters.
    min_shared_counts
        scVelo gene filter threshold (only when starting from raw counts).
    fit_sigmoids
        If ``True``, fit the per-gene sigmoids and populate ``layers['sigmoid']``.
    copy
        If ``True``, operate on and return a copy; otherwise modify in place and
        return ``None``.

    Returns
    -------
    :class:`~anndata.AnnData` or None
        The processed object if ``copy=True``, else ``None``.

    Notes
    -----
    Requires `scVelo <https://scvelo.readthedocs.io>`_. Install with
    ``pip install scvelo``.
    """
    try:
        import scanpy as sc
        import scvelo as scv
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise ImportError(
            "prepare_dataset requires scanpy and scvelo. "
            "Install with `pip install scvelo scanpy`."
        ) from exc
    from .sigmoid_fitting import compute_sigmoid, fit_all_sigmoids

    a = adata.copy() if copy else adata

    if spliced_key not in a.layers:
        scv.pp.filter_genes(a, min_shared_counts=min_shared_counts)
        scv.pp.normalize_per_cell(a)
        sc.pp.log1p(a)
        sc.pp.highly_variable_genes(a, n_top_genes=min(n_top_genes, a.n_vars))
        a._inplace_subset_var(a.var['highly_variable'].values)
        scv.pp.moments(a, n_pcs=n_pcs, n_neighbors=n_neighbors)
    elif 'connectivities' not in a.obsp:
        scv.pp.moments(a, n_pcs=n_pcs, n_neighbors=n_neighbors)

    if velocity_key not in a.layers:
        scv.tl.velocity(a, mode=velocity_mode)
        a.layers[velocity_key] = a.layers['velocity']
        a.var[degradation_key] = np.asarray(a.var['velocity_gamma']).astype(np.float32)
    elif degradation_key not in a.var:
        a.var[degradation_key] = np.float32(0.1)

    # keep genes with usable degradation + finite velocity
    gamma = np.asarray(a.var[degradation_key].values, dtype=float)
    finite_g = np.isfinite(gamma) & (gamma > 0)
    vel = np.asarray(a.layers[velocity_key])
    finite_v = np.isfinite(vel).all(axis=0)
    keep = finite_g & finite_v
    if not keep.all():
        a._inplace_subset_var(keep)

    a.var[used_key] = True
    if fit_sigmoids and 'sigmoid' not in a.layers:
        fit_all_sigmoids(a, genes=a.var[used_key].values, spliced_key=spliced_key)
        compute_sigmoid(a, spliced_key=spliced_key)

    if 'connectivities' not in a.obsp:
        raise ValueError("No neighbour graph in adata.obsp['connectivities'].")

    return a if copy else None
