"""Velocity estimation utilities."""

import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from typing import Union


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
    X = np.asarray(X, dtype=float)

    t = np.asarray(adata.obs[pseudotime_key].values, dtype=float) * scale
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
    V  = dX / (dT[:, None] + 1e-6)
    V[dT == 0] = 0.0

    if copy:
        return V

    adata.layers[store_key] = V
    return None
