"""Dynamical character of a cell or a cell type: how settled it is, how deep its basin, how
rotational its local flow.

Because the fitted interaction matrix is asymmetric, the Hopfield energy is a quasi-potential and
the leading Jacobian eigenvalue is on its own not a sufficient stability readout: a cell can sit in
a deep, dynamically settled basin and still carry a positive leading real part. These summaries
combine settling with basin depth instead, and separate the rotational (non-normal) character out
as its own axis.

Two conventions matter and are easy to get wrong, so both are explicit here.

**Which velocity.** ``velocity="input"`` (the default, and what the paper's figures use) reads the
input RNA velocity the model was fitted to, ``adata.layers[velocity_key]``. ``velocity="model"``
recomputes the fitted field :math:`W\\varphi(x) - \\Gamma x + I`. These are different quantities:
the first is data the model was fitted against, the second is the model's own prediction. Only the
basin-depth term is model-derived either way. State which one a reported number used.

**Which population is standardized over.** The per-cell index z-scores across cells; the
per-cell-type index z-scores across cell-type means. Because the normalizing populations differ,
the cell-type mean of the per-cell index is *not* the per-cell-type index. They are separate
functions here for that reason.

All of these are relative to the dataset they are computed on. They have no absolute zero and are
not comparable across datasets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "velocity_speed",
    "attractor_index",
    "settling_score",
    "oscillation_score",
    "celltype_character",
]


def _z(v):
    v = np.asarray(v, dtype=float)
    return (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)


def _minmax(v):
    v = np.asarray(v, dtype=float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    return (v - lo) / (hi - lo + 1e-12)


def velocity_speed(adata, velocity="input", velocity_key="velocity_S", cluster_key=None,
                   spliced_key="Ms"):
    """Per-cell Euclidean velocity magnitude over the fitted genes.

    Parameters
    ----------
    adata : AnnData
        Fitted object.
    velocity : {"input", "model"}, default "input"
        ``"input"`` uses ``adata.layers[velocity_key]``, the RNA velocity the model was fitted to.
        ``"model"`` recomputes the fitted Hopfield field. See the module docstring: these are not
        the same quantity, and a reported number should say which was used.
    velocity_key : str, default "velocity_S"
        Layer holding the input velocity.
    cluster_key : str, optional
        Required when ``velocity="model"``, to evaluate each cell under its own cell-type field.
    """
    if velocity == "model":
        if cluster_key is None:
            raise ValueError('velocity="model" needs cluster_key to pick each cell\'s own field')
        from .fate import model_velocity
        _, V, _ = model_velocity(adata, cluster_key, spliced_key=spliced_key)
        return np.linalg.norm(V, axis=1)
    if velocity != "input":
        raise ValueError(f'velocity must be "input" or "model", got {velocity!r}')
    return np.linalg.norm(np.asarray(adata.layers[velocity_key]), axis=1)


def attractor_index(adata, energy_key="energy_interaction", **kwargs):
    """Per-cell attractor index: settled *and* deeply bound, as a sum of two z-scores.

    .. math::  \\mathcal{A}_i = z(-v)_i + z(-E_{\\mathrm{int}})_i

    where :math:`v` is the velocity magnitude and :math:`E_{\\mathrm{int}}` the interaction energy.
    Both terms are dimensionless and equally weighted, so :math:`\\mathcal{A}_i > 0` marks cells
    that are simultaneously slower and more deeply bound than the dataset average (attractor-like)
    and :math:`\\mathcal{A}_i < 0` marks fast, shallow, transient cells.

    Extra keyword arguments go to :func:`velocity_speed`, including ``velocity``.

    Returns
    -------
    ndarray, shape (n_cells,)

    Notes
    -----
    The interaction energy is computed per cell under that cell's *own* cluster matrix
    :math:`W^{(c)}`, so z-scoring it across all cells compares depths taken from different fitted
    quasi-potentials. The index ranks cells by dynamical character, not by position in one shared
    landscape.
    """
    speed = velocity_speed(adata, **kwargs)
    eint = np.asarray(adata.obs[energy_key].values, dtype=float)
    return _z(-speed) + _z(-eint)


def settling_score(adata, **kwargs):
    """Per-cell settling on [0, 1]: ``1 - minmax(velocity magnitude)``, so 1 is fully settled.

    Extra keyword arguments go to :func:`velocity_speed`.

    Notes
    -----
    Min-max rather than percentile rescaling, so a single extreme cell compresses the whole axis.
    Read it as an ordering within one dataset, not as a calibrated score.
    """
    return 1.0 - _minmax(velocity_speed(adata, **kwargs))


def oscillation_score(adata, rotational_key="jacobian_rotational"):
    """Per-cell oscillation on [0, 1]: ``minmax`` of the Frobenius norm of the antisymmetric
    Jacobian part :math:`A(x) = \\tfrac{1}{2}(J - J^{T})`, the local rotational character.

    Populate ``rotational_key`` with :func:`~scHopfield.tools.compute_rotational_part` first. The
    same min-max caveat as :func:`settling_score` applies.
    """
    return _minmax(np.asarray(adata.obs[rotational_key].values, dtype=float))


def celltype_character(adata, cluster_key, energy_key="energy_interaction",
                       rotational_key="jacobian_rotational", **kwargs):
    """Per-cell-type character table: attractor index, settling and oscillation over cluster means.

    This is the cell-type-level counterpart of the per-cell functions above, and it is deliberately
    *not* the group mean of them: the standardizations run over the K cell-type means rather than
    over the M cells, so the two disagree by construction. Use this one for a cell-type map and the
    per-cell ones for a per-cell map, and do not mix them within a figure panel.

    Extra keyword arguments go to :func:`velocity_speed`.

    Returns
    -------
    DataFrame
        Indexed by cell type, with columns ``speed``, ``energy``, ``oscillation``, ``attractor``,
        ``settling``, ``oscillation_n``.
    """
    df = pd.DataFrame({
        "cluster": adata.obs[cluster_key].astype(str).values,
        "speed": velocity_speed(adata, **kwargs),
        "energy": np.asarray(adata.obs[energy_key].values, dtype=float),
        "oscillation": np.asarray(adata.obs[rotational_key].values, dtype=float),
    }).groupby("cluster").mean()
    df["attractor"] = _z(-df["speed"]) + _z(-df["energy"])
    df["settling"] = 1.0 - _minmax(df["speed"])
    df["oscillation_n"] = _minmax(df["oscillation"])
    return df
