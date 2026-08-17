"""Projection-free perturbation readout: terminal-state fate probabilities under an absorbing
Markov chain, and the per-gene permutation null that calibrates them.

This is the readout the paper reports for single-gene knockouts. It replaces the projected-flow
lineage bias (:func:`~scHopfield.tools.compute_lineage_bias`), which scores the direction of the
knockout displacement *after* projection to a two-dimensional embedding and therefore inherits two
weaknesses of that projection: it compresses a high-dimensional displacement into a plane, and the
projected direction is dominated by whichever gene carries the largest raw coordinate. A gene with
no outgoing edges but very high expression (the lncRNA *Malat1* is the standing example) produces a
displacement that is almost entirely its own coordinate, which projects to a coherent flow along
that gene's expression gradient and reads as a large lineage bias even though the knockout changes
no other gene.

The construction here is inert for such perturbations by design. Fate probabilities are read from
the model velocity, the knocked-out gene's own coordinate is neutralized before the transition
matrix is rebuilt, and a gene whose outgoing column of ``W`` is zero therefore leaves the velocity,
the transition matrix and the fate probabilities untouched, giving exactly zero.

The engine is native, with no CellRank dependency, but the core is the same absorbing-chain solve:
build a cell-cell transition matrix from the model velocity with the correlation kernel, make the
terminal-state cells absorbing, and solve :math:`(I - Q) B = R` for each cell's absorption
probability.

Typical use::

    from scHopfield.tools import fate_shift, permutation_null_floor

    res = fate_shift(adata, "clusters", A=["Alpha"], B=["Beta"], ko_genes=["Arx", "Pax4"])
    res["shift"]["Arx"]        # mean change in the alpha-versus-beta split, decider cells only

See the Methods section "Fate-Probability Lineage Effect" for the definitions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from sklearn.neighbors import NearestNeighbors

from .._utils.io import get_genes_used

__all__ = [
    "model_velocity",
    "fate_transition_matrix",
    "terminal_states",
    "fate_probabilities",
    "split_fraction",
    "decider_mask",
    "fate_shift",
    "permutation_null_floor",
]


def model_velocity(adata, cluster_key, genes_used=None, ko_gene=None, ko_level=0.0,
                   spliced_key="Ms"):
    """Model velocity at each cell's observed state, under its own cell-type field.

    Evaluates :math:`\\dot{x} = W_c \\varphi(x) - \\Gamma x + I` for every cell, using the fitted
    parameters of the cluster that cell belongs to. This is the *model* velocity, not the input RNA
    velocity the model was fitted to.

    Parameters
    ----------
    adata : AnnData
        Fitted object, carrying per-cluster parameters.
    cluster_key : str
        ``adata.obs`` column holding the cell-type assignment.
    genes_used : array-like of int, optional
        Gene indices the model was fitted on. Defaults to the fitted selection.
    ko_gene : str, optional
        Gene to hold fixed in the *input* at ``ko_level``, so its regulatory drive is set by the
        clamp. This is the knockout, and matches ``simulate_shift_ode``'s fixed-gene convention.
    ko_level : float, default 0.0
        Clamp level. Zero is a knockout; intermediate values give a dose sweep.
    spliced_key : str, default "Ms"
        Layer holding the expression state.

    Returns
    -------
    X, V, names : ndarray, ndarray, ndarray
        States, velocities and gene names, all restricted to ``genes_used``.
    """
    from ..dynamics.solver import create_solver          # local: dynamics imports tools

    if genes_used is None:
        genes_used = get_genes_used(adata)
    X = np.asarray(adata.layers[spliced_key])[:, genes_used].astype(float)
    names = np.asarray(adata.var_names.values)[genes_used]
    V = np.zeros_like(X)
    clusters = adata.obs[cluster_key].astype(str).values
    gidx = list(names).index(ko_gene) if (ko_gene is not None and ko_gene in names) else None
    for c in pd.unique(clusters):
        sel = np.where(clusters == c)[0]
        try:
            solver = create_solver(adata, c, spliced_key=spliced_key)
        except Exception:
            continue
        Xc = X[sel].copy()
        if gidx is not None:
            Xc[:, gidx] = ko_level
        V[sel] = solver.dynamics_batch(Xc, 0.0)
    return X, V, names


def fate_transition_matrix(X, V, knn_idx, sigma=0.05):
    """Correlation-kernel cell-cell transition matrix, the same kernel used for the embedding flow.

    :math:`T_{ij} \\propto \\exp(\\mathrm{corr}(V_i,\\, X_j - X_i)/\\sigma)` over the k nearest
    neighbors j of cell i, row-normalized. Note that the correlation normalizes by each cell's
    velocity norm, so the kernel is invariant to the overall scale of a cell's velocity; what moves
    the readout is the direction, and the displacement scale *relative* to the wild-type velocity.

    Parameters
    ----------
    X, V : ndarray, shape (n_cells, n_genes)
        States and velocities in the same gene space.
    knn_idx : ndarray, shape (n_cells, k)
        Neighbor indices, self excluded.
    sigma : float, default 0.05
        Kernel bandwidth. Smaller values sharpen the flow.
    """
    n = X.shape[0]
    rows, cols, vals = [], [], []
    Vc = V - V.mean(1, keepdims=True)
    vnorm = np.linalg.norm(Vc, axis=1) + 1e-12
    for i in range(n):
        js = knn_idx[i]
        dX = X[js] - X[i]
        dXc = dX - dX.mean(1, keepdims=True)
        corr = (dXc @ Vc[i]) / ((np.linalg.norm(dXc, axis=1) + 1e-12) * vnorm[i])
        w = np.exp(corr / sigma)
        w /= w.sum() + 1e-12
        rows.extend([i] * len(js))
        cols.extend(js.tolist())
        vals.extend(w.tolist())
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def terminal_states(clusters, terminal_clusters, vmag, frac=0.3, min_cells=5):
    """Absorbing sets: the field's stable cells inside each known terminal cell type.

    Within each terminal cluster, takes the ``frac`` of cells with the smallest model-velocity
    magnitude, that is, the cells nearest the fitted attractor. Using the field's own stable cells
    rather than the whole cluster keeps the absorbing set consistent with the dynamics.

    Returns
    -------
    dict
        ``{cluster name: array of absorbing cell indices}``.
    """
    out = {}
    for c in terminal_clusters:
        idx = np.where(np.asarray(clusters) == str(c))[0]
        if len(idx) == 0:
            continue
        k = max(min_cells, int(round(len(idx) * frac)))
        out[str(c)] = idx[np.argsort(vmag[idx])[:k]]
    return out


def fate_probabilities(T, term_sets):
    """Absorption probabilities of the absorbing Markov chain induced by ``T``.

    Splits ``T`` into transient and absorbing blocks and solves :math:`(I - Q) B = R`. Rows are
    renormalized after the solve, which guards against probability leaking through a transient sink.

    Parameters
    ----------
    T : sparse matrix, shape (n_cells, n_cells)
        Row-normalized transition matrix.
    term_sets : dict
        ``{state: array of absorbing cell indices}``, from :func:`terminal_states`.

    Returns
    -------
    fate, states : ndarray (n_cells, n_states), list of str
    """
    n = T.shape[0]
    states = list(term_sets)
    is_term = np.zeros(n, bool)
    for s in states:
        is_term[term_sets[s]] = True
    trans = np.where(~is_term)[0]
    Tt = T.tocsr()
    Q = Tt[trans][:, trans].tocsc()
    R = np.zeros((len(trans), len(states)))
    for k, s in enumerate(states):
        R[:, k] = np.asarray(Tt[trans][:, term_sets[s]].sum(1)).ravel()
    B = np.atleast_2d(spsolve(sp.eye(len(trans), format="csc") - Q, R))
    if B.shape[0] != len(trans):
        B = B.T
    fate = np.zeros((n, len(states)))
    fate[trans] = np.clip(B, 0, None)
    fate[trans] /= fate[trans].sum(1, keepdims=True) + 1e-12
    for k, s in enumerate(states):
        fate[term_sets[s], k] = 1.0
    return fate, states


def split_fraction(fate, a_cols, b_cols):
    """Per-cell A-versus-B split fraction :math:`p_A / (p_A + p_B)`.

    Normalizing by the two arms of one decision keeps a globally dominant terminal fate from
    swamping every knockout, so the readout reflects the decision rather than the overall funnel.
    """
    sa = fate[:, a_cols].sum(1)
    sb = fate[:, b_cols].sum(1)
    return sa / (sa + sb + 1e-12)


def decider_mask(split_wt, transitional=None, clusters=None, min_cells=10):
    """Cells the fate shift is summarized over: the transitional "decider" population.

    Averaging over every cell understates the effect, because a committed cell (wild-type split
    near 0 or 1) contributes almost exactly zero no matter what the perturbation does. The summary
    is taken over the cells whose fate is still labile: an explicit transitional population when
    the dataset names one, otherwise the labile-split band, falling back to all cells when a
    decision is so sharp that no such band exists.

    Returns
    -------
    ndarray of bool, shape (n_cells,)
    """
    if transitional is not None and clusters is not None:
        m = np.isin(np.asarray(clusters), [str(c) for c in transitional])
        if m.sum() >= min_cells:
            return m
    m = (split_wt > 0.1) & (split_wt < 0.9)
    if m.sum() >= min_cells:
        return m
    return np.ones(len(split_wt), bool)


def _knn_from_basis(adata, basis, n_neighbors):
    emb = np.asarray(adata.obsm[f"X_{basis or 'umap'}"])[:, :2]
    return NearestNeighbors(n_neighbors=n_neighbors + 1).fit(emb).kneighbors(
        return_distance=False)[:, 1:]


def fate_shift(adata, cluster_key, A, B, ko_genes, basis=None, n_neighbors=30, sigma=0.05,
               frac=0.3, transitional=None, spliced_key="Ms", neutralize="drop"):
    """Change a knockout induces in the A-versus-B terminal-fate split, over the decider cells.

    This is the reported single-gene lineage-effect readout:

    .. math::
        \\Delta_{\\mathrm{fate}} = \\big\\langle\\, s^{\\mathrm{KO}} - s^{\\mathrm{WT}}
        \\,\\big\\rangle_{\\text{decider cells}}, \\qquad s = p_A / (p_A + p_B)

    The knockout gene's own velocity coordinate is neutralized before the knockout transition
    matrix is built, so only the downstream, propagated response can move fate. A gene with no
    outgoing edges therefore gives exactly zero rather than a spurious bias.

    Parameters
    ----------
    adata : AnnData
        Fitted object.
    cluster_key : str
        ``adata.obs`` column holding the cell-type assignment.
    A, B : sequence of str
        Cluster names forming the two arms of the decision. Positive shift means the knockout
        redirects fate toward A.
    ko_genes : sequence of str
        Genes to knock out, one at a time.
    basis : str, optional
        Embedding used only to build the neighbor graph. Defaults to ``"umap"``.
    n_neighbors, sigma, frac : int, float, float
        Neighbor count, kernel bandwidth, and the stable-cell fraction defining absorbing sets.
    transitional : sequence of str, optional
        Explicit transitional clusters for :func:`decider_mask`. Without it the labile-split band
        is used.
    neutralize : {"drop", "hold"}, default "drop"
        How the knocked-out gene's own coordinate is removed from the readout, so that only the
        downstream response can move fate. ``"drop"`` removes the gene from the kernel entirely and
        recomputes a gene-matched wild-type baseline in that reduced space; ``"hold"`` keeps the
        gene but pins its knockout velocity to its wild-type value. Both give exactly zero for a
        gene with no outgoing edges, which is the property that matters, but they are not
        numerically identical because the correlation kernel is nonlinear in the gene set. ``drop``
        is the stricter of the two and is what the reported knockout-recovery panel uses; ``hold``
        is cheaper, since the wild-type chain is solved once for all genes.

    Returns
    -------
    dict
        ``shift`` (Series over ``ko_genes``), ``pvals`` (Wilcoxon signed-rank over the decider
        cells), ``split_wt``, ``decider`` (boolean mask), ``n_decider``, and ``states``.
        With ``neutralize="drop"`` the wild-type baseline differs per gene, so ``split_wt`` and
        ``decider`` are the full-gene-space wild type, reported for reference.

    Notes
    -----
    The Wilcoxon p-value tests whether the per-cell changes are centered at zero. It does not test
    whether the effect exceeds what an unstructured perturbation of the same magnitude would
    produce; use :func:`permutation_null_floor` for that.
    """
    from scipy.stats import wilcoxon

    g = get_genes_used(adata)
    clusters = adata.obs[cluster_key].astype(str).values
    knn = _knn_from_basis(adata, basis, n_neighbors)
    X, V_wt, names = model_velocity(adata, cluster_key, g, spliced_key=spliced_key)
    vmag = np.linalg.norm(V_wt, axis=1)

    arms = [str(c) for c in list(A) + list(B)]
    term = terminal_states(clusters, arms, vmag, frac=frac)
    fate_wt, states = fate_probabilities(fate_transition_matrix(X, V_wt, knn, sigma), term)
    sidx = {s: k for k, s in enumerate(states)}
    a_cols = [sidx[str(c)] for c in A if str(c) in sidx]
    b_cols = [sidx[str(c)] for c in B if str(c) in sidx]
    if not a_cols or not b_cols:
        raise ValueError("neither arm of the decision resolved to a terminal state; check A/B "
                         f"against the cluster labels in adata.obs[{cluster_key!r}]")

    split_wt = split_fraction(fate_wt, a_cols, b_cols)
    focus = decider_mask(split_wt, transitional, clusters)
    name_list = list(names)

    if neutralize not in ("drop", "hold"):
        raise ValueError(f'neutralize must be "drop" or "hold", got {neutralize!r}')

    shift, pvals = {}, {}
    for gene in dict.fromkeys(ko_genes):
        gi = name_list.index(gene) if gene in name_list else None
        if gi is None:
            shift[gene], pvals[gene] = np.nan, np.nan       # gene outside the fitted selection
            continue
        _, V_ko, _ = model_velocity(adata, cluster_key, g, ko_gene=gene, spliced_key=spliced_key)
        if neutralize == "drop":
            keep = np.ones(len(names), bool)
            keep[gi] = False
            base, _ = fate_probabilities(
                fate_transition_matrix(X[:, keep], V_wt[:, keep], knn, sigma), term)
            s_base = split_fraction(base, a_cols, b_cols)   # gene-matched wild-type baseline
            fate_ko, _ = fate_probabilities(
                fate_transition_matrix(X[:, keep], V_ko[:, keep], knn, sigma), term)
        else:
            V_ko = V_ko.copy()
            V_ko[:, gi] = V_wt[:, gi]                       # pin the gene's own coordinate to WT
            s_base = split_wt
            fate_ko, _ = fate_probabilities(fate_transition_matrix(X, V_ko, knn, sigma), term)
        d = (split_fraction(fate_ko, a_cols, b_cols) - s_base)[decider_mask(s_base, transitional,
                                                                           clusters)]
        shift[gene] = float(np.nanmean(d))
        nz = np.abs(d) > 1e-12
        pvals[gene] = float(wilcoxon(d[nz]).pvalue) if nz.sum() > 10 else np.nan

    return {"shift": pd.Series(shift), "pvals": pd.Series(pvals), "split_wt": split_wt,
            "decider": focus, "n_decider": int(focus.sum()), "states": states}


def permutation_null_floor(X, V_wt, displacement, knn_idx, term_sets, a_cols, b_cols,
                           n=30, seed=0, sigma=0.05, transitional=None, clusters=None,
                           percentile=95.0):
    """Per-gene noise floor for a fate shift, from a magnitude-preserving permutation null.

    Predicted knockout displacements differ across genes by orders of magnitude, so a fixed
    threshold would compare a strong regulator against a weak one on unequal terms. Each knockout
    is instead calibrated against its own null: the *same* displacement is reapplied to permuted
    cells,

    .. math::  \\tilde{V} = V^{\\mathrm{WT}} + \\Pi_\\pi D,

    which preserves the multiset of per-cell displacement vectors exactly, and with it
    :math:`\\lVert D \\rVert_F`, while destroying the pairing between a displacement and the cell
    state it acts on. Every draw goes through the identical readout, including the same decider
    aggregation, so effect and floor stay on one scale.

    Parameters
    ----------
    X, V_wt : ndarray, shape (n_cells, n_genes)
        Wild-type states and velocities, already restricted to the genes the readout sees (drop the
        knocked-out gene's coordinate from both before calling).
    displacement : ndarray, shape (n_cells, n_genes)
        The predicted knockout displacement to permute, in the same space.
    term_sets, a_cols, b_cols
        Absorbing sets and the column indices of the two arms, as in :func:`fate_shift`.
    n : int, default 30
        Number of permutation draws.
    seed : int, default 0
        Fixed by default so every gene, and every method being compared, is scored against the same
        permutations. That matching is deliberate; it also makes the per-gene floors statistically
        dependent, so a panel-level count of resolved genes is not a set of independent tests.
    percentile : float, default 95.0
        Percentile of the absolute null shifts taken as the floor.

    Returns
    -------
    floor, null_shifts : float, ndarray
        The floor, and the ``n`` absolute null shifts it was taken from. A knockout is resolved
        when ``abs(observed_shift) > floor``.

    Notes
    -----
    Magnitude preservation holds in the aggregate, not per cell. The transition kernel normalizes
    by each cell's velocity norm, so the readout is driven by the per-cell ratio
    ``|D_i| / |V_wt_i|``, and permuting rows reassigns large displacements onto cells with small
    wild-type velocity. The null therefore controls for overall displacement magnitude and for the
    absence of spatial structure, not for the per-cell scale of the perturbation.

    This is a calibration threshold, not a p-value: no tail probability is computed and no
    multiplicity correction is applied across a panel. With ``n=30`` the 95th percentile
    interpolates between the second and third largest absolute null shift.
    """
    rng = np.random.default_rng(seed)
    fate_wt, _ = fate_probabilities(fate_transition_matrix(X, V_wt, knn_idx, sigma), term_sets)
    split_wt = split_fraction(fate_wt, a_cols, b_cols)
    focus = decider_mask(split_wt, transitional, clusters)
    null = []
    for _ in range(n):
        p = rng.permutation(displacement.shape[0])
        T = fate_transition_matrix(X, V_wt + displacement[p], knn_idx, sigma)
        fate, _ = fate_probabilities(T, term_sets)
        null.append(float(np.nanmean((split_fraction(fate, a_cols, b_cols) - split_wt)[focus])))
    null = np.abs(np.asarray(null))
    return float(np.percentile(null, percentile)), null
