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

import multiprocessing as mp
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
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
    "fate_scaffold",
    "lineage_pair_axes",
    "perturbed_fate",
    "perturbed_fates",
    "pairwise_fate_bias",
    "per_cell_fate_shift",
    "dose_fate_bias",
    "fate_embedding_flow",
    "commitment_time",
]

#: One lineage decision, as ``(A_clusters, B_clusters, A_name, B_name)``. Every readout in this
#: module reports a decision the same way round: positive means the perturbation moves cells
#: toward arm A.
LineagePair = Tuple[Sequence[str], Sequence[str], str, str]

#: The wild-type quantities every perturbed evaluation reuses, as built by :func:`fate_scaffold`.
FateScaffold = Dict[str, Any]


def model_velocity(adata: AnnData, cluster_key: str, genes_used=None,
                   ko_gene: Optional[str] = None, ko_level: float = 0.0,
                   clamp: Optional[Mapping[str, float]] = None,
                   spliced_key: str = "Ms") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        Clamp level for ``ko_gene``. Zero is a knockout; intermediate values give a dose sweep.
    clamp : mapping of str to float, optional
        Several genes held at once, as ``{gene: level}``. A joint knockout cannot be expressed
        through the single-gene ``ko_gene`` clamp, and this is what the combinatorial readouts
        use. Merged on top of ``ko_gene``, so the two may be combined. Genes outside the fitted
        selection are ignored, exactly as an unmatched ``ko_gene`` is.
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

    held: Dict[str, float] = {}
    if ko_gene is not None:
        held[ko_gene] = float(ko_level)
    if clamp:
        held.update({g: float(lvl) for g, lvl in clamp.items()})
    name_list = list(names)
    fixed = [(name_list.index(g), lvl) for g, lvl in held.items() if g in name_list]

    for c in pd.unique(clusters):
        sel = np.where(clusters == c)[0]
        try:
            solver = create_solver(adata, c, spliced_key=spliced_key)
        except Exception:
            continue
        Xc = X[sel].copy()
        for gi, lvl in fixed:
            Xc[:, gi] = lvl
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


# --------------------------------------------------------------------------------------------- #
# Shared wild-type scaffold
#
# Every perturbed evaluation below re-solves the absorbing chain against the same wild-type
# quantities: the neighbor graph, the wild-type model velocity, the absorbing sets and the
# wild-type fate probabilities. Building them once and passing the result keeps a screen over
# hundreds of perturbations from recomputing the expensive part hundreds of times.
# --------------------------------------------------------------------------------------------- #
def fate_scaffold(adata: AnnData, cluster_key: str, lineage_pairs: Sequence[LineagePair],
                  basis: Optional[str] = None, n_neighbors: int = 30, sigma: float = 0.05,
                  frac: float = 0.3, spliced_key: str = "Ms") -> FateScaffold:
    """Wild-type quantities every perturbed fate evaluation reuses.

    The absorbing sets cover every cluster named as an arm of any decision in ``lineage_pairs``,
    so both fates of every decision are absorbing macrostates, including a progenitor arm.

    Parameters
    ----------
    adata : AnnData
        Fitted object.
    cluster_key : str
        ``adata.obs`` column holding the cell-type assignment.
    lineage_pairs : sequence of LineagePair
        The decisions to be scored, as ``(A_clusters, B_clusters, A_name, B_name)``.
    basis : str, optional
        Embedding used only to build the neighbor graph. Defaults to ``"umap"``.
    n_neighbors, sigma, frac : int, float, float
        Neighbor count, kernel bandwidth, and the stable-cell fraction defining absorbing sets.
    spliced_key : str, default "Ms"
        Layer holding the expression state.

    Returns
    -------
    dict
        ``X``, ``V_wt``, ``names``, ``knn``, ``term``, ``clusters``, ``vmag``, ``fate_wt``,
        ``states``, ``T_wt``, ``sidx`` (state name to column), ``g`` (fitted gene indices),
        ``sigma``, ``basis`` and ``spliced_key``.
    """
    g = get_genes_used(adata)
    clusters = adata.obs[cluster_key].astype(str).values
    knn = _knn_from_basis(adata, basis, n_neighbors)
    X, V_wt, names = model_velocity(adata, cluster_key, g, spliced_key=spliced_key)
    vmag = np.linalg.norm(V_wt, axis=1)
    arms = sorted({str(c) for A, B, An, Bn in lineage_pairs for c in list(A) + list(B)})
    term = terminal_states(clusters, arms, vmag, frac=frac)
    T_wt = fate_transition_matrix(X, V_wt, knn, sigma)
    fate_wt, states = fate_probabilities(T_wt, term)
    return dict(g=g, X=X, V_wt=V_wt, names=list(names), knn=knn, term=term, clusters=clusters,
                vmag=vmag, fate_wt=fate_wt, states=states, T_wt=T_wt,
                sidx={s: k for k, s in enumerate(states)}, sigma=sigma,
                basis=basis, spliced_key=spliced_key)


def lineage_pair_axes(scaffold: FateScaffold, lineage_pairs: Sequence[LineagePair],
                      transitional: Optional[Mapping[Tuple[str, str], Sequence[str]]] = None
                      ) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Per decision, the two arms' absorbing-state columns, the wild-type split, and the deciders.

    Parameters
    ----------
    scaffold : dict
        From :func:`fate_scaffold`.
    lineage_pairs : sequence of LineagePair
        The decisions to resolve.
    transitional : mapping, optional
        ``{(A_name, B_name): [cluster, ...]}`` naming each decision's transitional population.
        A decision left out falls back to the labile-split band; see :func:`decider_mask`.

    Returns
    -------
    dict
        ``{(A_name, B_name): {"Ac", "Bc", "split_wt", "focus"}}``.
    """
    axes: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for A, B, An, Bn in lineage_pairs:
        a_cols = [scaffold["sidx"][str(c)] for c in A if str(c) in scaffold["sidx"]]
        b_cols = [scaffold["sidx"][str(c)] for c in B if str(c) in scaffold["sidx"]]
        split_wt = split_fraction(scaffold["fate_wt"], a_cols, b_cols)
        trans = (transitional or {}).get((An, Bn))
        focus = decider_mask(split_wt, trans, scaffold["clusters"])
        axes[(An, Bn)] = dict(Ac=a_cols, Bc=b_cols, split_wt=split_wt, focus=focus)
    return axes


def _held_levels(genes: Union[str, Sequence[str]],
                 levels: Optional[Union[float, Sequence[float]]]) -> Dict[str, float]:
    """Normalize the ``genes`` / ``levels`` argument pair into a ``{gene: level}`` clamp."""
    gene_list = [genes] if isinstance(genes, str) else list(genes)
    if levels is None:
        lvl = [0.0] * len(gene_list)
    elif np.isscalar(levels):
        lvl = [float(levels)] * len(gene_list)
    else:
        lvl = [float(v) for v in levels]
    return dict(zip(gene_list, lvl))


def _perturbed_transition(adata: AnnData, cluster_key: str, scaffold: FateScaffold,
                          genes: Union[str, Sequence[str]],
                          levels: Optional[Union[float, Sequence[float]]] = None):
    """Transition matrix with the held genes' own velocity coordinates neutralized to wild type.

    Returns ``(T, held_indices)``, or ``(None, [])`` when no held gene is in the fitted selection.
    Neutralizing each held gene's own coordinate is what makes the readout projection-free: only
    the downstream, propagated response can move fate, so a gene with no outgoing edges leaves the
    kernel untouched and scores exactly zero.
    """
    names = scaffold["names"]
    held = _held_levels(genes, levels)
    idx = [names.index(g) for g in held if g in names]
    if not idx:
        return None, []
    _, v_pert, _ = model_velocity(adata, cluster_key, scaffold["g"], clamp=held,
                                  spliced_key=scaffold.get("spliced_key", "Ms"))
    v_k = v_pert.copy()
    for gi in idx:
        v_k[:, gi] = scaffold["V_wt"][:, gi]
    return fate_transition_matrix(scaffold["X"], v_k, scaffold["knn"], scaffold["sigma"]), idx


def perturbed_fate(adata: AnnData, cluster_key: str, scaffold: FateScaffold,
                   genes: Union[str, Sequence[str]],
                   levels: Optional[Union[float, Sequence[float]]] = None) -> np.ndarray:
    """Per-cell terminal-fate probabilities with one or several genes held at a level.

    One gene at level zero is the single knockout; two genes at zero is the joint double
    knockout, which a single-gene clamp cannot express; a nonzero level is a dose. Each held
    gene's own velocity coordinate is neutralized to wild type first, so only the downstream
    response moves fate and a pure sink gives the wild-type answer exactly.

    Parameters
    ----------
    adata : AnnData
        Fitted object.
    cluster_key : str
        ``adata.obs`` column holding the cell-type assignment.
    scaffold : dict
        From :func:`fate_scaffold`.
    genes : str or sequence of str
        Gene or genes to hold.
    levels : float or sequence of float, optional
        Level per gene. A scalar applies to all of them. Defaults to zero, the knockout.

    Returns
    -------
    ndarray, shape (n_cells, n_states)
        Fate probabilities, or the scaffold's wild-type probabilities when no named gene is in
        the fitted selection.
    """
    trans, _ = _perturbed_transition(adata, cluster_key, scaffold, genes, levels)
    if trans is None:
        return scaffold["fate_wt"]
    fate, _ = fate_probabilities(trans, scaffold["term"])
    return fate


# Worker state for the forked pool below. The scaffold and the AnnData are read-only in the
# workers, so forking shares them copy-on-write instead of pickling a copy per task.
_FATE_WORKER: Dict[str, Any] = {}


def _init_fate_worker() -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass


def _fate_worker_job(genes):
    return perturbed_fate(_FATE_WORKER["adata"], _FATE_WORKER["cluster_key"],
                          _FATE_WORKER["scaffold"], list(genes))


def perturbed_fates(adata: AnnData, cluster_key: str, scaffold: FateScaffold,
                    gene_lists: Sequence[Sequence[str]], workers: int = 1) -> List[np.ndarray]:
    """:func:`perturbed_fate` over many perturbations, optionally across CPU processes.

    The readout is entirely CPU-bound, in the transition-matrix loop and the sparse solve, so
    independent perturbations parallelize cleanly. Each worker is pinned to one thread to keep
    the BLAS from oversubscribing the cores the pool is already using.

    Parameters
    ----------
    adata : AnnData
        Fitted object.
    cluster_key : str
        ``adata.obs`` column holding the cell-type assignment.
    scaffold : dict
        From :func:`fate_scaffold`.
    gene_lists : sequence of sequence of str
        One entry per perturbation; each entry is the genes held jointly for that evaluation.
    workers : int, default 1
        Processes to fork. One, or a single perturbation, runs serially in this process.

    Returns
    -------
    list of ndarray
        Fate probabilities per entry of ``gene_lists``, in order. Identical to the serial result:
        every evaluation is independent and deterministic.
    """
    tasks = [list(g) for g in gene_lists]
    if workers <= 1 or len(tasks) <= 1:
        return [perturbed_fate(adata, cluster_key, scaffold, g) for g in tasks]
    try:
        ctx = mp.get_context("fork")
    except ValueError:                       # no fork on this platform; the result is the same
        return [perturbed_fate(adata, cluster_key, scaffold, g) for g in tasks]
    _FATE_WORKER["adata"] = adata
    _FATE_WORKER["cluster_key"] = cluster_key
    _FATE_WORKER["scaffold"] = scaffold
    with ctx.Pool(min(workers, len(tasks)), initializer=_init_fate_worker) as pool:
        return pool.map(_fate_worker_job, tasks, chunksize=1)


def pairwise_fate_bias(adata: AnnData, cluster_key: str, lineage_pairs: Sequence[LineagePair],
                       genes: Sequence[str], basis: Optional[str] = None, n_neighbors: int = 30,
                       sigma: float = 0.05, frac: float = 0.3,
                       transitional: Optional[Mapping[Tuple[str, str], Sequence[str]]] = None,
                       scaffold: Optional[FateScaffold] = None, workers: int = 1
                       ) -> Dict[Tuple[str, str], Dict[str, pd.Series]]:
    """Per-decision knockout bias in the A-versus-B fate split, over the decider cells.

    This is the reported single-gene lineage-effect screen. For each candidate and each decision,

    .. math::
        \\mathrm{bias}(g) = \\big\\langle\\, s^{\\mathrm{KO}(g)} - s^{\\mathrm{WT}}
        \\,\\big\\rangle_{\\text{decider cells}}, \\qquad s = p_A / (p_A + p_B)

    Positive means the knockout shifts fate toward arm A. Each candidate is evaluated once and
    scored against every decision, which is what makes it a screen rather than a loop over
    :func:`fate_shift`.

    Parameters
    ----------
    adata : AnnData
        Fitted object.
    cluster_key : str
        ``adata.obs`` column holding the cell-type assignment.
    lineage_pairs : sequence of LineagePair
        Decisions to score.
    genes : sequence of str
        Candidate knockouts. Deduplicated, and restricted to measured genes.
    basis, n_neighbors, sigma, frac
        Passed to :func:`fate_scaffold` when ``scaffold`` is not supplied.
    transitional : mapping, optional
        Decider populations per decision; see :func:`lineage_pair_axes`.
    scaffold : dict, optional
        A scaffold to reuse instead of building one.
    workers : int, default 1
        CPU processes for the candidate evaluations; see :func:`perturbed_fates`.

    Returns
    -------
    dict
        ``{(A_name, B_name): {"bias": Series sorted ascending, "pvals": Series}}``. The p-value is
        a Wilcoxon signed-rank test over the decider cells, testing whether the per-cell changes
        are centered at zero.
    """
    from scipy.stats import wilcoxon

    scaf = scaffold or fate_scaffold(adata, cluster_key, lineage_pairs, basis=basis,
                                     n_neighbors=n_neighbors, sigma=sigma, frac=frac)
    cands = [g for g in dict.fromkeys(genes) if g in adata.var_names]
    fates = dict(zip(cands, perturbed_fates(adata, cluster_key, scaf,
                                            [[g] for g in cands], workers)))
    axes = lineage_pair_axes(scaf, lineage_pairs, transitional)

    out: Dict[Tuple[str, str], Dict[str, pd.Series]] = {}
    for A, B, An, Bn in lineage_pairs:
        ax = axes[(An, Bn)]
        bias, pvals = {}, {}
        for gene in cands:
            d = (split_fraction(fates[gene], ax["Ac"], ax["Bc"]) - ax["split_wt"])[ax["focus"]]
            bias[gene] = float(d.mean())
            nz = np.abs(d) > 1e-12
            pvals[gene] = float(wilcoxon(d[nz]).pvalue) if nz.sum() > 10 else np.nan
        out[(An, Bn)] = {"bias": pd.Series(bias).sort_values(), "pvals": pd.Series(pvals)}
    return out


def per_cell_fate_shift(adata: AnnData, cluster_key: str, lineage_pairs: Sequence[LineagePair],
                        genes: Sequence[str], basis: Optional[str] = None, n_neighbors: int = 30,
                        sigma: float = 0.05, frac: float = 0.3,
                        scaffold: Optional[FateScaffold] = None, workers: int = 1
                        ) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Per-cell change in the A-versus-B split fraction under each knockout.

    The spatial, projection-free counterpart of :func:`pairwise_fate_bias`: instead of one number
    per gene it keeps every cell's change, so the effect can be mapped over the embedding.
    Positive means the knockout redirects that cell toward arm A. A committed cell moves little
    by construction, and a pure sink moves nothing anywhere.

    Returns
    -------
    dict
        ``{(A_name, B_name): {"shift": {gene: ndarray (n_cells,)}, "split_wt": ndarray}}``. The
        wild-type split is returned so a caller can separate deciders from committed cells.
    """
    scaf = scaffold or fate_scaffold(adata, cluster_key, lineage_pairs, basis=basis,
                                     n_neighbors=n_neighbors, sigma=sigma, frac=frac)
    gene_list = [g for g in dict.fromkeys(genes) if g in adata.var_names]
    fates = dict(zip(gene_list, perturbed_fates(adata, cluster_key, scaf,
                                                [[g] for g in gene_list], workers)))
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for A, B, An, Bn in lineage_pairs:
        a_cols = [scaf["sidx"][str(c)] for c in A if str(c) in scaf["sidx"]]
        b_cols = [scaf["sidx"][str(c)] for c in B if str(c) in scaf["sidx"]]
        split_wt = split_fraction(scaf["fate_wt"], a_cols, b_cols)
        out[(An, Bn)] = {
            "shift": {g: (split_fraction(fates[g], a_cols, b_cols) - split_wt) for g in gene_list},
            "split_wt": split_wt,
        }
    return out


def dose_fate_bias(adata: AnnData, cluster_key: str, lineage_pairs: Sequence[LineagePair],
                   genes: Sequence[str], fractions: Optional[Sequence[float]] = None,
                   spliced_key: str = "Ms", percentile: float = 99.0,
                   basis: Optional[str] = None, n_neighbors: int = 30, sigma: float = 0.05,
                   frac: float = 0.3,
                   transitional: Optional[Mapping[Tuple[str, str], Sequence[str]]] = None,
                   scaffold: Optional[FateScaffold] = None
                   ) -> Dict[Tuple[str, str], Dict[str, pd.DataFrame]]:
    """Fate-split shift as a function of dose, from knockout through overexpression.

    Each gene is held at a fraction of its own natural maximum, taken as the ``percentile`` th
    percentile of its observed expression, so a dose is comparable across genes on very different
    scales. Dose zero reproduces the knockout value, which makes :func:`pairwise_fate_bias` the
    dose-zero slice of this sweep.

    Parameters
    ----------
    fractions : sequence of float, optional
        Multiples of the natural maximum. Defaults to
        ``[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]``, spanning knockout to twofold overexpression.
    percentile : float, default 99.0
        Percentile of observed expression defining each gene's natural maximum. A gene whose
        percentile is zero or which is not measured falls back to a unit maximum.

    Returns
    -------
    dict
        ``{(A_name, B_name): {gene: DataFrame with columns ``level_frac`` and ``fate_bias``}}``.
    """
    scaf = scaffold or fate_scaffold(adata, cluster_key, lineage_pairs, basis=basis,
                                     n_neighbors=n_neighbors, sigma=sigma, frac=frac)
    if fractions is None:
        fractions = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    gene_list = [g for g in dict.fromkeys(genes) if g in adata.var_names]
    expr = np.asarray(adata.layers[spliced_key])
    var_names = list(adata.var_names)
    axes = lineage_pair_axes(scaf, lineage_pairs, transitional)

    rec: Dict[Tuple[str, str], Dict[str, list]] = {(An, Bn): {} for A, B, An, Bn in lineage_pairs}
    for gene in gene_list:
        natural = float(np.percentile(expr[:, var_names.index(gene)], percentile)) \
            if gene in var_names else 1.0
        natural = natural if natural > 0 else 1.0
        for fr in fractions:
            fate = perturbed_fate(adata, cluster_key, scaf, gene, fr * natural)
            for A, B, An, Bn in lineage_pairs:
                ax = axes[(An, Bn)]
                bias = float((split_fraction(fate, ax["Ac"], ax["Bc"])
                              - ax["split_wt"])[ax["focus"]].mean())
                rec[(An, Bn)].setdefault(gene, []).append((float(fr), bias))
    return {pair: {g: pd.DataFrame(rows, columns=["level_frac", "fate_bias"])
                   for g, rows in genemap.items()}
            for pair, genemap in rec.items()}


def fate_embedding_flow(adata: AnnData, cluster_key: str, lineage_pairs: Sequence[LineagePair],
                        genes: Sequence[str], basis: Optional[str] = None, n_neighbors: int = 30,
                        sigma: float = 0.05, frac: float = 0.3,
                        scaffold: Optional[FateScaffold] = None) -> Dict[str, np.ndarray]:
    """Change a knockout induces in the embedding flow, from the same kernel as the fate readout.

    The transition matrix induces an embedding velocity :math:`v = T e - e`, the expected step to
    a neighbor, since each row of ``T`` sums to one. This returns :math:`v^{\\mathrm{KO}} -
    v^{\\mathrm{WT}}`: how the knockout redirects the flow on the embedding. Built from the fate
    kernel rather than from an ODE, so it explains the per-cell fate map instead of sitting beside
    it, and a gene outside the fitted selection gives an exactly zero field.

    Returns
    -------
    dict
        ``{gene: ndarray (n_cells, 2)}``.
    """
    scaf = scaffold or fate_scaffold(adata, cluster_key, lineage_pairs, basis=basis,
                                     n_neighbors=n_neighbors, sigma=sigma, frac=frac)
    use_basis = basis if basis is not None else scaf.get("basis")
    emb = np.asarray(adata.obsm[f"X_{use_basis or 'umap'}"])[:, :2].astype(float)
    v_wt = np.asarray(scaf["T_wt"] @ emb) - emb
    out: Dict[str, np.ndarray] = {}
    for gene in [g for g in dict.fromkeys(genes) if g in adata.var_names]:
        trans, _ = _perturbed_transition(adata, cluster_key, scaf, gene, 0.0)
        if trans is None:
            out[gene] = np.zeros_like(emb)
            continue
        out[gene] = (np.asarray(trans @ emb) - emb) - v_wt
    return out


def commitment_time(adata: AnnData, cluster_key: str, lineage_pairs: Sequence[LineagePair],
                    genes: Sequence[str], basis: Optional[str] = None, n_neighbors: int = 30,
                    sigma: float = 0.05, frac: float = 0.3,
                    scaffold: Optional[FateScaffold] = None) -> Dict[str, float]:
    """Change in expected time to commitment under each knockout.

    The mean first-passage time of the absorbing chain is the expected number of transition steps
    before a cell reaches any terminal fate. Solving :math:`(I - Q) t = 1` over the transient
    cells gives it wild type and per knockout; the readout is the median change over those cells.
    Positive means the knockout delays commitment, negative that it accelerates it.

    Returns
    -------
    dict
        ``{gene: median change in chain steps}``.

    Notes
    -----
    The unit is chain steps, not real time. The kernel is a similarity, not a rate, so these
    numbers compare knockouts against each other rather than against a clock.
    """
    scaf = scaffold or fate_scaffold(adata, cluster_key, lineage_pairs, basis=basis,
                                     n_neighbors=n_neighbors, sigma=sigma, frac=frac)
    is_terminal = np.zeros(len(scaf["clusters"]), bool)
    is_terminal[np.concatenate([v for v in scaf["term"].values()])] = True
    transient = np.where(~is_terminal)[0]

    def first_passage(trans_matrix):
        Q = trans_matrix.tocsr()[transient][:, transient].tocsc()
        return spsolve(sp.eye(len(transient), format="csc") - Q, np.ones(len(transient)))

    t_wt = first_passage(scaf["T_wt"])
    out: Dict[str, float] = {}
    for gene in [g for g in dict.fromkeys(genes) if g in scaf["names"]]:
        trans, _ = _perturbed_transition(adata, cluster_key, scaf, gene, 0.0)
        out[gene] = float(np.nanmedian(first_passage(trans) - t_wt))
    return out
