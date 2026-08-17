"""Fate-probability perturbation metric for scHopfield -- a better KO-on-lineage readout than the
projected-cosine lineage bias (which is the CellOracle/RegVelo-local style and is fooled by the
high-expression/sink projection artifact, e.g. Malat1; see the high-expression-magnitude-artifact
note).

Mirrors the modern methods, which score where cells END UP rather than a local displacement's
direction:
  - RegVelo: recompute velocity under KO, feed to CellRank -> change in terminal-state fate
    (absorption) probabilities.
  - Dynamo:  propagate the perturbation through the vector field.
  - DynNet:  terminal-state / attractor reassignment in a dynamical model.

This module is the NATIVE engine (no CellRank dependency): it builds the cell-cell transition
matrix from the model velocity (CellOracle/scVelo correlation kernel), makes the terminal-state
cells absorbing, and solves the absorbing Markov chain for each cell's fate probability to each
terminal state -- which is exactly CellRank's core. The KO velocity is the fitted field evaluated
with the gene held at 0, so a pure sink (no out-edges) leaves the velocity, transition matrix, and
fate probabilities unchanged -> ~0 effect, by construction.

Terminal states = the field's STABLE cells within each known terminal cell type (the cells nearest
the fitted attractor = lowest model-velocity magnitude), per the design decision.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from sklearn.neighbors import NearestNeighbors

import os
import sys

# The reproducibility tree is flat one level up; this directory holds the compute
# helpers. Both are anchored to this file rather than to the working directory, so a
# script runs the same from anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import paths                                                     # noqa: E402
from scHopfield.tools.flow import get_genes_used            # noqa: E402
from scHopfield.dynamics.solver import create_solver        # noqa: E402


def model_velocity(adata, ck, genes_used, ko_gene=None, ko_level=0.0, spliced_key="Ms"):
    """Model velocity dx/dt = W_c sigma(x) - gamma x + I at each cell's OBSERVED state, under its
    own cluster-specific field. If ``ko_gene`` is given, that gene is HELD at ``ko_level`` in the
    input (0 = knockout, matching simulate_shift_ode's fixed-gene KO; >0 = knockdown/overexpression
    for a dose sweep), so its regulatory drive is set by the clamp. Returns (X, V, names), all in
    genes_used space."""
    X = np.asarray(adata.layers[spliced_key])[:, genes_used].astype(float)
    names = np.asarray(adata.var_names.values)[genes_used]
    V = np.zeros_like(X)
    clusters = adata.obs[ck].astype(str).values
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


def transition_matrix(X, V, knn_idx, sigma=0.05):
    """CellOracle/scVelo correlation kernel: T_ij proportional to exp(pearson(V_i, X_j - X_i)/sigma)
    over the KNN neighbors j of cell i, row-normalized. X, V in the same (gene) space; knn_idx is
    (n, k) neighbor indices (self excluded)."""
    n = X.shape[0]
    rows, cols, vals = [], [], []
    vnorm = np.linalg.norm(V - V.mean(1, keepdims=True), axis=1) + 1e-12
    Vc = V - V.mean(1, keepdims=True)
    for i in range(n):
        js = knn_idx[i]
        dX = X[js] - X[i]                                   # (k, g)
        dXc = dX - dX.mean(1, keepdims=True)
        num = dXc @ Vc[i]
        den = (np.linalg.norm(dXc, axis=1) + 1e-12) * vnorm[i]
        corr = num / den
        w = np.exp(corr / sigma)
        w /= w.sum() + 1e-12
        rows.extend([i] * len(js)); cols.extend(js.tolist()); vals.extend(w.tolist())
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def terminal_states(clusters, terminal_clusters, vmag, frac=0.3, min_cells=5):
    """Field stable states for each known terminal cell type: within each terminal cluster, the
    fraction of cells with the smallest model-velocity magnitude (nearest the fitted attractor)."""
    out = {}
    for c in terminal_clusters:
        idx = np.where(clusters == str(c))[0]
        if len(idx) == 0:
            continue
        k = max(min_cells, int(round(len(idx) * frac)))
        out[str(c)] = idx[np.argsort(vmag[idx])[:k]]
    return out


def fate_probabilities(T, term_sets):
    """Absorbing-Markov-chain fate probabilities. term_sets: {state: array of absorbing cell idx}.
    Returns (fate [n x n_states], state_names). Robust to transient sinks via a tiny diagonal
    damping toward the terminal set if (I - Q) is near-singular."""
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
    A = (sp.eye(len(trans), format="csc") - Q)
    B = spsolve(A, R)
    B = np.atleast_2d(B)
    if B.shape[0] != len(trans):
        B = B.T
    fate = np.zeros((n, len(states)))
    fate[trans] = np.clip(B, 0, None)
    rs = fate[trans].sum(1, keepdims=True)                  # renormalize (guards against leak)
    fate[trans] = fate[trans] / (rs + 1e-12)
    for k, s in enumerate(states):
        fate[term_sets[s], k] = 1.0
    return fate, states


def ko_fate_shift(adata, ck, terminal_clusters, ko_genes, basis=None,
                  n_neighbors=30, sigma=0.05, frac=0.3):
    """Full pipeline: WT + per-KO terminal-state fate probabilities and the per-lineage change.
    Returns a dict with WT fate, per-KO fate, the mean fate shift per terminal state, and a
    Wilcoxon p-value per state (KO vs WT fate probability across cells)."""
    from scipy.stats import wilcoxon
    g = get_genes_used(adata)
    clusters = adata.obs[ck].astype(str).values
    if basis is None:
        basis = "umap"
    emb = np.asarray(adata.obsm[f"X_{basis}"])[:, :2]
    knn_idx = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(emb).kneighbors(
        return_distance=False)[:, 1:]

    X, V_wt, names = model_velocity(adata, ck, g)
    vmag = np.linalg.norm(V_wt, axis=1)
    term = terminal_states(clusters, terminal_clusters, vmag, frac=frac)
    T_wt = transition_matrix(X, V_wt, knn_idx, sigma)
    fate_wt, states = fate_probabilities(T_wt, term)      # all-gene WT baseline (for reporting)

    out = {"states": states, "fate_wt": fate_wt, "terminal_sets": term,
           "clusters": clusters, "ko": {}}
    for gene in ko_genes:
        # The transition kernel must see only the DOWNSTREAM (propagation) response, so drop the KO
        # gene's own coordinate from both the WT and KO velocity fields and compare gene-matched.
        # A pure sink then leaves the off-gene velocity untouched -> identical T -> zero fate shift
        # (same principle as excluding the KO gene from the cascade norm).
        gi = list(names).index(gene) if gene in names else None
        keep = np.ones(len(names), bool)
        if gi is not None:
            keep[gi] = False
        _, V_ko, _ = model_velocity(adata, ck, g, ko_gene=gene)
        T_wt_g = transition_matrix(X[:, keep], V_wt[:, keep], knn_idx, sigma)
        T_ko_g = transition_matrix(X[:, keep], V_ko[:, keep], knn_idx, sigma)
        fate_wt_g, _ = fate_probabilities(T_wt_g, term)
        fate_ko_g, _ = fate_probabilities(T_ko_g, term)
        d = fate_ko_g - fate_wt_g
        shift = d.mean(0)
        pvals = []
        for k in range(len(states)):
            diff = d[:, k]
            nz = np.abs(diff) > 1e-12
            pvals.append(float(wilcoxon(diff[nz]).pvalue) if nz.sum() > 10 else np.nan)
        out["ko"][gene] = {"fate": fate_ko_g, "shift": shift, "pvals": np.array(pvals),
                           "max_abs_shift": float(np.abs(shift).max())}
    return out


def pairwise_fate_bias(adata, ck, lps, cands, basis=None, n_neighbors=30, sigma=0.05, frac=0.3,
                       transitional=None):
    """Per-lineage-pair fate-probability bias, the fate-based replacement for the projected-cosine
    single-KO lineage bias (panel e). Terminal states = the field's stable cells (low model-velocity)
    within EVERY cluster that appears as a lineage-pair arm, so both fates of every decision (incl. a
    progenitor arm) are absorbing macrostates. For each candidate KO and each pair (A vs B):
      bias(gene) = mean_{decider cells} [ split_KO - split_WT ],  split = fate_A / (fate_A + fate_B)
    The mean is taken over the TRANSITIONAL 'decider' cells (see ``_focus_mask``), NOT all cells, so
    the committed cells (fate already fixed, shift ~0) do not dilute the summary. ``transitional`` is
    an optional {(An,Bn): [clusters]} map naming each decision's transitional population; without it
    the labile-split band is used. Positive = the KO shifts fate toward lineage A. The KO gene's OWN
    velocity coordinate is neutralized (set to its WT value) before building the KO transition matrix,
    so only the downstream (propagation) response moves fate -> a pure sink gives exactly 0
    (Malat1-robust). Returns {(An,Bn): {'bias': Series over cands (sorted), 'pvals': Series
    (Wilcoxon over the decider cells)}}."""
    from scipy.stats import wilcoxon
    g = get_genes_used(adata)
    clusters = adata.obs[ck].astype(str).values
    basis = basis or "umap"
    emb = np.asarray(adata.obsm[f"X_{basis}"])[:, :2]
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(emb).kneighbors(return_distance=False)[:, 1:]
    X, V_wt, names = model_velocity(adata, ck, g)
    vmag = np.linalg.norm(V_wt, axis=1)
    arm_clusters = sorted({str(c) for A, B, An, Bn in lps for c in list(A) + list(B)})
    term = terminal_states(clusters, arm_clusters, vmag, frac=frac)
    T_wt = transition_matrix(X, V_wt, knn, sigma)
    fate_wt, states = fate_probabilities(T_wt, term)
    sidx = {s: k for k, s in enumerate(states)}
    name_list = list(names)
    cands = [c for c in dict.fromkeys(cands) if c in adata.var_names]
    fate_ko_all = {}
    for gene in cands:
        gi = name_list.index(gene) if gene in name_list else None
        Vk = V_wt.copy()
        if gi is not None:
            _, Vko, _ = model_velocity(adata, ck, g, ko_gene=gene)
            Vk = Vko.copy(); Vk[:, gi] = V_wt[:, gi]           # neutralize the KO gene's own coordinate
        Tk = transition_matrix(X, Vk, knn, sigma)
        fate_ko_all[gene], _ = fate_probabilities(Tk, term)    # (n_cells, n_states)
    out = {}
    for A, B, An, Bn in lps:
        Ac = [sidx[str(c)] for c in A if str(c) in sidx]; Bc = [sidx[str(c)] for c in B if str(c) in sidx]
        # SPLIT fraction fate_A / (fate_A + fate_B): the shift in the A-vs-B decision, normalized so a
        # globally dominant terminal fate (e.g. beta) does not swamp every KO. A pure sink -> 0.
        split_wt = _split(fate_wt, Ac, Bc)
        trans = transitional.get((An, Bn)) if transitional else None
        focus = _focus_mask(clusters, split_wt, trans)         # aggregate over the decider cells only
        bias, pv = {}, {}
        for gene in cands:
            diff = _split(fate_ko_all[gene], Ac, Bc) - split_wt
            d = diff[focus]
            bias[gene] = float(d.mean())
            nz = np.abs(d) > 1e-12
            pv[gene] = float(wilcoxon(d[nz]).pvalue) if nz.sum() > 10 else np.nan
        out[(An, Bn)] = {"bias": pd.Series(bias).sort_values(), "pvals": pd.Series(pv)}
    return out


def _fate_scaffold(adata, ck, lps, basis=None, n_neighbors=30, sigma=0.05, frac=0.3):
    """Shared WT fate scaffold reused by the per-cell map (panel d) and the dose sweep (panel f):
    the KNN graph, the WT model velocity, the terminal (absorbing) sets over every lineage-pair arm,
    and the WT fate probabilities. Built once so the perturbed evaluations only re-solve the chain."""
    g = get_genes_used(adata)
    clusters = adata.obs[ck].astype(str).values
    basis = basis or "umap"
    emb = np.asarray(adata.obsm[f"X_{basis}"])[:, :2]
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(emb).kneighbors(return_distance=False)[:, 1:]
    X, V_wt, names = model_velocity(adata, ck, g)
    vmag = np.linalg.norm(V_wt, axis=1)
    arm_clusters = sorted({str(c) for A, B, An, Bn in lps for c in list(A) + list(B)})
    term = terminal_states(clusters, arm_clusters, vmag, frac=frac)
    T_wt = transition_matrix(X, V_wt, knn, sigma)
    fate_wt, states = fate_probabilities(T_wt, term)
    return dict(g=g, X=X, V_wt=V_wt, names=list(names), knn=knn, term=term, clusters=clusters,
                vmag=vmag, fate_wt=fate_wt, states=states, T_wt=T_wt,
                sidx={s: k for k, s in enumerate(states)}, sigma=sigma)


def ko_embedding_flow(adata, ck, lps, genes, basis=None, n_neighbors=30, sigma=0.05, frac=0.3):
    """The KO-induced CHANGE in the embedding flow, per gene: the expected per-cell displacement under
    the KO transition kernel minus the WT one. The transition-matrix embedding velocity is
    v = T @ emb - emb (each row of T sums to 1, so this is the expected step to a neighbor); the
    returned field is v_KO - v_WT, i.e. how the knockout redirects the velocity on the UMAP. Built
    from the SAME kernels as the fate metric (KO gene's own coordinate neutralized), so it explains
    the panel-d fate map: arrows show the redirected flow, color shows the resulting fate change.
    Returns {gene: ndarray(n_cells, 2)}."""
    sc = _fate_scaffold(adata, ck, lps, basis=basis, n_neighbors=n_neighbors, sigma=sigma, frac=frac)
    emb = np.asarray(adata.obsm[f"X_{basis or 'umap'}"])[:, :2].astype(float)
    v_wt = np.asarray(sc["T_wt"] @ emb) - emb
    names = sc["names"]
    genes = [g for g in dict.fromkeys(genes) if g in adata.var_names]
    out = {}
    for g in genes:
        gi = names.index(g) if g in names else None
        if gi is None:
            out[g] = np.zeros_like(emb)
            continue
        _, Vp, _ = model_velocity(adata, ck, sc["g"], ko_gene=g, ko_level=0.0)
        Vk = Vp.copy()
        Vk[:, gi] = sc["V_wt"][:, gi]                          # neutralize own coordinate (fate-consistent)
        Tk = transition_matrix(sc["X"], Vk, sc["knn"], sc["sigma"])
        out[g] = (np.asarray(Tk @ emb) - emb) - v_wt
    return out


def _focus_mask(clusters, split_wt, transitional=None):
    """Cells to aggregate the fate shift over for the scalar summaries (panels e/f): the transitional
    'decider' population, so committed cells (fate already fixed) do not dilute the mean. If explicit
    transitional clusters are given for the pair, use them; else the cells whose WT fate is labile
    (split 0.1-0.9); else (a sharp, near-binary split with no labile band) fall back to all cells so
    the summary is still defined."""
    if transitional:
        m = np.isin(clusters, [str(c) for c in transitional])
        if m.sum() >= 10:
            return m
    m = (split_wt > 0.1) & (split_wt < 0.9)
    if m.sum() >= 10:
        return m
    return np.ones(len(split_wt), bool)


def _perturbed_fate(adata, ck, sc, gene, level):
    """Per-cell fate probabilities with ``gene`` held at ``level`` in the input and its OWN velocity
    coordinate neutralized (set to WT), so only the downstream (propagation) response moves fate --
    the same projection-free construction as pairwise_fate_bias. ``level``=0 reproduces the KO. A
    pure sink (no out-edges) leaves the off-gene velocity untouched -> WT fate (zero shift)."""
    names = sc["names"]
    gi = names.index(gene) if gene in names else None
    if gi is None:
        return sc["fate_wt"]
    _, Vp, _ = model_velocity(adata, ck, sc["g"], ko_gene=gene, ko_level=level)
    Vk = Vp.copy()
    Vk[:, gi] = sc["V_wt"][:, gi]
    Tk = transition_matrix(sc["X"], Vk, sc["knn"], sc["sigma"])
    fate, _ = fate_probabilities(Tk, sc["term"])
    return fate


def _split(fate, Ac, Bc):
    """A-vs-B split fraction fate_A / (fate_A + fate_B) per cell."""
    sA = fate[:, Ac].sum(1)
    sB = fate[:, Bc].sum(1)
    return sA / (sA + sB + 1e-12)


def per_cell_fate_shift(adata, ck, lps, genes, basis=None, n_neighbors=30, sigma=0.05, frac=0.3):
    """Panel d: the per-cell change in the A-vs-B fate split fraction after KO of each gene, mapped
    over the embedding (the spatial, projection-free version of the panel-e decision). Positive =
    the KO redirects that cell toward lineage A; a stable-committed cell is ~0, while a cell whose
    already-committed fate the KO erodes (reprogramming) shows a nonzero shift. A pure sink gives ~0
    everywhere. Also returns the WT split per pair so the plot can gray the unchanged cells and the
    reports can separate deciders from committed cells. Returns
    {(An,Bn): {'shift': {gene: array(n_cells)}, 'split_wt': array(n_cells)}}."""
    sc = _fate_scaffold(adata, ck, lps, basis=basis, n_neighbors=n_neighbors, sigma=sigma, frac=frac)
    genes = [g for g in dict.fromkeys(genes) if g in adata.var_names]
    fate_ko = {g: _perturbed_fate(adata, ck, sc, g, 0.0) for g in genes}
    out = {}
    for A, B, An, Bn in lps:
        Ac = [sc["sidx"][str(c)] for c in A if str(c) in sc["sidx"]]
        Bc = [sc["sidx"][str(c)] for c in B if str(c) in sc["sidx"]]
        split_wt = _split(sc["fate_wt"], Ac, Bc)
        out[(An, Bn)] = {"shift": {g: (_split(fate_ko[g], Ac, Bc) - split_wt) for g in genes},
                         "split_wt": split_wt}
    return out


def dose_fate_bias(adata, ck, lps, genes, fractions=None, spliced_key="Ms", percentile=99.0,
                   basis=None, n_neighbors=30, sigma=0.05, frac=0.3, transitional=None):
    """Panel f: the A-vs-B fate split-fraction shift as a function of dose, the fate-based
    replacement for the projected-cosine dose-response. For each gene and dose fraction (of the
    gene's natural maximum), hold the gene at that level, neutralize its own coordinate, and take
    the mean shift in the A-vs-B split fraction vs WT over the TRANSITIONAL decider cells (same
    ``_focus_mask`` / ``transitional`` map as panel e, so committed cells do not dilute it). dose=0
    reproduces the panel-e KO value, so panel e is the dose-0 slice of panel f. Returns
    {(An,Bn): {gene: DataFrame[level_frac, fate_bias]}}."""
    sc = _fate_scaffold(adata, ck, lps, basis=basis, n_neighbors=n_neighbors, sigma=sigma, frac=frac)
    if fractions is None:
        fractions = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    genes = [g for g in dict.fromkeys(genes) if g in adata.var_names]
    Ms = np.asarray(adata.layers[spliced_key])
    vn = list(adata.var_names)
    splits_wt = {}
    for A, B, An, Bn in lps:
        Ac = [sc["sidx"][str(c)] for c in A if str(c) in sc["sidx"]]
        Bc = [sc["sidx"][str(c)] for c in B if str(c) in sc["sidx"]]
        split_wt = _split(sc["fate_wt"], Ac, Bc)
        trans = transitional.get((An, Bn)) if transitional else None
        focus = _focus_mask(sc["clusters"], split_wt, trans)
        splits_wt[(An, Bn)] = (split_wt, Ac, Bc, focus)
    rec = {(An, Bn): {} for A, B, An, Bn in lps}
    for gene in genes:
        nat = float(np.percentile(Ms[:, vn.index(gene)], percentile)) if gene in vn else 1.0
        nat = nat if nat > 0 else 1.0
        for fr in fractions:
            fate = _perturbed_fate(adata, ck, sc, gene, fr * nat)
            for A, B, An, Bn in lps:
                split_wt, Ac, Bc, focus = splits_wt[(An, Bn)]
                bias = float((_split(fate, Ac, Bc) - split_wt)[focus].mean())
                rec[(An, Bn)].setdefault(gene, []).append((float(fr), bias))
    out = {}
    for pair, genemap in rec.items():
        out[pair] = {g: pd.DataFrame(rows, columns=["level_frac", "fate_bias"])
                     for g, rows in genemap.items()}
    return out
