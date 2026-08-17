"""Compute + cache the perturbation-dynamics analyses for the composite figure
(make_perturbation_dynamics.py). All the ODE integrations live here so the figure can be
iterated cheaply. Reuses the exact scHopfield calls the per-dataset report uses.

  a  embedding + WT velocity streamlines            (wt reference flow)
  b  KO Delta_x (ODE) + inner product Delta_x . v_ref, for 4 developmental TFs
  c  single-KO lineage bias, for each configured lineage pair
  d  dose-response of the lineage bias, for the 4 TFs
  e  short-time cascade: per-cluster mean |Delta_x| vs ODE time, WT + 4 KOs

Run:  python reproducibility/compute/_perturb_dynamics_compute.py --dataset pancreas
"""
from __future__ import annotations
import argparse, os, sys, pickle
import numpy as np
import pandas as pd
import anndata as ad

# The reproducibility tree is flat one level up; this directory holds the compute
# helpers. Both are anchored to this file rather than to the working directory, so a
# script runs the same from anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import paths                                                     # noqa: E402
import scHopfield as sch                                        # noqa: E402
from sections import basis_of, get_colors, present_clusters    # noqa: E402
from scHopfield.tools.embedding import build_correlation_projector  # noqa: E402
from scHopfield.tools.flow import get_genes_used               # noqa: E402
from scHopfield.dynamics.solver import create_solver           # noqa: E402

VARIANT_SUF = ""   # set in main() from --variant; suffixes the driver_scores CSVs read for selection

# developmental TFs of pancreatic endocrinogenesis to perturb (panels b, d, e)
# known pancreatic developmental regulators: 3 progenitor->endocrine + 3 alpha/beta
# (Arx = alpha spec, Pax4 = beta spec, Nkx2-2 = alpha/beta decision).
TFS_BY_DATASET = {
    "pancreas": ["Neurog3", "Neurod1", "Pax6", "Arx", "Pax4", "Nkx2-2"],
    "paul15": ["Gata1", "Klf1", "Gata2", "Spi1", "Cebpa", "Gfi1"],
    "paul15_coarse": ["Gata1", "Klf1", "Gata2", "Spi1", "Cebpa", "Gfi1"],
    "dynamo_hematopoiesis": ["GATA1", "KLF1", "GATA2", "SPI1", "CEBPA", "GFI1"],   # dynamo = uppercase
    "schwann": ["Sox8", "Egr2", "Tfap2a", "Phox2b", "Ascl1", "Gata3"],
    # murine_nc / human_limb: 2 data-driven lineage pairs each, so 2 gene groups (order = group0
    # genes then group1 genes so the panel-b group headers line up over the columns).
    # neural crest: glia/melanocyte SoxE + melanocyte + general-crest, then neuronal/sensory.
    "murine_nc": ["Sox10", "Tfap2a", "Mitf", "Isl1", "Gata3", "Pou4f1"],
    # myogenesis: progenitor/specification TFs, then the myogenic regulatory factors.
    "human_limb": ["PAX3", "PAX7", "PITX2", "MYOD1", "MYOG", "MEF2C"],
}
TF_GROUPS = {
    "pancreas": {"progenitor / differentiated": ["Neurog3", "Neurod1", "Pax6"],
                 "alpha / beta": ["Arx", "Pax4", "Nkx2-2"]},
    "paul15": {"erythroid / myeloid": ["Gata1", "Klf1", "Gata2", "Spi1", "Cebpa", "Gfi1"]},
    "paul15_coarse": {"erythroid / myeloid": ["Gata1", "Klf1", "Gata2", "Spi1", "Cebpa", "Gfi1"]},
    "dynamo_hematopoiesis": {"erythroid / myeloid": ["GATA1", "KLF1", "GATA2", "SPI1", "CEBPA", "GFI1"]},
    "schwann": {"glia / neuron": ["Sox8", "Egr2", "Tfap2a", "Phox2b", "Ascl1", "Gata3"]},
    # data-driven pairs (see the module docstring): NC pair0 = enteric-neuron vs olfactory-
    # ensheathing GLIA, pair1 = PNS-neuron vs DRG (sensory); limb pair0/pair1 are both
    # progenitor vs differentiated myocyte. Group k is tied to lineage pair k (panels c/d).
    "murine_nc": {"glia / melanocyte": ["Sox10", "Tfap2a", "Mitf"],
                  "neuronal / sensory": ["Isl1", "Gata3", "Pou4f1"]},
    "human_limb": {"progenitor / specification": ["PAX3", "PAX7", "PITX2"],
                   "myogenic differentiation": ["MYOD1", "MYOG", "MEF2C"]},
}
FRACTIONS = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

# Transitional 'decider' cells per lineage decision, for the panel-e/f fate-shift SUMMARY only (the
# panel-d map always shows every cell). Aggregating the scalar over these cells, instead of all cells,
# stops the committed cells (whose fate is already fixed, shift ~0) from diluting the mean. Keyed by
# (A_name, B_name); datasets left out fall back to the labile-split band (see _focus_mask).
TRANSITIONAL_BY_DATASET = {
    "pancreas": {("alpha", "beta"): ["Pre-endocrine"],
                 ("differentiated", "progenitor"): ["Ngn3 high EP", "Pre-endocrine"]},
    # paul15 has no bipotent CMP cluster (every cluster is already committed, eryth_frac ~0 or 1); the
    # branch-point progenitors of the two arms (MEP + GMP) are the closest deciders, so aggregate the
    # fate shift over them instead of the labile-split band (which was erythroid-biased). See the
    # perturbation Methods note on decider-cell selection.
    "paul15": {("erythroid", "myeloid"): ["7MEP", "9GMP", "10GMP"]},
}
CASCADE_TMAX = 20.0                                          # short-time cascade horizon (ODE time)
CASCADE_NSTEP = 20                                           # -> dt = 1.0 per advanced step


def compute_cascade(a, ck, tfs, dev, tmax=CASCADE_TMAX, nstep=CASCADE_NSTEP):
    """Short-time cascade relative to WT: integrate WT and each KO forward in nstep segments,
    each starting from the previous segment's final state (state advanced via simulated_count).
    Metric per cluster = mean |x_KO(t) - x_WT(t)| over model genes (the KO gene excluded); WT and
    every KO share x(0), so each curve starts at 0 (t=0). WT-vs-WT is 0 -> only KOs returned."""
    genes_used = get_genes_used(a)
    model_names = np.asarray(a.var_names.values)[genes_used]
    clusters = a.obs[ck].astype(str).values
    cl_names = list(pd.unique(clusters))
    dt = tmax / nstep

    print(f"[e] cascade WT reference (tmax={tmax})", flush=True)   # WT trajectory x_WT at each step
    cur_wt = a.copy(); wt_states = []
    for step in range(1, nstep + 1):
        cur_wt = sch.dyn.simulate_shift_ode(cur_wt, {}, cluster_key=ck, dt=float(dt), n_steps=50,
                                            use_cluster_specific_GRN=True, device=dev)
        cur_wt.layers["Ms"] = np.asarray(cur_wt.layers["simulated_count"])
        wt_states.append(np.asarray(cur_wt.layers["Ms"])[:, genes_used].copy())

    rows = []
    for tf in tfs:
        keep = np.array([g != tf for g in model_names])
        print(f"[e] cascade vs WT: {tf} KO", flush=True)
        cur = a.copy()
        for cl in cl_names:                                   # t=0: KO == WT == x(0)
            rows.append(dict(perturbation=f"{tf} KO", cluster=str(cl), t=0.0, mean_abs_delta=0.0))
        for step in range(1, nstep + 1):
            cur = sch.dyn.simulate_shift_ode(cur, {tf: 0.0}, cluster_key=ck, dt=float(dt), n_steps=50,
                                             use_cluster_specific_GRN=True, device=dev)
            cur.layers["Ms"] = np.asarray(cur.layers["simulated_count"])
            dev_cell = np.abs(np.asarray(cur.layers["Ms"])[:, genes_used]
                              - wt_states[step - 1])[:, keep].mean(1)
            means = pd.DataFrame({"cluster": clusters, "val": dev_cell}).groupby(
                "cluster", observed=True)["val"].mean()
            for cl, m in means.items():
                rows.append(dict(perturbation=f"{tf} KO", cluster=str(cl), t=step * dt,
                                 mean_abs_delta=float(m)))
    return pd.DataFrame(rows)


def compute_jacobian_predictions(a, ck, tfs, lps, groups, spliced_key="Ms", eps=1e-2):
    """Both first-order Jacobian predictions in ONE finite-difference pass (the columns are shared):
      - response    {gene: Series over targets}      = mean r_i = -J_ig x_g over all cells (figure
                    barplots, regulator-filtered on display);
      - response_ct {gene: DataFrame targets x celltype} = the same response resolved PER CELL TYPE
                    (the report heatmap, per gene and per cell type);
      - push        {gene: (An, Bn, per-cell push)}  = r projected on the gene's fate axis d =
                    normalize(centroid_A - centroid_B) (figure UMAPs, the first-order 'commitment push').
    J column g is a central difference of the fitted solver dynamics (exact for the fitted field)."""
    from perturbation_measures import out_strength
    g_used = get_genes_used(a)
    names = list(np.asarray(a.var_names)[g_used])
    X = np.asarray(a.layers[spliced_key])[:, g_used].astype(float)
    clusters = a.obs[ck].astype(str).values
    tfs = [g for g in dict.fromkeys(tfs) if g in names]
    gnames = list(groups); gene_pair = {}; axis = {}
    for k, (A, B, An, Bn) in enumerate(lps):
        if k >= len(gnames):
            break
        Am = np.isin(clusters, [str(c) for c in A]); Bm = np.isin(clusters, [str(c) for c in B])
        if Am.sum() == 0 or Bm.sum() == 0:
            continue
        d = X[Am].mean(0) - X[Bm].mean(0)
        axis[k] = (d / (np.linalg.norm(d) + 1e-12), An, Bn)
        for gene in groups.get(gnames[k], []):
            gene_pair[gene] = k
    resp_sum = {g: np.zeros(len(names)) for g in tfs}
    resp_ct = {g: {} for g in tfs}
    push = {g: np.zeros(len(clusters)) for g in tfs}
    ntot = 0
    for c in pd.unique(clusters):
        sel = np.where(clusters == str(c))[0]
        try:
            solver = create_solver(a, str(c), spliced_key=spliced_key)
        except Exception:
            continue
        Xc = X[sel]
        for g in tfs:
            gi = names.index(g)
            Xp = Xc.copy(); Xp[:, gi] += eps
            Xm = Xc.copy(); Xm[:, gi] -= eps
            jcol = (solver.dynamics_batch(Xp, 0.0) - solver.dynamics_batch(Xm, 0.0)) / (2 * eps)
            r = -jcol * Xc[:, gi][:, None]                     # (n_cells_c, n_targets)
            resp_sum[g] += r.sum(0)
            resp_ct[g][str(c)] = r.mean(0)
            k = gene_pair.get(g)
            if k in axis:
                push[g][sel] = r @ axis[k][0]
        ntot += len(sel)
    response = {g: pd.Series(resp_sum[g] / max(ntot, 1), index=names).drop(labels=[g], errors="ignore")
                for g in tfs}
    response_ct = {g: pd.DataFrame(resp_ct[g], index=names) for g in tfs}   # targets x cell type
    push_out = {g: (axis[gene_pair[g]][1], axis[gene_pair[g]][2], push[g])
                for g in tfs if gene_pair.get(g) in axis}
    return response, response_ct, push_out, out_strength(a, ck)


def wt_reference_flow(adata, ck, basis, cfg):
    """WT developmental reference flow v_ref in the embedding (report idiom): the dataset's
    own precomputed embedded velocity if configured, else velocity_S projected with the same
    correlation (transition-probability) scheme used for the perturbation flows."""
    wt_key = f"original_velocity_flow_{basis}"
    vek = cfg.get("velocity_embedding_key")
    if vek and vek in adata.obsm:
        adata.obsm[wt_key] = np.asarray(adata.obsm[vek])[:, :2].astype(float)
    else:
        g = get_genes_used(adata)
        vin = np.asarray(adata.layers["velocity_S"])[:, g]
        proj = build_correlation_projector(adata, basis=basis)
        adata.obsm[wt_key] = np.asarray(proj(vin))[:, :2].astype(float)
    return wt_key


def _wing_pool(a, ds, ck, lps, known, q=95, pool_per_wing=6):
    """Per lineage pair, the POOL of lineage-specific candidate REGULATORS to probe for the discovery
    figure: genes past the qth-percentile driver-score threshold on ONE lineage axis but BELOW it on
    the OTHER (the specificity 'wings', not the high-on-both Pareto corner of generalists), regulators
    only (out-strength > 0 in the fitted GRN, since a pure sink cannot propagate a KO), excluding the
    known regulators, deduped across pairs, up to ``pool_per_wing`` per wing ranked by specificity.
    W is W[target, regulator], so out-strength is the column sum of |W|. Returns
    (poolmap {pair_index: {'A': [genes], 'B': [genes]}}, flat unique candidate list)."""
    genes = np.asarray(a.var_names.values)
    outs = np.zeros(a.n_vars)
    for c in present_clusters(a, ck):
        key = f"W_{c}"
        if key in a.varp:
            outs += np.abs(np.asarray(a.varp[key])).sum(0)            # out-strength = |W| column sum
    outs = pd.Series(outs, index=genes)
    used = set(known); poolmap = {}; flat = []
    for kk, (A, B, An, Bn) in enumerate(lps):
        df = pd.read_csv(f"{paths.REPORTS}/{ds}/data/driver_scores_{kk + 1}{VARIANT_SUF}.csv", index_col=0)
        thrA = np.percentile(df.score_A, q); thrB = np.percentile(df.score_B, q)
        reg = outs.reindex(df.index).fillna(0).values > 0             # regulators only (KO propagates)
        elig = df[reg & ~df.index.isin(used)].copy()                  # not known, not already picked
        wingA = elig[(elig.score_A > thrA) & (elig.score_B <= thrB)].copy()   # A-specific (above A only)
        wingB = elig[(elig.score_B > thrB) & (elig.score_A <= thrA)].copy()   # B-specific (above B only)
        wingA["spec"] = wingA.score_A - wingA.score_B                 # rank each wing by its specificity
        wingB["spec"] = wingB.score_B - wingB.score_A
        aa = list(wingA.sort_values("spec", ascending=False).index[:pool_per_wing])
        bb = list(wingB.sort_values("spec", ascending=False).index[:pool_per_wing])
        poolmap[kk] = {"A": aa, "B": bb}
        for g in aa + bb:
            if g not in used:
                flat.append(g); used.add(g)
    return poolmap, flat


def _pick_by_effect(lps, poolmap, fb, per_pair=3, alpha=0.05):
    """Choose, per lineage pair, the ``per_pair`` probed candidates with the strongest fate-probability
    effect: the largest |decider-mean shift| among the SIGNIFICANT ones (Wilcoxon p < alpha), falling
    back to the largest |effect| overall if too few are significant. This selects genes by their
    actual perturbation result, not by structural driver score, and does not force directional balance
    (an axis with strong drivers of only one lineage should show them). Returns (tfs_ordered, groups)."""
    tfs = []; groups = {}; used = set()
    for kk, (A, B, An, Bn) in enumerate(lps):
        pool = poolmap.get(kk, {}).get("A", []) + poolmap.get(kk, {}).get("B", [])
        bias = fb.get((An, Bn), {}).get("bias", pd.Series(dtype=float))
        pv = fb.get((An, Bn), {}).get("pvals", pd.Series(dtype=float))
        cand = [g for g in pool if g in bias.index and g not in used]
        sig = [g for g in cand if float(pv.get(g, 1.0)) < alpha]
        ranked = sorted(sig or cand, key=lambda g: abs(float(bias.get(g, 0.0))), reverse=True)
        picks = ranked[:per_pair]
        groups[f"{An} vs {Bn}"] = picks; used.update(picks); tfs += picks
    return tfs, groups


def probe_select_discovery(a, ds, ck, lps, known, basis, transitional,
                           q=95, pool_per_wing=6, per_pair=3):
    """Discovery-gene selection by ACTUAL perturbation effect (not structural driver score alone):
    build the wing candidate pool, PROBE every candidate with the fate-probability metric, then keep
    the ``per_pair`` genes per pair with the strongest significant fate effect (balanced by fate
    direction). Deterministic, so ``--only`` re-runs reproduce the cached selection."""
    from _fate_probability import pairwise_fate_bias
    poolmap, flat = _wing_pool(a, ds, ck, lps, known, q=q, pool_per_wing=pool_per_wing)
    if not flat:
        return [], {}
    print(f"[discovery] probing {len(flat)} wing candidates with the fate metric", flush=True)
    fb = pairwise_fate_bias(a, ck, lps, flat, basis=basis, transitional=transitional)
    return _pick_by_effect(lps, poolmap, fb, per_pair=per_pair)


def compute_ko_flow(a, ck, tfs, basis, dev):
    """KO displacement flow (delta x) per gene for the panel-d flow row, computed exactly as the
    report's perturbation-flow panels (section 5.1.3): integrate the perturbed ODE with the gene held
    at zero and project the resulting absolute displacement delta_X = x_KO - x_0 onto the embedding
    with the correlation (transition-probability) scheme. The absolute displacement is a smooth,
    coherent field (it retains the developmental component and grid-averages cleanly); the
    knockout-specific residual dX_KO - dX_WT is deliberately NOT used here, as subtracting the shared
    baseline leaves a small, noisy field that renders poorly. Gene-specific quantification is carried
    by the fate map / bias / dose panels; this row is the mechanistic perturbed-flow visualization.
    Returns ({gene: absolute Delta_x flow (n,2)}, wt_ode_flow (n,2)): the wild-type displacement flow
    is returned too so the figure can color cells by the KO-SPECIFIC alignment
    (Delta_x_KO - Delta_x_WT) . v_ref while keeping the smooth absolute arrows."""
    fk = f"perturbation_flow_{basis}"
    wt = sch.dyn.simulate_shift_ode(a, {}, cluster_key=ck, n_steps=100, method="euler", device=dev)
    sch.tl.calculate_flow(wt, source="delta", basis=basis, method="correlation",
                          cluster_key=ck, store_key=fk, verbose=False)
    wt_ode = np.asarray(wt.obsm[fk])[:, :2].copy()
    out = {}
    for tf in tfs:
        pert = sch.dyn.simulate_shift_ode(a, {tf: 0.0}, cluster_key=ck, n_steps=100, method="euler",
                                          device=dev)
        sch.tl.calculate_flow(pert, source="delta", basis=basis, method="correlation",
                              cluster_key=ck, store_key=fk, verbose=False)
        out[tf] = np.asarray(pert.obsm[fk])[:, :2].copy()
    return out, wt_ode


def wt_flow_field(a, ck, basis, cfg):
    """Projected wild-type reference velocity (the input RNA velocity, correlation-projected to the
    embedding), stored so the figure can color the KO displacement flow by its inner product (cosine
    alignment) with development."""
    return np.asarray(a.obsm[wt_reference_flow(a, ck, basis, cfg)])[:, :2].astype(float).copy()


def fate_bias_candidates(a, ck, lps, tfs):
    """Candidate gene set for the panel-e fate-bias screen: the curated TFs plus, per lineage pair,
    the top structural drivers toward each arm (so the bar chart is contextualized against the
    strongest data-driven candidates). Deduped, restricted to measured genes."""
    allc = list(tfs)
    for A, B, An, Bn in lps:
        tfsc = sch.tl.score_driver_tfs(a, A, B, cluster_key=ck)
        allc += list(tfsc[tfsc.lineage_bias > 0].sort_values("score_A", ascending=False).head(6).index)
        allc += list(tfsc[tfsc.lineage_bias <= 0].sort_values("score_B", ascending=False).head(6).index)
    return [g for g in dict.fromkeys(allc) if g in a.var_names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pancreas")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", default="validation", choices=["validation", "discovery"],
                    help="validation = curated known regulators; discovery = above-threshold driver genes")
    ap.add_argument("--only", default=None, help="'cascade' to recompute only the cascade and merge")
    ap.add_argument("--variant", default="", help="fit-cache tag (e.g. 'bimodal'): read adata_analyzed_<tag>.h5ad "
                    "and write perturb_dynamics_<tag>.pkl, alongside the canonical caches.")
    args = ap.parse_args()
    ds, dev = args.dataset, args.device
    discovery = args.mode == "discovery"
    global VARIANT_SUF
    suf = f"_{args.variant}" if args.variant else ""
    VARIANT_SUF = suf

    from config import DATASETS
    cfg = DATASETS[ds]
    ck = cfg["cluster_key"]
    a = ad.read_h5ad(f"{paths.REPORTS}/{ds}/data/adata_analyzed{suf}.h5ad")
    basis = basis_of(a)
    colors = get_colors(a, ck)

    # lineage pairs: reuse the report's resolver (config lineage_pairs -> lineages -> data-driven),
    # so datasets that use `lineages` (e.g. paul15 erythroid/myeloid) or data-driven pairs work too.
    from sections import _lineage_pairs
    present = present_clusters(a, ck)
    lps = [(list(A), list(B), An, Bn) for A, B, An, Bn in _lineage_pairs(a, ds, ck, cfg)]

    if discovery:                                             # discovery = probe wing candidates, keep best
        out = f"{paths.REPORTS}/{ds}/data/perturb_dynamics_discovery{suf}.pkl"
        if args.only:                                         # reuse the cached (deterministic) selection
            tfs, groups = [], {}
        else:
            known = [g for g in TFS_BY_DATASET.get(ds, []) if g in a.var_names]
            # keep the discovery figure at 6 genes: 3 per pair for a two-decision dataset, or 6 from
            # the single pair when there is only one lineage decision.
            per_pair = 6 if len(lps) == 1 else 3
            tfs, groups = probe_select_discovery(a, ds, ck, lps, known, basis,
                                                 TRANSITIONAL_BY_DATASET.get(ds), per_pair=per_pair)
    else:                                                     # curated known regulators
        tfs = [g for g in TFS_BY_DATASET.get(ds, []) if g in a.var_names]
        groups = TF_GROUPS.get(ds, {})
        out = f"{paths.REPORTS}/{ds}/data/perturb_dynamics{suf}.pkl"
    print(f"[{ds}] mode={args.mode} basis={basis} tfs={tfs}", flush=True)

    if args.only == "cascade":                                # recompute only the cascade, keep the rest
        with open(out, "rb") as fh:
            cache = pickle.load(fh)
        tfs = cache.get("tfs", tfs)                            # cached selection (matches the cascade genes)
        cache["cascade"] = compute_cascade(a, ck, tfs, dev)
        cache["cascade_time"] = CASCADE_TMAX
        with open(out, "wb") as fh:
            pickle.dump(cache, fh)
        print(f"[{ds}] cascade recomputed -> {out} (rows={len(cache['cascade'])})", flush=True)
        return

    if args.only == "fatepanels":                             # recompute all three fate panels, keep the rest
        with open(out, "rb") as fh:                            # (reuses the existing cascade)
            cache = pickle.load(fh)
        tfs = cache.get("tfs", tfs)                            # cached selection (matches the cascade genes)
        groups = cache.get("tf_groups", groups)
        from _fate_probability import per_cell_fate_shift, dose_fate_bias, pairwise_fate_bias
        transitional = TRANSITIONAL_BY_DATASET.get(ds)
        print(f"[d] per-cell fate-shift map ({len(tfs)} KOs)", flush=True)
        cache["fate_map"] = per_cell_fate_shift(a, ck, lps, tfs, basis=basis)
        print(f"[d2] KO displacement flow (delta x) ({len(tfs)} KOs)", flush=True)
        cache["ko_flow"], cache["wt_ode_flow"] = compute_ko_flow(a, ck, tfs, basis, dev)
        cache["wt_flow"] = wt_flow_field(a, ck, basis, cfg)
        if not discovery:                                     # e: single-KO fate-shift bar (decider-cell mean)
            cache["fate_bias"] = {}; cache["fate_pvals"] = {}; cache["single_bias"] = {}
            allc = fate_bias_candidates(a, ck, lps, tfs)
            print(f"[e] fate-probability lineage-bias screen ({len(allc)} candidates)", flush=True)
            for (An, Bn), d in pairwise_fate_bias(a, ck, lps, allc, basis=basis,
                                                  transitional=transitional).items():
                cache["fate_bias"][(An, Bn)] = d["bias"]; cache["fate_pvals"][(An, Bn)] = d["pvals"]
        print(f"[f] fate-probability dose response ({len(tfs)} genes)", flush=True)
        cache["fate_dose"] = dose_fate_bias(a, ck, lps, tfs, fractions=FRACTIONS, basis=basis,
                                            transitional=transitional)
        for k in ("ko_ip", "dose"):                           # drop the retired projected readouts
            cache.pop(k, None)
        with open(out, "wb") as fh:
            pickle.dump(cache, fh)
        print(f"[{ds}] fate panels (map + flow + bias + dose) merged -> {out}", flush=True)
        return

    if args.only == "koflow":                                 # add just the KO displacement-flow field
        with open(out, "rb") as fh:
            cache = pickle.load(fh)
        tfs = cache.get("tfs", tfs)
        print(f"[d2] KO displacement flow (delta x) ({len(tfs)} KOs)", flush=True)
        cache["ko_flow"], cache["wt_ode_flow"] = compute_ko_flow(a, ck, tfs, basis, dev)
        cache["wt_flow"] = wt_flow_field(a, ck, basis, cfg)
        with open(out, "wb") as fh:
            pickle.dump(cache, fh)
        print(f"[{ds}] KO flow merged -> {out}", flush=True)
        return

    if args.only == "jacobian":                               # add the two first-order Jacobian predictions
        with open(out, "rb") as fh:
            cache = pickle.load(fh)
        tfs = cache.get("tfs", tfs)
        groups = cache.get("tf_groups", groups)
        print(f"[jac] Jacobian predictions (response + push) for {len(tfs)} KOs", flush=True)
        resp, resp_ct, push, reg = compute_jacobian_predictions(a, ck, tfs, lps, groups)
        cache["jac_response"] = resp; cache["jac_response_ct"] = resp_ct
        cache["commit_push"] = push; cache["out_strength"] = reg
        with open(out, "wb") as fh:
            pickle.dump(cache, fh)
        print(f"[{ds}] Jacobian predictions merged -> {out}", flush=True)
        return

    cache = dict(dataset=ds, basis=basis, cluster_key=ck,
                 emb=np.asarray(a.obsm[f"X_{basis}"])[:, :2].astype(float),
                 clusters=a.obs[ck].astype(str).values, colors=dict(colors),
                 cluster_order=[c for c in (cfg.get("order") or present) if c in present],
                 tfs=tfs, tf_groups=groups,
                 lineage_pairs=[(A, B, An, Bn) for A, B, An, Bn in lps])

    # ---- d: per-cell fate-probability shift MAP (the projection-free spatial version of the panel-e
    #        decision). For each KO, every cell is colored by its change in the A-vs-B fate split
    #        fraction (KO - WT); a pure sink gives ~0 everywhere. Replaces the old projected
    #        displacement / inner-product map (ko_flow/ko_ip), which shared the projection artifact
    #        we retired from the bias. ----
    from _fate_probability import per_cell_fate_shift
    print(f"[d] per-cell fate-shift map ({len(tfs)} KOs)", flush=True)
    cache["fate_map"] = per_cell_fate_shift(a, ck, lps, tfs, basis=basis)
    print(f"[d2] KO displacement flow (delta x) ({len(tfs)} KOs)", flush=True)
    cache["ko_flow"], cache["wt_ode_flow"] = compute_ko_flow(a, ck, tfs, basis, dev)
    cache["wt_flow"] = wt_flow_field(a, ck, basis, cfg)

    # ---- c: single-KO lineage bias per lineage pair, as a FATE-PROBABILITY shift (validation figure
    #        only; discovery drops it). Replaces the old run_ko_screen projected-cosine bias, which was
    #        fooled by the high-expression/sink projection artifact (Malat1). This measures the change
    #        in terminal-state absorption probability (WT vs KO) aggregated per lineage arm; a sink
    #        (no out-edges) gives exactly 0. See _fate_probability.py / fate-probability-metric memory. ----
    cache["fate_bias"] = {}; cache["fate_pvals"] = {}
    cache["single_bias"] = {}                               # kept empty for backward-compat readers
    if not discovery:
        from _fate_probability import pairwise_fate_bias
        allc = fate_bias_candidates(a, ck, lps, tfs)
        print(f"[c] fate-probability lineage-bias screen ({len(allc)} candidates)", flush=True)
        fb = pairwise_fate_bias(a, ck, lps, allc, basis=basis,
                                transitional=TRANSITIONAL_BY_DATASET.get(ds))
        for (An, Bn), d in fb.items():
            cache["fate_bias"][(An, Bn)] = d["bias"]
            cache["fate_pvals"][(An, Bn)] = d["pvals"]

    # ---- f: dose-response of the fate split-fraction (fate-based; dose=0 reproduces the panel-e KO,
    #        so panel e is the dose-0 slice of panel f). Replaces the projected-cosine lineage-bias
    #        dose-response, which used the retired flow-alignment metric. ----
    from _fate_probability import dose_fate_bias
    print(f"[f] fate-probability dose response ({len(tfs)} genes)", flush=True)
    cache["fate_dose"] = dose_fate_bias(a, ck, lps, tfs, fractions=FRACTIONS, basis=basis,
                                        transitional=TRANSITIONAL_BY_DATASET.get(ds))

    # ---- e: short-time cascade relative to WT (sequential state advancement) ----
    cache["cascade"] = compute_cascade(a, ck, tfs, dev)
    cache["cascade_time"] = CASCADE_TMAX

    with open(out, "wb") as fh:
        pickle.dump(cache, fh)
    print(f"[{ds}] mode={args.mode} cached -> {out}  "
          f"(tfs={tfs}; pairs={[ (An,Bn) for _,_,An,Bn in lps]}; "
          f"cascade rows={len(cache['cascade'])})", flush=True)


if __name__ == "__main__":
    main()
