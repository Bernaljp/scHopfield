"""Compute + cache the perturbation-dynamics analyses for the composite figure
(make_perturbation_dynamics.py). The ODE integrations are cached here so the figure can be
iterated cheaply.

Every readout is computed by calling the scHopfield public API; what lives in this file is the
paper's configuration and orchestration, not the method. The tables below name which genes and
which lineage decisions each dataset is analysed with, `main` sequences the calls and writes the
cache, and the only computation kept here is the part that reads the report tree, which is not
part of the package.

  a  embedding + WT velocity streamlines            (sch.tl.reference_flow)
  b  KO Delta_x (ODE) + inner product Delta_x . v_ref (sch.dyn.knockout_displacement_flow)
  c  single-KO lineage bias, for each configured lineage pair (sch.tl.pairwise_fate_bias)
  d  dose-response of the lineage bias (sch.tl.dose_fate_bias)
  e  short-time cascade: per-cluster mean |Delta_x| vs ODE time (sch.dyn.perturbation_cascade)

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


def _wing_pool(a, ds, ck, lps, known, q=95, pool_per_wing=6):
    """Per lineage pair, the pool of lineage-specific candidate regulators to probe.

    The selection is `sch.tl.select_specificity_wings`. What is kept here is the part the package
    has no business knowing: reading this dataset's driver scores out of the report tree, and
    carrying the already-spoken-for genes from one decision to the next so a gene is never
    offered twice. Out-strength is summed over the clusters the report considers present, which is
    a stricter set than every label appearing in `obs`. Returns
    (poolmap {pair index: {'A': [genes], 'B': [genes]}}, flat unique candidate list).
    """
    out_strength = sch.tl.regulatory_out_strength(a, ck, clusters=present_clusters(a, ck))
    used = set(known)
    poolmap, flat = {}, []
    for kk, (A, B, An, Bn) in enumerate(lps):
        scores = pd.read_csv(
            f"{paths.REPORTS}/{ds}/data/driver_scores_{kk + 1}{VARIANT_SUF}.csv", index_col=0)
        wings = sch.tl.select_specificity_wings(scores, out_strength, exclude=used,
                                                q=q, pool_per_wing=pool_per_wing)
        poolmap[kk] = wings
        for g in wings["A"] + wings["B"]:
            if g not in used:
                flat.append(g)
                used.add(g)
    return poolmap, flat


def probe_select_discovery(a, ds, ck, lps, known, basis, transitional,
                           q=95, pool_per_wing=6, per_pair=3):
    """Discovery-gene selection by measured perturbation effect, not by driver score alone.

    Builds the wing candidate pool, probes every candidate with the fate-probability metric, then
    keeps the genes with the strongest significant effect per decision. Deterministic, so an
    `--only` re-run reproduces the cached selection.
    """
    poolmap, flat = _wing_pool(a, ds, ck, lps, known, q=q, pool_per_wing=pool_per_wing)
    if not flat:
        return [], {}
    print(f"[discovery] probing {len(flat)} wing candidates with the fate metric", flush=True)
    fb = sch.tl.pairwise_fate_bias(a, ck, lps, flat, basis=basis, transitional=transitional)
    return sch.tl.rank_by_fate_effect(lps, poolmap, fb, per_pair=per_pair)


def wt_flow_field(a, ck, basis, cfg):
    """Projected wild-type reference velocity, stored so the figure can colour the KO displacement
    flow by its alignment with development."""
    return sch.tl.reference_flow(a, basis=basis,
                                 velocity_embedding_key=cfg.get("velocity_embedding_key")).copy()


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
        cache["cascade"] = sch.dyn.perturbation_cascade(a, ck, tfs, device=dev, tmax=CASCADE_TMAX,
                                                        n_segments=CASCADE_NSTEP, verbose=True)
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
        transitional = TRANSITIONAL_BY_DATASET.get(ds)
        print(f"[d] per-cell fate-shift map ({len(tfs)} KOs)", flush=True)
        cache["fate_map"] = sch.tl.per_cell_fate_shift(a, ck, lps, tfs, basis=basis)
        print(f"[d2] KO displacement flow (delta x) ({len(tfs)} KOs)", flush=True)
        cache["ko_flow"], cache["wt_ode_flow"] = sch.dyn.knockout_displacement_flow(
            a, ck, tfs, basis=basis, device=dev)
        cache["wt_flow"] = wt_flow_field(a, ck, basis, cfg)
        if not discovery:                                     # e: single-KO fate-shift bar (decider-cell mean)
            cache["fate_bias"] = {}; cache["fate_pvals"] = {}; cache["single_bias"] = {}
            allc = sch.tl.fate_bias_candidates(a, ck, lps, tfs)
            print(f"[e] fate-probability lineage-bias screen ({len(allc)} candidates)", flush=True)
            for (An, Bn), d in sch.tl.pairwise_fate_bias(a, ck, lps, allc, basis=basis,
                                                         transitional=transitional).items():
                cache["fate_bias"][(An, Bn)] = d["bias"]; cache["fate_pvals"][(An, Bn)] = d["pvals"]
        print(f"[f] fate-probability dose response ({len(tfs)} genes)", flush=True)
        cache["fate_dose"] = sch.tl.dose_fate_bias(a, ck, lps, tfs, fractions=FRACTIONS,
                                                   basis=basis, transitional=transitional)
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
        cache["ko_flow"], cache["wt_ode_flow"] = sch.dyn.knockout_displacement_flow(
            a, ck, tfs, basis=basis, device=dev)
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
        jac = sch.tl.jacobian_knockout_response(a, ck, genes=tfs, lineage_pairs=lps, groups=groups)
        cache["jac_response"] = jac["response"]
        cache["jac_response_ct"] = jac["response_by_celltype"]
        cache["commit_push"] = jac["commitment_push"]
        cache["out_strength"] = sch.tl.regulatory_out_strength(a, ck)
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
    print(f"[d] per-cell fate-shift map ({len(tfs)} KOs)", flush=True)
    cache["fate_map"] = sch.tl.per_cell_fate_shift(a, ck, lps, tfs, basis=basis)
    print(f"[d2] KO displacement flow (delta x) ({len(tfs)} KOs)", flush=True)
    cache["ko_flow"], cache["wt_ode_flow"] = sch.dyn.knockout_displacement_flow(
        a, ck, tfs, basis=basis, device=dev)
    cache["wt_flow"] = wt_flow_field(a, ck, basis, cfg)

    # ---- c: single-KO lineage bias per lineage pair, as a FATE-PROBABILITY shift (validation figure
    #        only; discovery drops it). Replaces the old run_ko_screen projected-cosine bias, which was
    #        fooled by the high-expression/sink projection artifact (Malat1). This measures the change
    #        in terminal-state absorption probability (WT vs KO) aggregated per lineage arm; a sink
    #        (no out-edges) gives exactly 0. See sch.tl.pairwise_fate_bias and the Methods section
    #        "Fate-Probability Lineage Effect". ----
    cache["fate_bias"] = {}; cache["fate_pvals"] = {}
    cache["single_bias"] = {}                               # kept empty for backward-compat readers
    if not discovery:
        allc = sch.tl.fate_bias_candidates(a, ck, lps, tfs)
        print(f"[c] fate-probability lineage-bias screen ({len(allc)} candidates)", flush=True)
        fb = sch.tl.pairwise_fate_bias(a, ck, lps, allc, basis=basis,
                                       transitional=TRANSITIONAL_BY_DATASET.get(ds))
        for (An, Bn), d in fb.items():
            cache["fate_bias"][(An, Bn)] = d["bias"]
            cache["fate_pvals"][(An, Bn)] = d["pvals"]

    # ---- f: dose-response of the fate split-fraction (fate-based; dose=0 reproduces the panel-e KO,
    #        so panel e is the dose-0 slice of panel f). Replaces the projected-cosine lineage-bias
    #        dose-response, which used the retired flow-alignment metric. ----
    print(f"[f] fate-probability dose response ({len(tfs)} genes)", flush=True)
    cache["fate_dose"] = sch.tl.dose_fate_bias(a, ck, lps, tfs, fractions=FRACTIONS, basis=basis,
                                               transitional=TRANSITIONAL_BY_DATASET.get(ds))

    # ---- e: short-time cascade relative to WT (sequential state advancement) ----
    cache["cascade"] = sch.dyn.perturbation_cascade(a, ck, tfs, device=dev, tmax=CASCADE_TMAX,
                                                    n_segments=CASCADE_NSTEP, verbose=True)
    cache["cascade_time"] = CASCADE_TMAX

    with open(out, "wb") as fh:
        pickle.dump(cache, fh)
    print(f"[{ds}] mode={args.mode} cached -> {out}  "
          f"(tfs={tfs}; pairs={[ (An,Bn) for _,_,An,Bn in lps]}; "
          f"cascade rows={len(cache['cascade'])})", flush=True)


if __name__ == "__main__":
    main()
