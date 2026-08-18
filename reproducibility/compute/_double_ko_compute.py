"""Compute + cache the combinatorial (double-knockout) perturbation analyses for the second
perturbation figure (make_double_perturbation.py). Companion to _perturb_dynamics_compute.py, which
does the single-KO figure; this module reuses the exact same projection-free fate-probability engine
(sch.tl.perturbed_fate) so the two figures are directly comparable, differing only in how many
genes are clamped at once.

Two caches are written, both under ``<SCHOPFIELD_REPORTS>/<ds>/data/``.
``double_ko_screen.pkl`` holds the all-pairs screen and is the one the figure reads; it is written
only by ``--only screen``. ``double_ko.pkl`` holds the legacy single-block caches and is written by
the other modes.

Panels it feeds:
  a  anchor-partner double KO: fix an anchor regulator, knock it out jointly with each partner, and
     show BOTH the total joint fate shift and the non-additive (synergy / epistasis) component.
  b  perturbation matrix: the double-KO fate shift over every pair of drivers (diagonal = single KO),
     plus the synergy matrix (double minus the two singles).
  c  per-cell fate-shift map for the interesting pairs (spatial, projection-free, like the single fig).
  d  double-KO displacement flow + scalar product with development (computed by --only flow; GPU).
  e  Jacobian commitment push for the interesting pairs (first-order, additive over the two genes).
  f  literature tier classification of the interesting pairs (annotated in the figure module; T1
     known/validated, T2 mentioned-but-unexplored, T3 novel for this process). No compute here.

All fate readouts change a terminal-state absorption probability, so a pure sink gene (no out-edges,
e.g. Malat1) gives ~0 by construction, exactly as in the single-KO figure.

Run:  python reproducibility/compute/_double_ko_compute.py --dataset pancreas
      python reproducibility/compute/_double_ko_compute.py --dataset pancreas --only flow   # heavy GPU
"""
from __future__ import annotations
import argparse, itertools, os, sys, pickle
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
import scHopfield as sch                                          # noqa: E402
from sections import basis_of, get_colors, present_clusters, _lineage_pairs   # noqa: E402
from _perturb_dynamics_compute import (TFS_BY_DATASET, TF_GROUPS,   # noqa: E402
                                       TRANSITIONAL_BY_DATASET, wt_flow_field)

# Anchor regulator per lineage decision: the gene held knocked out while every partner is swept in
# panel a. Pancreas: Neurog3 is the master endocrine switch (progenitor -> differentiated), Nkx2-2 the
# alpha/beta decision node. Keyed by (A_name, B_name); a pair with no entry uses its strongest single
# driver as the anchor (see _resolve_anchor).
ANCHOR_BY_PAIR = {
    "pancreas": {("differentiated", "progenitor"): "Neurog3", ("alpha", "beta"): "Nkx2-2"},
}
# Data-driven (discovery) anchors: a gene surfaced by the discovery screen, held knocked out against the
# curated drivers of its decision, to test whether a *discovered* candidate combinatorially modulates the
# known regulators. Pancreas: Prox1 (endocrine-progenitor TF) and Vdr (beta-identity factor).
DISCOVERY_ANCHOR_BY_PAIR = {
    "pancreas": {("differentiated", "progenitor"): "Prox1", ("alpha", "beta"): "Vdr"},
}
N_INTERESTING = 3                                                 # interesting pairs kept per decision

# Canonical KNOWN developmental regulators per system. On the curated side the screen keeps only pairs
# with >=1 gene from this set, so a known regulator carries all the way through the deep-dive; any other
# regulator is "discovered" (surfaced by the method). Must contain that system's TF_GROUPS anchors.
CURATED_TFS = {
    "pancreas": {"Pdx1", "Ptf1a", "Nkx6-1", "Nkx6-2", "Sox9", "Hnf1b", "Foxa1", "Foxa2", "Foxa3",
                 "Gata4", "Gata6", "Onecut1", "Onecut2", "Mnx1", "Neurog3", "Neurod1", "Neurod2",
                 "Insm1", "Rfx6", "Pax4", "Pax6", "Arx", "Nkx2-2", "Isl1", "Mafa", "Mafb", "Hnf1a",
                 "Hnf4a", "Glis3", "Myt1", "Sox4", "Pou3f4", "Rfx3", "Prox1", "Meis2"},
    # hematopoiesis (mouse): erythroid vs myeloid master regulators
    "paul15": {"Gata1", "Gata2", "Klf1", "Spi1", "Cebpa", "Cebpe", "Cebpb", "Gfi1", "Gfi1b", "Fli1",
               "Tal1", "Lmo2", "Lyl1", "Runx1", "Myb", "Zfpm1", "Nfe2", "Irf8", "Ikzf1", "Bcl11a",
               "Meis1", "Hoxa9", "Sox6", "Klf4", "Egr1", "Junb", "Etv6", "Mef2c", "Stat3", "E2f4", "Nr4a1"},
    # hematopoiesis (human, dynamo, uppercase symbols)
    "dynamo_hematopoiesis": {"GATA1", "GATA2", "KLF1", "SPI1", "CEBPA", "CEBPE", "CEBPB", "GFI1", "GFI1B",
                             "FLI1", "TAL1", "LMO2", "LYL1", "RUNX1", "MYB", "ZFPM1", "NFE2", "IRF8",
                             "IKZF1", "BCL11A", "MEIS1", "HOXA9", "SOX6", "KLF4", "EGR1", "JUNB", "ETV6",
                             "MEF2C", "STAT3", "E2F4", "NR4A1"},
    # neural crest (mouse): SoxE / melanocyte / neuronal-sensory regulators
    "murine_nc": {"Sox10", "Sox9", "Sox8", "Sox2", "Foxd3", "Tfap2a", "Tfap2b", "Pax3", "Pax7", "Mitf",
                  "Ets1", "Zeb2", "Twist1", "Snai2", "Phox2b", "Phox2a", "Ascl1", "Neurog2", "Neurod1",
                  "Isl1", "Pou4f1", "Gata3", "Gata2", "Hand2", "Dlx1", "Dlx2", "Six1", "Runx3", "Prdm12",
                  "Id2", "Mycn"},
    # myogenesis (human limb, uppercase): MRFs + progenitor/specification TFs
    "human_limb": {"PAX3", "PAX7", "MYF5", "MYOD1", "MYOG", "MYF6", "MEF2A", "MEF2C", "MEF2D", "PITX2",
                   "SIX1", "SIX4", "EYA1", "EYA2", "TCF21", "LBX1", "SIM1", "MSC", "SOX8", "TWIST2",
                   "TBX3", "HOXA9", "HOXA11", "HOXA13", "ID3"},
    # Schwann / neural crest lineage
    "schwann": {"Sox10", "Sox8", "Sox2", "Sox9", "Egr2", "Pou3f1", "Pou3f2", "Nfatc4", "Yy1", "Zeb2",
                "Tfap2a", "Tfap2b", "Pax3", "Foxd3", "Ets1", "Id2", "Nab1", "Nab2", "Phox2b", "Phox2a",
                "Ascl1", "Gata3", "Hand2", "Runx3"},
}
CURATED_TFS["paul15_coarse"] = CURATED_TFS["paul15"]             # same hematopoietic regulators
N_PARTNER = 10                                                   # candidates per lineage screened in panel a (-> 20)
N_MATRIX = 5                                                     # genes per lineage kept for the square matrix (5+5)
N_BEST = 6                                                       # top synergy/cancellation pairs kept per block (for f)
N_BEST_CDE = 2                                                   # best pairs per block shown in c/d/e (balanced)
N_CAND = 30                                                      # driver-bias pre-filter per lineage before the Jacobian re-rank


VARIANT_SUF = ""   # set in main() from --variant; suffixes every reports/<ds>/data cache path


def _anchor_source_groups(ds):
    """Per (lineage-decision k, origin) the pool of KNOWN genes the 2 anchors are drawn from, taken from
    the SINGLE-KO figure so the two figures agree: curated = its biological TF groups (TF_GROUPS);
    discovered = its data-driven discovery genes (the cached tf_groups from the discovery run). Group k
    is tied to lineage pair k in both, so the ordering lines up."""
    src = {}
    for k, genes in enumerate(TF_GROUPS.get(ds, {}).values()):
        src[(k, "curated")] = list(genes)
    try:
        dg = pickle.load(open(f"{paths.REPORTS}/{ds}/data/perturb_dynamics_discovery{VARIANT_SUF}.pkl", "rb"))["tf_groups"]
        for k, genes in enumerate(dg.values()):
            src[(k, "discovered")] = list(genes)
    except Exception as e:                                        # discovery run not cached -> curated only
        print(f"[screen] no discovery anchor source ({e})", flush=True)
    return src


def _resolve_anchor(ds, An, Bn, drivers):
    a = ANCHOR_BY_PAIR.get(ds, {}).get((An, Bn))
    return a if a in drivers else (drivers[-1] if drivers else None)


def _pair_axes(sc, lps):
    """Per lineage pair: the arm columns, the WT split fraction and the decider focus mask, with
    this dataset's transitional populations applied so the summaries match the single-KO figure."""
    return sch.tl.lineage_pair_axes(sc, lps, TRANSITIONAL_BY_DATASET.get(sc["ds"], {}))


def compute_matrix(adata, ck, sc, lps, drivers):
    """Panels a + b. Per lineage decision, the single-KO fate shift for each driver (matrix diagonal),
    the joint double-KO fate shift for every driver pair (off-diagonal), and the synergy = double minus
    the sum of the two singles. All aggregated over the decision's decider cells. Returns a per-pair
    dict with the driver order, the shift matrix, the synergy matrix, the single-KO Series, and the
    anchor-partner bars (total joint shift + synergy for the fixed anchor against each partner)."""
    axes = _pair_axes(sc, lps)
    blocks, fate_single, fate_double = sch.tl.double_knockout_matrix(adata, ck, sc, drivers, axes)
    print(f"[double-ko] {len(drivers)} singles + {len(fate_double)} pairs", flush=True)

    out = {}
    for A, B, An, Bn in lps:
        block = blocks[(An, Bn)]
        genes = block["genes"]
        idx = {g: i for i, g in enumerate(genes)}
        syn = block["synergy"]
        anchor = _resolve_anchor(sc["ds"], An, Bn, drivers)
        partners = [g for g in drivers if g != anchor]
        total = pd.Series({g: block["matrix"][idx[anchor], idx[g]] for g in partners})
        syn_bar = pd.Series({g: block["synergy_matrix"][idx[anchor], idx[g]] for g in partners})
        interesting = sorted(syn, key=lambda k: abs(syn[k]), reverse=True)[:N_INTERESTING]
        out[(An, Bn)] = dict(drivers=drivers, matrix=block["matrix"],
                             synergy_matrix=block["synergy_matrix"], single=block["single"],
                             synergy=syn, anchor=anchor, partners=partners,
                             anchor_total=total, anchor_synergy=syn_bar, interesting=interesting)
    return out, fate_single, fate_double, axes


def compute_deep(adata, ck, sc, lps, mat, axes):
    """Panels c + e for the interesting pairs. c: per-cell change in the A-vs-B fate split fraction under
    the joint double KO (spatial map). e: first-order Jacobian 'commitment push' for the joint KO,
    approximated as the sum of the two single-gene pushes (the first-order response is linear in the
    clamped coordinates, and both partners of an interesting pair share the decision axis)."""
    groups = TF_GROUPS.get(sc["ds"], {})
    push_single = {g: arr for g, (_, _, arr) in
                   sch.tl.jacobian_commitment_push(adata, ck, lps, groups).items()}
    fate_map, push = {}, {}
    for A, B, An, Bn in lps:
        a = axes[(An, Bn)]
        fmap, pmap = {}, {}
        for (g1, g2) in mat[(An, Bn)]["interesting"]:
            fate = sch.tl.perturbed_fate(adata, ck, sc, [g1, g2])
            fmap[(g1, g2)] = sch.tl.split_fraction(fate, a["Ac"], a["Bc"]) - a["split_wt"]
            if g1 in push_single and g2 in push_single:
                pmap[(g1, g2)] = push_single[g1] + push_single[g2]
        fate_map[(An, Bn)] = fmap
        push[(An, Bn)] = pmap
    return fate_map, push


def compute_discovery_anchor(adata, ck, sc, lps, axes, ds):
    """Panel a (discovery-anchor block). For each decision, hold a DISCOVERED gene knocked out against
    each curated driver of that decision and report the total joint fate shift and the synergy. Answers:
    does a data-driven candidate combinatorially modulate the known regulators? Returns per-decision
    {disc_anchor, disc_partners, disc_total (Series), disc_synergy (Series), disc_anchor_single}."""
    groups = TF_GROUPS.get(ds, {})
    gnames = list(groups)
    out = {}
    for k, (A, B, An, Bn) in enumerate(lps):
        anchor = DISCOVERY_ANCHOR_BY_PAIR.get(ds, {}).get((An, Bn))
        if not anchor or anchor not in adata.var_names:
            continue
        partners = [g for g in groups.get(gnames[k], []) if g in adata.var_names] if k < len(gnames) else []
        ax = axes[(An, Bn)]

        def dshift(fate):
            return float((sch.tl.split_fraction(fate, ax["Ac"], ax["Bc"])
                          - ax["split_wt"])[ax["focus"]].mean())

        s_anchor = dshift(sch.tl.perturbed_fate(adata, ck, sc, [anchor]))
        total, syn = {}, {}
        for p in partners:
            s_p = dshift(sch.tl.perturbed_fate(adata, ck, sc, [p]))
            s_ap = dshift(sch.tl.perturbed_fate(adata, ck, sc, [anchor, p]))
            total[p] = s_ap
            syn[p] = s_ap - (s_anchor + s_p)
        out[(An, Bn)] = dict(disc_anchor=anchor, disc_partners=partners,
                             disc_total=pd.Series(total), disc_synergy=pd.Series(syn),
                             disc_anchor_single=s_anchor)
        print(f"  disc-anchor {anchor} ({An} vs {Bn}) vs {partners}", flush=True)
    return out


def _screen_blocks(adata, ck, lps, ds, reg, sc, n_cand=N_CAND):
    """Per (lineage decision k) x (origin in {curated, discovered}), FIX the 2 anchors from the single-KO
    figure's known genes (one leaning toward each lineage by driver bias = score_A - score_B) and
    pre-select the driver-score candidate partners the Jacobian re-rank (compute_screen) then scores: the
    top ``n_cand`` regulators biasing toward each lineage. Returns per block {anchors, candA, candB,
    scores}; compute_screen fills partners/partA/partB via the driver x Jacobian score."""
    src = _anchor_source_groups(ds)
    names = set(sc["names"])
    meta = {}
    for k, (A, B, An, Bn) in enumerate(lps):
        df = pd.read_csv(f"{paths.REPORTS}/{ds}/data/driver_scores_{k + 1}{VARIANT_SUF}.csv", index_col=0)
        df = df[reg.reindex(df.index).fillna(0).values > 0].copy()      # regulators only
        df["bias"] = df["score_A"] - df["score_B"]
        for origin in ("curated", "discovered"):
            source = [g for g in src.get((k, origin), []) if g in df.index and g in names]
            if len(source) < 2:
                continue
            sb = df.loc[source, "bias"].sort_values(ascending=False)
            anchors = (sb.index[0], sb.index[-1])                        # anchors: toward A / toward B
            pool = df.drop(index=[g for g in anchors], errors="ignore")
            candA = list(pool.sort_values("bias", ascending=False).index[:n_cand])   # driver-lean toward A
            candB = list(pool.sort_values("bias", ascending=True).index[:n_cand])    # driver-lean toward B
            meta[(k, origin)] = dict(An=An, Bn=Bn, origin=origin, anchors=anchors,
                                     candA=candA, candB=candB,
                                     scores=df[["score_A", "score_B", "bias"]].copy())
    return meta


def compute_screen(adata, ck, lps, ds, basis, reg, n_partner=N_PARTNER, n_matrix=N_MATRIX, workers=24):
    """The 4-block screen. Panel a: for each block's two anchors, the SYNERGY against all ~20 screened
    partners. Panel b: the 5+5 genes selected from those 20 by strongest synergy (anchors forced in), all
    pairs -> a square matrix (lower = double-KO fate shift, upper = synergy, diagonal = single). Two CPU-
    parallel rounds (anchor-partner doubles, then matrix doubles), fates shared across blocks."""
    import time as _t
    sc = sch.tl.fate_scaffold(adata, ck, lps, basis=basis); sc["ds"] = ds
    axes = _pair_axes(sc, lps)
    meta = _screen_blocks(adata, ck, lps, ds, reg, sc)

    # ---- partner score = driver bias toward the anchor's lineage x Jacobian coupling to the anchor ----
    # (a PREDICTIVE prior: a good paired-KO partner both drives the same lineage as the anchor and is
    # coupled to it in the fitted dynamics; panels b/c then show the ACTUAL synergy/shift as the test).
    need = sorted({g for mb in meta.values() for g in list(mb["anchors"]) + mb["candA"] + mb["candB"]})
    print(f"[screen] Jacobian coupling over {len(need)} genes", flush=True)
    Ccoup = sch.tl.regulatory_coupling(adata, need)

    def _couple(anc, g):                                            # |cos(Jacobian columns)| in [0,1]
        try:
            return float(Ccoup.loc[anc, g])
        except Exception:
            return 0.0

    for (k, origin), mb in meta.items():
        aA, aB = mb["anchors"]; sco = mb["scores"]

        def _rank_side(anchor, cands, toward_A):
            rows = [(g, sco.loc[g, "bias"] * (1.0 if toward_A else -1.0), _couple(anchor, g)) for g in cands]
            dfc = pd.DataFrame(rows, columns=["gene", "driver", "coupling"]).set_index("gene")
            dfc["combined"] = dfc["driver"].rank(pct=True) + dfc["coupling"].rank(pct=True)
            top = list(dfc.sort_values("combined", ascending=False).index[:n_partner])
            return top, dfc

        partA, dfA = _rank_side(aA, mb["candA"], True)
        partB, dfB = _rank_side(aB, mb["candB"], False)
        partners = list(dict.fromkeys(partA + partB))
        allc = list(dict.fromkeys(list(mb["anchors"]) + mb["candA"] + mb["candB"]))
        cand = sco.reindex(allc).copy()
        cand["coupA"] = [_couple(aA, g) for g in allc]; cand["coupB"] = [_couple(aB, g) for g in allc]
        cand["combA"] = dfA["combined"].reindex(allc); cand["combB"] = dfB["combined"].reindex(allc)
        mb.update(partners=partners, partA=partA, partB=partB, candidates=cand,
                  bias=sco["bias"].reindex(partners))
        print(f"  [{origin} | {mb['An']} vs {mb['Bn']}] anchors={mb['anchors']} "
              f"partA={partA[:5]} partB={partB[:5]}", flush=True)

    def dshift(fate, ax):
        return float((sch.tl.split_fraction(fate, ax["Ac"], ax["Bc"])
                      - ax["split_wt"])[ax["focus"]].mean())

    # ---- round 1: singles (union) + anchor x 20-partner doubles ----
    singles = sorted({g for b in meta.values() for g in b["partners"]} |
                     {a for b in meta.values() for a in b["anchors"]})
    ap_pairs = sorted({tuple(sorted((anc, p))) for b in meta.values()
                       for anc in b["anchors"] for p in b["partners"] if anc != p})
    print(f"[screen] round1: {len(singles)} singles + {len(ap_pairs)} anchor-partner doubles", flush=True)
    t0 = _t.time()
    r1 = sch.tl.perturbed_fates(adata, ck, sc,
                                [[g] for g in singles] + [list(p) for p in ap_pairs], workers)
    fate = {(g,): r1[i] for i, g in enumerate(singles)}
    fate.update({p: r1[len(singles) + i] for i, p in enumerate(ap_pairs)})
    print(f"[screen] round1 done in {_t.time() - t0:.0f}s", flush=True)

    # ---- per block: anchor synergy vs 20; select the 5+5 for the matrix ----
    for (k, origin), mb in meta.items():
        ax = axes[(mb["An"], mb["Bn"])]
        ssingle = {g: dshift(fate[(g,)], ax) for g in mb["partners"] + list(mb["anchors"])}
        a_syn = {}
        for anc in mb["anchors"]:
            a_syn[anc] = {p: dshift(fate[tuple(sorted((anc, p)))], ax) - (ssingle[anc] + ssingle[p])
                          for p in mb["partners"] if p != anc}
        mb["a_syn"] = a_syn; mb["single_shift"] = ssingle

        def maxsyn(g):
            return max((abs(a_syn[anc].get(g, 0.0)) for anc in mb["anchors"]), default=0.0)

        def pick(side, anchor):                                  # anchor forced in, then top by |synergy|
            rest = [g for g in side if g != anchor]
            return [anchor] + sorted(rest, key=maxsyn, reverse=True)[:n_matrix - 1]
        selA = pick(mb["partA"], mb["anchors"][0]); selB = pick(mb["partB"], mb["anchors"][1])
        mb["genes"] = list(dict.fromkeys(selA + selB))

    # ---- round 2: matrix doubles among the 5+5 not already computed ----
    mat_pairs = sorted({tuple(sorted(p)) for b in meta.values()
                        for p in itertools.combinations(b["genes"], 2)})
    need = [p for p in mat_pairs if p not in fate]
    print(f"[screen] round2: {len(need)} matrix doubles", flush=True)
    t0 = _t.time()
    r2 = sch.tl.perturbed_fates(adata, ck, sc, [list(p) for p in need], workers)
    fate.update({p: r2[i] for i, p in enumerate(need)})
    print(f"[screen] round2 done in {_t.time() - t0:.0f}s", flush=True)

    # ---- assemble matrices + best pairs ----
    blocks = {}
    for (k, origin), mb in meta.items():
        An, Bn = mb["An"], mb["Bn"]; ax = axes[(An, Bn)]; genes = mb["genes"]; n = len(genes)
        idx = {g: i for i, g in enumerate(genes)}
        single = {g: mb["single_shift"][g] for g in genes}
        shiftM = np.full((n, n), np.nan); synM = np.full((n, n), np.nan); syn_pairs = {}
        for g in genes:
            shiftM[idx[g], idx[g]] = single[g]                          # diagonal = single KO
        for (g1, g2) in itertools.combinations(genes, 2):
            d = dshift(fate[tuple(sorted((g1, g2)))], ax)
            s = d - (single[g1] + single[g2])
            i, j = idx[g1], idx[g2]
            shiftM[max(i, j), min(i, j)] = d                            # lower = fate shift
            synM[min(i, j), max(i, j)] = s                              # upper = synergy
            syn_pairs[(g1, g2)] = s
        # CURATED side: every carried pair must contain >=1 known (curated) gene, so a canonical
        # regulator propagates all the way through the deep-dive (c/d/e) and the tier table (f). The
        # DISCOVERED side keeps all pairs (data-driven, may be all-novel).
        curated_set = CURATED_TFS.get(ds, set())
        cand_pairs = ([p for p in syn_pairs if p[0] in curated_set or p[1] in curated_set]
                      if origin == "curated" else list(syn_pairs))
        best = sorted(cand_pairs, key=lambda p: abs(syn_pairs[p]), reverse=True)[:N_BEST]
        # panel-a synergy as Series per anchor over the 20 partners
        a_series = {anc: pd.Series(mb["a_syn"][anc]) for anc in mb["anchors"]}
        blocks[(k, origin)] = dict(genes=genes, partners=mb["partners"], anchors=mb["anchors"],
                                   An=An, Bn=Bn, origin=origin, shift=shiftM, synergy=synM,
                                   single=pd.Series(single), bias=mb["bias"], syn_pairs=syn_pairs,
                                   best=best, a_synergy=a_series, candidates=mb["candidates"])
        print(f"  [{origin} | {An} vs {Bn}] anchors={mb['anchors']} genes={genes} best={best[:3]}",
              flush=True)
    # rebuild fate_single / fate_double views for the deep-dive (best pairs)
    fate_single = {g: fate[(g,)] for g in singles}
    fate_double = {p: fate[p] for p in fate if len(p) == 2}
    return blocks, sc, axes, fate_single, fate_double


def compute_double_flow(adata, ck, pairs_by_decision, basis, dev):
    """Panel d (heavy, GPU). Joint double-KO displacement flow for the interesting pairs, computed
    exactly like the single-KO figure's compute_ko_flow (simulate_shift_ode with both genes held at 0,
    absolute displacement projected with the correlation scheme), plus the WT ODE flow so the figure can
    color by the KO-specific residual alignment with development. Returns
    ({(g1,g2): flow (n,2)}, wt_ode_flow (n,2))."""
    return sch.dyn.knockout_displacement_flow(adata, ck, list(pairs_by_decision), basis=basis,
                                             device=dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pancreas")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default=None, choices=[None, "flow", "disc-anchor", "screen", "screenflow"],
                    help="screen = the 4-block 10x10 all-pairs screen (curated/discovery x 2 decisions) "
                         "+ deep-dive fate maps/pushes; screenflow = add the flow for the best pairs; "
                         "flow / disc-anchor = legacy single-block cache updates")
    ap.add_argument("--workers", type=int, default=24, help="CPU workers for the parallel fate screen")
    ap.add_argument("--variant", default="", help="fit-cache tag (e.g. 'bimodal'): read adata_analyzed_<tag>.h5ad "
                    "+ the <tag> discovery cache, and write double_ko[_screen]_<tag>.pkl.")
    args = ap.parse_args()
    ds, dev = args.dataset, args.device
    global VARIANT_SUF
    VARIANT_SUF = f"_{args.variant}" if args.variant else ""
    out = f"{paths.REPORTS}/{ds}/data/double_ko{VARIANT_SUF}.pkl"
    screen_out = f"{paths.REPORTS}/{ds}/data/double_ko_screen{VARIANT_SUF}.pkl"

    from config import DATASETS
    cfg = DATASETS[ds]
    ck = cfg["cluster_key"]
    a = ad.read_h5ad(f"{paths.REPORTS}/{ds}/data/adata_analyzed{VARIANT_SUF}.h5ad")
    basis = basis_of(a)
    lps = [(list(A), list(B), An, Bn) for A, B, An, Bn in _lineage_pairs(a, ds, ck, cfg)]
    drivers = [g for g in TFS_BY_DATASET.get(ds, []) if g in a.var_names]

    if args.only == "screen":
        reg = sch.tl.regulatory_out_strength(a, ck)
        blocks, sc, axes, fate_single, fate_double = compute_screen(a, ck, lps, ds, basis, reg,
                                                                    workers=args.workers)
        # balanced c/d/e: N_BEST_CDE synergy/cancellation pairs from EACH block, deduplicated GLOBALLY so
        # the same pair is never shown twice across the curated/discovery halves (the shared partner pool
        # can make a decision's two blocks agree on the top pair); a block then shows its next-best unique.
        seen = set(); best_sel = []
        for (k, origin), b in blocks.items():
            cnt = 0
            for pr in b["best"]:
                key = tuple(sorted(pr))
                if key in seen:
                    continue
                seen.add(key); best_sel.append((key, k, origin)); cnt += 1
                if cnt >= N_BEST_CDE:
                    break
        fate_map = {}                                                # per-cell split shift (reuse fate_double)
        for pr, k, origin in best_sel:
            An, Bn = lps[k][2], lps[k][3]; ax = axes[(An, Bn)]
            fate_map[(pr, k, origin)] = (sch.tl.split_fraction(fate_double[pr], ax["Ac"],
                                                               ax["Bc"]) - ax["split_wt"])
        gnames = [f"{An} vs {Bn}" for _, _, An, Bn in lps]           # jacobian push on the fate axis
        groups_for_push = {n: [] for n in gnames}
        for pr, k, origin in best_sel:
            groups_for_push[gnames[k]] += [pr[0], pr[1]]
        groups_for_push = {n: list(dict.fromkeys(g)) for n, g in groups_for_push.items()}
        push_single = {g: arr for g, (_, _, arr) in
                       sch.tl.jacobian_commitment_push(a, ck, lps, groups_for_push).items()}
        push_map = {(pr, k, origin): push_single[pr[0]] + push_single[pr[1]]
                    for pr, k, origin in best_sel if pr[0] in push_single and pr[1] in push_single}
        present = present_clusters(a, ck)
        cache = dict(dataset=ds, basis=basis, cluster_key=ck,
                     emb=np.asarray(a.obsm[f"X_{basis}"])[:, :2].astype(float),
                     clusters=a.obs[ck].astype(str).values, colors=dict(get_colors(a, ck)),
                     cluster_order=[c for c in (cfg.get("order") or present) if c in present],
                     lineage_pairs=[(A, B, An, Bn) for A, B, An, Bn in lps],
                     curated=sorted(CURATED_TFS.get(ds, set())),
                     blocks=blocks, best_sel=best_sel, fate_map=fate_map, push=push_map)
        with open(screen_out, "wb") as fh:
            pickle.dump(cache, fh)
        print(f"[{ds}] screen cached -> {screen_out}  ({len(blocks)} blocks, {len(best_sel)} best pairs)",
              flush=True)
        return

    if args.only == "screenflow":
        with open(screen_out, "rb") as fh:
            cache = pickle.load(fh)
        pairs = [pr for pr, k, origin in cache["best_sel"]]
        print(f"[{ds}] screen flow for {len(set(pairs))} best pairs", flush=True)
        cache["flow"], cache["wt_ode_flow"] = compute_double_flow(a, ck, pairs, basis, dev)
        cache["wt_flow"] = wt_flow_field(a, ck, basis, cfg)
        with open(screen_out, "wb") as fh:
            pickle.dump(cache, fh)
        print(f"[{ds}] screen flow merged -> {screen_out}", flush=True)
        return

    if args.only == "disc-anchor":
        with open(out, "rb") as fh:
            cache = pickle.load(fh)
        sc = sch.tl.fate_scaffold(a, ck, lps, basis=basis)
        sc["ds"] = ds
        axes = _pair_axes(sc, lps)
        da = compute_discovery_anchor(a, ck, sc, lps, axes, ds)
        for pair, d in da.items():
            cache["decision"][pair].update(d)
        with open(out, "wb") as fh:
            pickle.dump(cache, fh)
        print(f"[{ds}] discovery-anchor block merged -> {out}", flush=True)
        return

    if args.only == "flow":
        with open(out, "rb") as fh:
            cache = pickle.load(fh)
        pairs = []
        for pair in cache["lineage_pairs"]:
            An, Bn = pair[2], pair[3]
            pairs += cache["decision"][(An, Bn)]["interesting"]
        print(f"[{ds}] double-KO flow for {len(set(pairs))} interesting pairs", flush=True)
        cache["flow"], cache["wt_ode_flow"] = compute_double_flow(a, ck, pairs, basis, dev)
        cache["wt_flow"] = wt_flow_field(a, ck, basis, cfg)      # projected WT reference velocity (for the IP color)
        with open(out, "wb") as fh:
            pickle.dump(cache, fh)
        print(f"[{ds}] double-KO flow merged -> {out}", flush=True)
        return

    sc = sch.tl.fate_scaffold(a, ck, lps, basis=basis)
    sc["ds"] = ds
    print(f"[{ds}] fate scaffold built; drivers={drivers}", flush=True)
    mat, fate_single, fate_double, axes = compute_matrix(a, ck, sc, lps, drivers)
    fate_map, push = compute_deep(a, ck, sc, lps, mat, axes)

    cache = dict(dataset=ds, basis=basis, cluster_key=ck,
                 emb=np.asarray(a.obsm[f"X_{basis}"])[:, :2].astype(float),
                 clusters=a.obs[ck].astype(str).values, colors=dict(get_colors(a, ck)),
                 cluster_order=[c for c in (cfg.get("order") or present_clusters(a, ck))
                                if c in present_clusters(a, ck)],
                 drivers=drivers, lineage_pairs=[(A, B, An, Bn) for A, B, An, Bn in lps],
                 decision=mat, fate_map=fate_map, push=push)
    with open(out, "wb") as fh:
        pickle.dump(cache, fh)
    print(f"[{ds}] double-KO cached -> {out}  (drivers={drivers}; "
          f"pairs={[(An, Bn) for _, _, An, Bn in lps]})", flush=True)
    for A, B, An, Bn in lps:
        d = mat[(An, Bn)]
        print(f"  {An} vs {Bn}: anchor={d['anchor']} interesting={d['interesting']}", flush=True)


if __name__ == "__main__":
    main()
