"""Compare no-scaffold (pseudoinverse) inference vs the 6 scaffold settings.

The network_reg_sensitivity settings all use w_scaffold + only_TFs=True (the
scaffold's TF MASK is active even at reg=0). This runs a TRUE no-scaffold fit
(w_scaffold=None -> Moore-Penrose pseudoinverse, Methods 3.1: full dense W, no TF
restriction), computes the same top driver-score and top perturbation genes, and
measures how they overlap with the scaffold-based results.
"""
import itertools
import json
import os

import numpy as np
import scanpy as sc

import scHopfield as sch
from network_reg_sensitivity import (
    CLUSTER_KEY, SPLICED_KEY, VELOCITY_KEY, DEG_KEY, BASIS, WT_FLOW_KEY,
    ERY, MYE, KO_CANDIDATES, TOPN, OUTDIR,
)


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else float("nan")


def main():
    base = sc.read_h5ad("data/hematopoiesis/base_preprocessed.h5ad")
    ad = base.copy()
    # no-scaffold: pseudoinverse (deterministic, no torch training / no TF mask)
    sch.inf.fit_interactions(
        ad, cluster_key=CLUSTER_KEY, spliced_key=SPLICED_KEY, velocity_key=VELOCITY_KEY,
        degradation_key=DEG_KEY, w_scaffold=None, skip_all=True, w_threshold=1e-12,
    )
    print("no-scaffold pseudoinverse fit done", flush=True)

    scores = sch.tl.score_driver_tfs(ad, lineage_A_clusters=ERY, lineage_B_clusters=MYE,
                                     cluster_key=CLUSTER_KEY)
    top_score_ery = scores.sort_values("score_A", ascending=False).head(TOPN).index.tolist()
    top_score_mye = scores.sort_values("score_B", ascending=False).head(TOPN).index.tolist()

    sch.tl.calculate_flow(ad, source="original", basis=BASIS, method="hopfield",
                          cluster_key=CLUSTER_KEY, store_key=WT_FLOW_KEY, verbose=False)
    cand = [g for g in KO_CANDIDATES if g in ad.var_names]
    bias_dict, _ = sch.dyn.run_ko_screen(ad, genes=cand, lineage_A_clusters=ERY,
        lineage_B_clusters=MYE, basis=BASIS, wt_flow_key=WT_FLOW_KEY,
        cluster_key=CLUSTER_KEY, verbose=False)
    pert = sorted(cand, key=lambda g: abs(bias_dict[g]["lineage_bias"]), reverse=True)
    top_pert = [(g, round(bias_dict[g]["lineage_bias"], 4)) for g in pert[:TOPN]]
    ns = {"top_score_ery": top_score_ery, "top_score_mye": top_score_mye, "top_pert": top_pert}

    # compare against the 6 scaffold settings
    scaf = json.load(open(f"{OUTDIR}/results.json"))
    ns_pert = [g for g, _ in top_pert]
    print("\n=== no-scaffold top perturbation genes ===", flush=True)
    print(" ", ns_pert[:10], flush=True)
    print("=== no-scaffold top score(ery) ===", flush=True)
    print(" ", top_score_ery[:10], flush=True)

    comp = {"no_scaffold": ns, "jaccard_vs_scaffold": {}}
    print("\n=== Jaccard: no-scaffold vs each scaffold setting ===", flush=True)
    for tag, v in scaf.items():
        j_pert = jaccard(ns_pert, [g for g, _ in v["top_pert"]])
        j_sery = jaccard(top_score_ery, v["top_score_ery"])
        j_smye = jaccard(top_score_mye, v["top_score_mye"])
        comp["jaccard_vs_scaffold"][tag] = {"pert": round(j_pert, 3),
                                            "score_ery": round(j_sery, 3), "score_mye": round(j_smye, 3)}
        print(f"  {tag:22s} pert={j_pert:.2f}  score_ery={j_sery:.2f}  score_mye={j_smye:.2f}", flush=True)

    # mean over settings
    for key in ["pert", "score_ery", "score_mye"]:
        vals = [comp["jaccard_vs_scaffold"][t][key] for t in scaf]
        comp.setdefault("mean_jaccard_vs_scaffold", {})[key] = round(float(np.mean(vals)), 3)
    # consensus perturbation genes (in >=4 of 6 scaffold settings) and how many no-scaffold recovers
    from collections import Counter
    cnt = Counter(g for v in scaf.values() for g in [x for x, _ in v["top_pert"]][:10])
    consensus = [g for g, c in cnt.items() if c >= 4]
    comp["scaffold_consensus_pert"] = consensus
    comp["no_scaffold_recovers_consensus"] = sorted(set(ns_pert[:10]) & set(consensus))
    print(f"\nscaffold-consensus perturbation drivers (>=4/6): {consensus}", flush=True)
    print(f"no-scaffold recovers: {comp['no_scaffold_recovers_consensus']} "
          f"({len(comp['no_scaffold_recovers_consensus'])}/{len(consensus)})", flush=True)
    print(f"\nmean Jaccard vs scaffold: {comp['mean_jaccard_vs_scaffold']}", flush=True)

    os.makedirs(OUTDIR, exist_ok=True)
    json.dump(comp, open(f"{OUTDIR}/no_scaffold_compare.json", "w"), indent=2)
    print(f"wrote {OUTDIR}/no_scaffold_compare.json", flush=True)


if __name__ == "__main__":
    main()
