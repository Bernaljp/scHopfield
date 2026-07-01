"""Known-TF KO panel for scHopfield on Paul 2015 hematopoiesis.

Scores whether scHopfield's predicted lineage bias for single-gene KOs of
literature-established master regulators has the correct DIRECTION:

  bias = score_erythroid - score_myeloid  (+ = erythroid-biasing, - = myeloid-biasing)

A KO of an erythroid master (Gata1, Klf1, ...) should REMOVE erythroid drive
-> bias < 0. A KO of a myeloid master (Spi1/PU.1, Cebpa, ...) should remove
myeloid drive -> bias > 0. Directional accuracy = fraction of the panel whose
KO-bias sign matches expectation. This is the CellOracle validation protocol
(known KO phenotype recovery), applied to scHopfield and reported as a proper
ground-truth-anchored metric (not a scale-confounded magnitude table).
"""
import json
import os

import numpy as np
import anndata as ad
import scHopfield as sch

CLUSTER_KEY = "paul15_clusters"
BASIS = "draw_graph_fa"
WT_FLOW_KEY = f"original_velocity_flow_{BASIS}"

ERY = ["1Ery", "2Ery", "3Ery", "4Ery", "5Ery", "6Ery", "7MEP"]
MYE = ["9GMP", "10GMP", "11DC", "12Baso", "13Baso", "14Mo", "15Mo", "16Neu", "17Neu", "18Eos"]

# Literature-established direction of KO effect. +1 = KO should be erythroid-biasing
# (bias>0, a myeloid master); -1 = KO should be myeloid-biasing (bias<0, erythroid master).
PANEL = {
    # erythroid / megakaryocyte masters -> KO removes erythroid drive -> bias < 0
    "Gata1": -1, "Klf1": -1, "Zfpm1": -1, "Tal1": -1, "Nfe2": -1, "Gata2": -1,
    # myeloid masters -> KO removes myeloid drive -> bias > 0
    "Spi1": +1, "Cebpa": +1, "Cebpe": +1, "Gfi1": +1, "Irf8": +1,
}


def main():
    adata = ad.read_h5ad("data/hematopoiesis/adata_schopfield.h5ad")
    genes = [g for g in PANEL if g in adata.var_names]
    missing = [g for g in PANEL if g not in adata.var_names]
    print(f"panel present: {genes}\nmissing: {missing}", flush=True)

    # WT Hopfield velocity flow in embedding (reference for lineage bias)
    sch.tl.calculate_flow(adata, source="original", basis=BASIS, method="hopfield",
                          cluster_key=CLUSTER_KEY, store_key=WT_FLOW_KEY, verbose=False)
    print("WT flow computed", flush=True)

    bias_dict, _effects = sch.dyn.run_ko_screen(
        adata, genes=genes, lineage_A_clusters=ERY, lineage_B_clusters=MYE,
        basis=BASIS, wt_flow_key=WT_FLOW_KEY, cluster_key=CLUSTER_KEY, verbose=True,
    )

    rows = []
    for g in genes:
        bias = float(bias_dict[g]["lineage_bias"])
        expected = PANEL[g]
        pred_sign = int(np.sign(bias)) if bias != 0 else 0
        correct = (pred_sign == expected)
        rows.append({"gene": g, "lineage_bias": round(bias, 4),
                     "expected_sign": expected, "pred_sign": pred_sign, "correct": bool(correct),
                     "role": "erythroid-master" if expected < 0 else "myeloid-master"})
        print(f"  {g:7s} bias={bias:+.4f} expect={'ery-block(-)' if expected<0 else 'mye-block(+)'} "
              f"-> {'OK' if correct else 'MISS'}", flush=True)

    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n if n else float("nan")
    out = {"n": n, "directional_accuracy": acc, "rows": rows,
           "ery_clusters": ERY, "mye_clusters": MYE, "seed": int(adata.uns["scHopfield"].get("seed", -1))}
    os.makedirs("benchmark_results/hemato_ko", exist_ok=True)
    json.dump(out, open("benchmark_results/hemato_ko/schopfield_ko_panel.json", "w"), indent=2)
    print(f"\nscHopfield directional accuracy: {acc:.2f} ({sum(r['correct'] for r in rows)}/{n})", flush=True)
    print("wrote benchmark_results/hemato_ko/schopfield_ko_panel.json", flush=True)


if __name__ == "__main__":
    main()
