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

    # Directional KO scoring promoted to the package: sch.dyn.score_ko_panel.
    table, acc = sch.dyn.score_ko_panel(
        adata, panel=PANEL, lineage_A_clusters=ERY, lineage_B_clusters=MYE,
        basis=BASIS, wt_flow_key=WT_FLOW_KEY, cluster_key=CLUSTER_KEY, verbose=True,
    )
    rows = table.to_dict("records")
    for r in rows:
        r["role"] = "erythroid-master" if r["expected_sign"] < 0 else "myeloid-master"

    n = len(rows)
    out = {"n": n, "directional_accuracy": acc, "rows": rows,
           "ery_clusters": ERY, "mye_clusters": MYE, "seed": int(adata.uns["scHopfield"].get("seed", -1))}
    os.makedirs("benchmark_results/hemato_ko", exist_ok=True)
    json.dump(out, open("benchmark_results/hemato_ko/schopfield_ko_panel.json", "w"), indent=2)
    print(f"\nscHopfield directional accuracy: {acc:.2f} ({int(table['correct'].sum())}/{n})", flush=True)
    print("wrote benchmark_results/hemato_ko/schopfield_ko_panel.json", flush=True)


if __name__ == "__main__":
    main()
