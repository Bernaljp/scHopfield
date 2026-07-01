"""CellOracle known-TF KO panel on Paul 2015, for a FAIR head-to-head vs scHopfield.

Runs in the CellOracle env (.venv-co). For each TF in the same panel, runs
CellOracle's native in-silico KO (simulate_shift -> transition prob -> embedding
shift) and scores the DIRECTION of the predicted lineage shift with the SAME
metric applied to scHopfield: project the mean per-cell embedding shift onto the
erythroid-minus-myeloid axis in the draw_graph_fa embedding.

  direction(+) = shift toward erythroid pole ; direction(-) = toward myeloid pole

Expected: KO of an erythroid master -> shift AWAY from erythroid (toward myeloid,
sign -1); KO of a myeloid master -> toward erythroid (sign +1). Directional
accuracy = fraction correct. Same panel + same geometry as the scHopfield score,
so the two accuracies are directly comparable (a fair, ground-truth-anchored
comparison, not a scale-confounded magnitude table).
"""
import json
import os

import numpy as np

import _co_shim  # noqa: F401  (sibling module; must precede celloracle import)
import celloracle as co  # noqa: E402

CLUSTER_KEY = "paul15_clusters"
BASIS = "draw_graph_fa"
ERY = ["1Ery", "2Ery", "3Ery", "4Ery", "5Ery", "6Ery", "7MEP"]
MYE = ["9GMP", "10GMP", "11DC", "12Baso", "13Baso", "14Mo", "15Mo", "16Neu", "17Neu", "18Eos"]
PANEL = {"Gata1": -1, "Klf1": -1, "Zfpm1": -1, "Tal1": -1, "Nfe2": -1, "Gata2": -1,
         "Spi1": +1, "Cebpa": +1, "Cebpe": +1, "Gfi1": +1, "Irf8": +1}


def main():
    oracle = co.data.load_tutorial_oracle_object()
    # fit the simulation GRN (the tutorial object ships the TFdict but not the fitted coefs)
    oracle.fit_GRN_for_simulation(alpha=10, use_cluster_specific_TFdict=True, verbose_level=0)
    adata = oracle.adata
    emb = adata.obsm[f"X_{BASIS}"]
    cl = adata.obs[CLUSTER_KEY].astype(str).values
    ery_centroid = emb[np.isin(cl, ERY)].mean(0)
    mye_centroid = emb[np.isin(cl, MYE)].mean(0)
    axis = ery_centroid - mye_centroid
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    print(f"ery-vs-mye axis in {BASIS}: {axis}", flush=True)

    genes = [g for g in PANEL if g in adata.var_names]
    missing = [g for g in PANEL if g not in adata.var_names]
    print(f"panel present: {genes}\nmissing: {missing}", flush=True)

    rows = []
    for g in genes:
        try:
            oracle.simulate_shift(perturb_condition={g: 0.0}, n_propagation=3)
            oracle.estimate_transition_prob(n_neighbors=200, knn_random=True, sampled_fraction=1)
            oracle.calculate_embedding_shift(sigma_corr=0.05)
            delta = oracle.delta_embedding  # (n_cells, 2)
            proj = float(np.mean(delta @ axis))   # net movement along ery(+)/mye(-) axis
            expected = PANEL[g]
            pred_sign = int(np.sign(proj)) if proj != 0 else 0
            correct = (pred_sign == expected)
            rows.append({"gene": g, "axis_projection": round(proj, 6),
                         "expected_sign": expected, "pred_sign": pred_sign, "correct": bool(correct),
                         "role": "erythroid-master" if expected < 0 else "myeloid-master"})
            print(f"  {g:7s} proj={proj:+.5f} expect={'ery-block(-)' if expected<0 else 'mye-block(+)'} "
                  f"-> {'OK' if correct else 'MISS'}", flush=True)
        except Exception as e:
            print(f"  {g}: FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
            rows.append({"gene": g, "error": str(e)[:120], "correct": False})

    scored = [r for r in rows if "error" not in r]
    n = len(scored)
    acc = sum(r["correct"] for r in scored) / n if n else float("nan")
    out = {"n": n, "directional_accuracy": acc, "metric": "mean embedding-shift . (ery-mye) axis",
           "rows": rows, "ery_clusters": ERY, "mye_clusters": MYE}
    os.makedirs("benchmark_results/hemato_ko", exist_ok=True)
    json.dump(out, open("benchmark_results/hemato_ko/celloracle_ko_panel.json", "w"), indent=2)
    print(f"\nCellOracle directional accuracy: {acc:.2f} ({sum(r['correct'] for r in scored)}/{n})", flush=True)
    print("wrote benchmark_results/hemato_ko/celloracle_ko_panel.json", flush=True)


if __name__ == "__main__":
    main()
