"""scHopfield KO panel scored with the IDENTICAL metric used for CellOracle.

Closes the M4 metric-parity caveat: instead of run_ko_screen's native lineage
bias, score each KO by the mean per-cell embedding shift projected onto the
erythroid-minus-myeloid axis in draw_graph_fa, exactly as celloracle_ko_panel.py
does. Makes the head-to-head airtight (same panel, same geometry, same metric).
"""
import json
import os

import numpy as np
import anndata as ad
import scHopfield as sch

CK = "paul15_clusters"
BASIS = "draw_graph_fa"
ERY = ["1Ery", "2Ery", "3Ery", "4Ery", "5Ery", "6Ery", "7MEP"]
MYE = ["9GMP", "10GMP", "11DC", "12Baso", "13Baso", "14Mo", "15Mo", "16Neu", "17Neu", "18Eos"]
PANEL = {"Gata1": -1, "Klf1": -1, "Zfpm1": -1, "Nfe2": -1, "Gata2": -1,
         "Spi1": +1, "Cebpa": +1, "Cebpe": +1, "Gfi1": +1, "Irf8": +1}


def main():
    a = ad.read_h5ad("data/hematopoiesis/adata_schopfield.h5ad")
    emb = a.obsm[f"X_{BASIS}"]
    cl = a.obs[CK].astype(str).values
    axis = emb[np.isin(cl, ERY)].mean(0) - emb[np.isin(cl, MYE)].mean(0)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    genes = [g for g in PANEL if g in a.var_names]

    rows = []
    for g in genes:
        sch.dyn.simulate_shift_ode(a, {g: 0.0}, cluster_key=CK)  # -> layers['delta_X']
        flow = sch.tl.calculate_flow(a, source="delta", basis=BASIS, method="hopfield",
                                     cluster_key=CK, store_key=f"ko_flow_{g}", verbose=False)
        proj = float(np.nanmean(np.asarray(flow) @ axis))
        exp = PANEL[g]
        pred = int(np.sign(proj)) if proj != 0 else 0
        ok = (pred == exp)
        rows.append({"gene": g, "axis_projection": round(proj, 6), "expected_sign": exp,
                     "pred_sign": pred, "correct": bool(ok)})
        print(f"  {g:7s} proj={proj:+.5f} expect={'-' if exp<0 else '+'} -> {'OK' if ok else 'MISS'}", flush=True)

    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n if n else float("nan")
    out = {"n": n, "directional_accuracy": acc,
           "metric": "mean scHopfield KO embedding-shift . (ery-mye) axis (identical to CellOracle)",
           "rows": rows}
    os.makedirs("benchmark_results/hemato_ko", exist_ok=True)
    json.dump(out, open("benchmark_results/hemato_ko/schopfield_ko_axis.json", "w"), indent=2)
    print(f"\nscHopfield directional accuracy (axis metric): {acc:.2f} ({sum(r['correct'] for r in rows)}/{n})", flush=True)


if __name__ == "__main__":
    main()
