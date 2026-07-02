"""Second known-driver KO validation (#2): murine neural crest (glia vs neuron).

Scaffold-guided scHopfield fit (CellOracle mouse base GRN) on the neural-crest data,
then in-silico single-KO of literature glia/Schwann vs neuronal regulators, scored for
the DIRECTION of the predicted lineage shift. bias = score_glia - score_neuron:
a glia-driver KO should bias toward neuron (negative), a neuronal-driver KO toward glia
(positive). Directional accuracy = fraction correct. Generalizes the M4 hematopoiesis
validation to a second developmental system.
"""
import json
import os

import numpy as np
import pandas as pd
import anndata as ad
import scHopfield as sch

CK = "celltype_update"
BASIS = "umap"
WT_FLOW = f"original_velocity_flow_{BASIS}"
GLIA = ["Neural crest (PNS glia)", "Myelinating Schwann cells",
        "Myelinating Schwann cells (Tgfb2+)", "Olfactory ensheathing cells"]
NEUR = ["Neural crest (PNS neurons)", "Otic sensory neurons",
        "Dorsal root ganglion neurons", "Enteric neurons"]
# +1 = neuronal master (KO -> glia bias, positive); -1 = glia master (KO -> neuron bias, negative)
PANEL = {"Sox10": -1, "Erbb3": -1, "Mpz": -1, "Plp1": -1,
         "Neurod1": +1, "Isl1": +1, "Pou4f1": +1, "Elavl4": +1}
N_GENES = 600


def main():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a = ad.read_h5ad("data/generalize/murine_nc.h5ad")
    # gene subset: top-velocity UNION panel
    vmag = np.abs(a.layers["velocity_S"]).mean(0)
    top = set(np.argsort(np.asarray(vmag).ravel())[::-1][:N_GENES].tolist())
    for g in PANEL:
        if g in a.var_names:
            top.add(a.var_names.get_loc(g))
    a = a[:, sorted(top)].copy()
    a.var["scHopfield_used"] = True
    if "sigmoid" not in a.layers:
        sch.pp.fit_all_sigmoids(a, genes=a.var["scHopfield_used"].values)
        sch.pp.compute_sigmoid(a)
    print(f"neural crest {a.shape}; device={dev}", flush=True)

    base = pd.read_parquet("data/hematopoiesis/networks/mouse_scATAC_atlas.parquet")
    scaffold, ntf, nedge = sch.inf.build_scaffold(a, base, return_stats=True)
    print(f"scaffold: {ntf} TFs, {nedge} edges", flush=True)

    sch.inf.fit_interactions(
        a, cluster_key=CK, w_scaffold=scaffold.values.T, scaffold_regularization=0.1,
        reconstruction_regularization=100, bias_regularization=1, only_TFs=True,
        refit_gamma=True, w_threshold=1e-12, skip_all=True, n_epochs=600, batch_size=128,
        device=dev, learning_rate=0.1, use_plateau_scheduler=True, plateau_patience=100,
        plateau_factor=0.1, drop_last=True, include_neighbors=True, neighbor_fraction=0.2, seed=0)
    print("GRN inference done", flush=True)

    sch.tl.calculate_flow(a, source="original", basis=BASIS, method="hopfield",
                          cluster_key=CK, store_key=WT_FLOW, verbose=False)
    # Directional KO scoring promoted to the package: sch.dyn.score_ko_panel.
    table, acc = sch.dyn.score_ko_panel(
        a, panel=PANEL, lineage_A_clusters=GLIA, lineage_B_clusters=NEUR,
        basis=BASIS, wt_flow_key=WT_FLOW, cluster_key=CK, verbose=True)
    rows = table.to_dict("records")
    for r in rows:
        r["role"] = "glia-master" if r["expected_sign"] < 0 else "neuronal-master"
    os.makedirs("benchmark_results/nc_ko", exist_ok=True)
    json.dump({"directional_accuracy": acc, "n": len(rows), "rows": rows,
               "glia": GLIA, "neuron": NEUR}, open("benchmark_results/nc_ko/panel.json", "w"), indent=2)
    print(f"\nneural-crest directional accuracy: {acc:.2f} "
          f"({int(table['correct'].sum())}/{len(rows)})", flush=True)


if __name__ == "__main__":
    main()
