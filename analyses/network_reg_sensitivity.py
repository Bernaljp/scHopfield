"""Sensitivity of driver-score & perturbation top genes to network + scaffold reg.

Paul 2015. For CellOracle's two mouse base GRNs (scATAC atlas, promoter) x three
scaffold-regularization regimes (free=0, low elastic-net, high elastic-net), fit
scHopfield and report:
  (1) top driver-SCORE genes    -- score_driver_tfs (per lineage)
  (2) top PERTURBATION genes    -- KO a fixed candidate set, rank by |lineage_bias|
then quantify how stable the top-gene lists are across the 6 settings (Jaccard).

Preprocessing (velocity, sigmoid) is done once and reused; only fit_interactions
+ scoring + KO vary per setting. Seeded for reproducibility.
"""
import json
import os
import itertools

import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import torch

import scHopfield as sch

CLUSTER_KEY = "paul15_clusters"
SPLICED_KEY, VELOCITY_KEY, DEG_KEY = "Ms", "velocity_S", "gamma"
BASIS = "draw_graph_fa"
WT_FLOW_KEY = f"original_velocity_flow_{BASIS}"
VELOCITY_SCALE = 500.0
N_EPOCHS = 600
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTDIR = "benchmark_results/network_reg_sensitivity"

ERY = ["1Ery", "2Ery", "3Ery", "4Ery", "5Ery", "6Ery", "7MEP"]
MYE = ["9GMP", "10GMP", "11DC", "12Baso", "13Baso", "14Mo", "15Mo", "16Neu", "17Neu", "18Eos"]
REGS = {"free": 0.0, "low": 0.01, "high": 1.0}
NETWORKS = {"scATAC_atlas": "data/hematopoiesis/networks/mouse_scATAC_atlas.parquet",
            "promoter": "data/hematopoiesis/networks/mouse_promoter.parquet"}
# fixed candidate TF set for KO ranking (comparable across settings); kept if present
KO_CANDIDATES = ["Gata1", "Gata2", "Spi1", "Klf1", "Cebpa", "Cebpe", "Cebpb", "Runx1",
                 "Fli1", "Zfpm1", "Tal1", "Nfe2", "Gfi1", "Gfi1b", "Irf8", "Myb", "Myc",
                 "Ikzf1", "Bcl11a", "Meis1", "Lmo2", "Lyl1", "Sox4", "Mef2c", "Nfia",
                 "Stat1", "Stat3", "Zbtb7a", "Ezh2", "E2f4"]
TOPN = 15


def build_scaffold(adata, base_GRN):
    gene_names = adata.var.index[adata.var["scHopfield_used"].values]
    scaffold = pd.DataFrame(0, index=gene_names, columns=gene_names)
    tfs = list(set(base_GRN.columns.str.lower()) & set(scaffold.index.str.lower()))
    imap = {g.lower(): g for g in scaffold.index}
    cmap = {g.lower(): g for g in scaffold.columns}
    for tf in tfs:
        tf_col = [c for c in base_GRN.columns if c.lower() == tf][0]
        for tgt in base_GRN[base_GRN[tf_col] == 1]["gene_short_name"]:
            if tgt.lower() in cmap:
                scaffold.loc[imap[tf], cmap[tgt.lower()]] = 1
    return scaffold, len(tfs), int(scaffold.sum().sum())


def preprocess():
    adata = sc.read_h5ad("data/hematopoiesis/paul15_oracle.h5ad")
    adata.layers["spliced"] = adata.layers["normalized_count"]
    adata.layers["unspliced"] = adata.layers["normalized_count"]
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    del adata.layers["unspliced"]
    adata.var[DEG_KEY] = 0.1
    sch.pp.estimate_velocity_from_pseudotime(adata, pseudotime_key="Pseudotime",
        spliced_key=SPLICED_KEY, connectivity_key="connectivities",
        scale=VELOCITY_SCALE, store_key=VELOCITY_KEY)
    adata.var["scHopfield_used"] = True
    sch.pp.fit_all_sigmoids(adata, genes=adata.var["scHopfield_used"].values, spliced_key=SPLICED_KEY)
    sch.pp.compute_sigmoid(adata, spliced_key=SPLICED_KEY)
    return adata


def run_setting(base, scaffold, reg, tag):
    ad = base.copy()
    sch.inf.fit_interactions(
        ad, cluster_key=CLUSTER_KEY, spliced_key=SPLICED_KEY, velocity_key=VELOCITY_KEY,
        degradation_key=DEG_KEY, n_epochs=N_EPOCHS, batch_size=128, device=DEVICE,
        refit_gamma=True, w_scaffold=scaffold.values.T, scaffold_regularization=reg,
        reconstruction_regularization=100, bias_regularization=1, only_TFs=True,
        w_threshold=1e-12, skip_all=True, learning_rate=0.1, use_plateau_scheduler=True,
        plateau_patience=100, plateau_factor=0.1, balanced_sampling=True, drop_last=True,
        include_neighbors=True, neighbor_fraction=0.2, get_plots=False, seed=SEED,
    )
    # (1) driver-score top genes
    scores = sch.tl.score_driver_tfs(ad, lineage_A_clusters=ERY, lineage_B_clusters=MYE,
                                     cluster_key=CLUSTER_KEY)
    top_score_ery = scores.sort_values("total_score_ery", ascending=False).head(TOPN).index.tolist()
    top_score_mye = scores.sort_values("total_score_mye", ascending=False).head(TOPN).index.tolist()
    # (2) perturbation top genes: KO fixed candidates, rank by |lineage_bias|
    sch.tl.calculate_flow(ad, source="original", basis=BASIS, method="hopfield",
                          cluster_key=CLUSTER_KEY, store_key=WT_FLOW_KEY, verbose=False)
    cand = [g for g in KO_CANDIDATES if g in ad.var_names]
    bias_dict, _ = sch.dyn.run_ko_screen(ad, genes=cand, lineage_A_clusters=ERY,
        lineage_B_clusters=MYE, basis=BASIS, wt_flow_key=WT_FLOW_KEY,
        cluster_key=CLUSTER_KEY, verbose=False)
    pert = sorted(cand, key=lambda g: abs(bias_dict[g]["lineage_bias"]), reverse=True)
    top_pert = [(g, round(bias_dict[g]["lineage_bias"], 4)) for g in pert[:TOPN]]
    return {"tag": tag, "reg": reg, "top_score_ery": top_score_ery,
            "top_score_mye": top_score_mye, "top_pert": top_pert}


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else float("nan")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"device={DEVICE}; preprocessing once...", flush=True)
    base = preprocess()
    results = {}
    for net_name, net_path in NETWORKS.items():
        base_GRN = pd.read_parquet(net_path).drop(columns=["peak_id"])
        scaffold, n_tf, n_edge = build_scaffold(base, base_GRN)
        print(f"[{net_name}] scaffold: {n_tf} TFs, {n_edge} edges", flush=True)
        for reg_name, reg in REGS.items():
            tag = f"{net_name}:{reg_name}"
            print(f"=== fitting {tag} (reg={reg}) ===", flush=True)
            res = run_setting(base, scaffold, reg, tag)
            results[tag] = res
            json.dump(results, open(f"{OUTDIR}/results.json", "w"), indent=2)
            print(f"  top score(ery): {res['top_score_ery'][:8]}", flush=True)
            print(f"  top pert:       {[g for g,_ in res['top_pert'][:8]]}", flush=True)

    # stability matrices (Jaccard of top lists across settings)
    tags = list(results)
    for key in ["top_score_ery", "top_score_mye"]:
        mat = {t1: {t2: round(jaccard(results[t1][key], results[t2][key]), 3) for t2 in tags} for t1 in tags}
        json.dump(mat, open(f"{OUTDIR}/jaccard_{key}.json", "w"), indent=2)
    pert_lists = {t: [g for g, _ in results[t]["top_pert"]] for t in tags}
    mat = {t1: {t2: round(jaccard(pert_lists[t1], pert_lists[t2]), 3) for t2 in tags} for t1 in tags}
    json.dump(mat, open(f"{OUTDIR}/jaccard_top_pert.json", "w"), indent=2)

    print("\n=== DONE; pairwise mean Jaccard ===", flush=True)
    for key, lab in [("top_score_ery", "score(ery)"), ("top_score_mye", "score(mye)")]:
        m = json.load(open(f"{OUTDIR}/jaccard_{key}.json"))
        off = [m[a][b] for a, b in itertools.combinations(tags, 2)]
        print(f"  {lab}: mean pairwise Jaccard = {np.mean(off):.3f}", flush=True)
    m = json.load(open(f"{OUTDIR}/jaccard_top_pert.json"))
    off = [m[a][b] for a, b in itertools.combinations(tags, 2)]
    print(f"  perturbation: mean pairwise Jaccard = {np.mean(off):.3f}", flush=True)
    print(f"wrote {OUTDIR}/results.json + jaccard_*.json", flush=True)


if __name__ == "__main__":
    main()
