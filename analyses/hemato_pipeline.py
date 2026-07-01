"""scHopfield hematopoiesis pipeline (Paul 2015), reproduced from notebook 05.

Runs in the scHopfield env (.venv) from the CellOracle-exported files
(data/hematopoiesis/paul15_oracle.h5ad + base_GRN.parquet), so it does not need
CellOracle. Pseudotime velocity -> sigmoid fit -> seeded scaffold-guided GRN
inference. Saves a fitted-adata checkpoint for the KO panel + notebook refresh.
"""
import os
import numpy as np
import pandas as pd
import scanpy as sc  # noqa
import scvelo as scv
import torch

import scHopfield as sch

CLUSTER_KEY = "paul15_clusters"
SPLICED_KEY = "Ms"
VELOCITY_KEY = "velocity_S"
DEGRADATION_KEY = "gamma"
VELOCITY_SCALE = 500.0
SCAFFOLD_REG = 1e-1
N_EPOCHS = 1000
BATCH_SIZE = 128
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "data/hematopoiesis/adata_schopfield.h5ad"


def build_scaffold(adata, base_GRN):
    gene_names = adata.var.index[adata.var["scHopfield_used"].values]
    scaffold = pd.DataFrame(0, index=gene_names, columns=gene_names)
    tfs = list(set(base_GRN.columns.str.lower()) & set(scaffold.index.str.lower()))
    index_map = {g.lower(): g for g in scaffold.index}
    col_map = {g.lower(): g for g in scaffold.columns}
    for tf in tfs:
        tf_col = [c for c in base_GRN.columns if c.lower() == tf][0]
        for tgt in base_GRN[base_GRN[tf_col] == 1]["gene_short_name"]:
            if tgt.lower() in col_map:
                scaffold.loc[index_map[tf], col_map[tgt.lower()]] = 1
    print(f"Scaffold: {len(tfs)} TFs, {int(scaffold.sum().sum())} edges", flush=True)
    return scaffold


def main():
    print(f"device={DEVICE}", flush=True)
    adata = sc.read_h5ad("data/hematopoiesis/paul15_oracle.h5ad")
    base_GRN = pd.read_parquet("data/hematopoiesis/base_GRN.parquet")
    base_GRN.drop(["peak_id"], axis=1, inplace=True)
    print(f"adata {adata.shape}; clusters={adata.obs[CLUSTER_KEY].nunique()}", flush=True)

    # Ms via scVelo moments (spliced == unspliced == normalized_count, per nb05)
    adata.layers["spliced"] = adata.layers["normalized_count"]
    adata.layers["unspliced"] = adata.layers["normalized_count"]
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    del adata.layers["unspliced"]

    # constant degradation (nb05: gamma = 0.1, refit during inference)
    adata.var[DEGRADATION_KEY] = 0.1

    # pseudotime velocity
    sch.pp.estimate_velocity_from_pseudotime(
        adata, pseudotime_key="Pseudotime", spliced_key=SPLICED_KEY,
        connectivity_key="connectivities", scale=VELOCITY_SCALE, store_key=VELOCITY_KEY,
    )
    print("velocity estimated", flush=True)

    # sigmoids on all genes
    adata.var["scHopfield_used"] = True
    sch.pp.fit_all_sigmoids(adata, genes=adata.var["scHopfield_used"].values, spliced_key=SPLICED_KEY)
    sch.pp.compute_sigmoid(adata, spliced_key=SPLICED_KEY)
    mse = adata.var.loc[adata.var["scHopfield_used"], "sigmoid_mse"]
    print(f"sigmoid MSE mean={mse.mean():.4f} median={mse.median():.4f}", flush=True)

    scaffold = build_scaffold(adata, base_GRN)

    # seeded scaffold-guided GRN inference (nb05 params + seed)
    sch.inf.fit_interactions(
        adata, cluster_key=CLUSTER_KEY, spliced_key=SPLICED_KEY, velocity_key=VELOCITY_KEY,
        degradation_key=DEGRADATION_KEY, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
        refit_gamma=True, w_scaffold=scaffold.values.T, scaffold_regularization=SCAFFOLD_REG,
        reconstruction_regularization=100, bias_regularization=1, only_TFs=True,
        w_threshold=1e-12, skip_all=True, learning_rate=0.1, use_plateau_scheduler=True,
        plateau_patience=100, plateau_factor=0.1, balanced_sampling=True, drop_last=True,
        include_neighbors=True, neighbor_fraction=0.2, get_plots=False, seed=SEED,
    )
    print("GRN inference complete", flush=True)

    # drop the trained torch models before writing (not h5ad-serializable)
    adata.uns.get("scHopfield", {}).pop("models", None)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    adata.write(OUT)
    print(f"wrote {OUT}: {adata.shape}", flush=True)


if __name__ == "__main__":
    main()
