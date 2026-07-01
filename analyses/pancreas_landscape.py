"""Figure 5: pancreatic endocrinogenesis energy landscape + Jacobian stability.

Seeded scHopfield fit on the scVelo-preprocessed pancreas (Bastidas-Ponce 2019),
then per-cell Jacobian eigenvalues and energy decomposition, aggregated by cell
type. Reproduces the manuscript's Figure 5 claim that terminally differentiated
Beta cells occupy more stable dynamical regimes while Delta/Epsilon and Ngn3-high
progenitors are less stable / more heterogeneous.

Genes are subset to the top velocity-magnitude genes to keep per-cell
eigendecomposition tractable.
"""
import json
import os

import numpy as np
import anndata as ad
import scanpy as sc  # noqa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scHopfield as sch

CLUSTER_KEY = "clusters"
N_GENES = 300
SEED = 0
ORDER = ["Ductal", "Ngn3 low EP", "Ngn3 high EP", "Pre-endocrine", "Beta", "Alpha", "Delta", "Epsilon"]
OUT = "benchmark_results/pancreas"


def main():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)

    adata = ad.read_h5ad("data/Pancreas/pancreas_scvelo_ready.h5ad")
    vmag = np.abs(adata.layers["velocity_S"]).mean(0)
    keep = np.argsort(np.asarray(vmag).ravel())[::-1][:N_GENES]
    adata = adata[:, keep].copy()
    print(f"pancreas subset {adata.shape}; device={dev}", flush=True)

    adata.var["scHopfield_used"] = True
    sch.pp.fit_all_sigmoids(adata, genes=adata.var["scHopfield_used"].values)
    sch.pp.compute_sigmoid(adata)
    # cell-type-specific pseudoinverse fit (deterministic), per cluster
    sch.inf.fit_interactions(adata, cluster_key=CLUSTER_KEY, w_scaffold=None,
                             skip_all=True, w_threshold=1e-12, seed=SEED)
    print("GRN inference done", flush=True)

    sch.tl.compute_energies(adata, cluster_key=CLUSTER_KEY)
    sch.tl.compute_jacobians(adata, cluster_key=CLUSTER_KEY, device=dev)
    sch.tl.compute_jacobian_stats(adata)
    print("energy + jacobian done", flush=True)

    # per-cell stability summaries
    obs = adata.obs
    present = [c for c in ORDER if c in obs[CLUSTER_KEY].unique()]
    stats = {}
    for c in present:
        idx = (obs[CLUSTER_KEY] == c).values
        row = {}
        for col in ["jacobian_positive_evals", "jacobian_eig1_real", "jacobian_trace",
                    "energy_total", "energy_interaction"]:
            if col in obs:
                row[col] = float(np.nanmedian(obs.loc[idx, col].values))
        stats[c] = row
    json.dump(stats, open(f"{OUT}/pancreas_stability.json", "w"), indent=2)

    # ---- Figure 5 panels ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    # A: leading Jacobian eigenvalue (real part) distribution per cluster
    ax = axes[0]
    data = [obs.loc[obs[CLUSTER_KEY] == c, "jacobian_eig1_real"].values for c in present] \
        if "jacobian_eig1_real" in obs else []
    if data:
        ax.boxplot(data, showfliers=False)
        ax.set_xticklabels(present, rotation=40, ha="right", fontsize=8)
        ax.axhline(0, color="crimson", lw=0.8, ls="--")
        ax.set_ylabel("leading Jacobian eigenvalue (Re)")
        ax.set_title("Local stability by cell type\n(> 0 = unstable)")
    # B: count of positive eigenvalues per cluster
    ax = axes[1]
    if "jacobian_positive_evals" in obs:
        data = [obs.loc[obs[CLUSTER_KEY] == c, "jacobian_positive_evals"].values for c in present]
        ax.boxplot(data, showfliers=False)
        ax.set_xticklabels(present, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("# positive real eigenvalues")
        ax.set_title("Instability count by cell type")
    # C: total energy per cluster
    ax = axes[2]
    if "energy_total" in obs:
        data = [obs.loc[obs[CLUSTER_KEY] == c, "energy_total"].values for c in present]
        ax.boxplot(data, showfliers=False)
        ax.set_xticklabels(present, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("total Hopfield energy")
        ax.set_title("Energy landscape depth by cell type")
    fig.suptitle("scHopfield pancreatic endocrinogenesis: stability + energy landscape", y=1.03)
    fig.tight_layout()
    fig.savefig(f"{OUT}/pancreas_landscape.png", dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}/pancreas_landscape.png", flush=True)

    print("\n=== median stability by cell type ===", flush=True)
    for c in present:
        s = stats[c]
        print(f"  {c:14s} eig1_real={s.get('jacobian_eig1_real', float('nan')):+.4f} "
              f"pos_evals={s.get('jacobian_positive_evals', float('nan')):.1f} "
              f"E_total={s.get('energy_total', float('nan')):.2f}", flush=True)


if __name__ == "__main__":
    main()
