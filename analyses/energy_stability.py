"""General energy-landscape + Jacobian-stability analysis (Fig-5 style) for any
scHopfield-ready dataset. Fits a cell-type-specific pseudoinverse GRN, computes
energies + per-cell Jacobian eigenvalue stats, and plots stability + energy by
cell type."""
import argparse
import json
import os

import numpy as np
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scHopfield as sch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cluster-key", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n-genes", type=int, default=250)
    ap.add_argument("--out", default="benchmark_results/energy_stability")
    args = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    a = ad.read_h5ad(args.data)
    vmag = np.abs(a.layers["velocity_S"]).mean(0)
    keep = np.argsort(np.asarray(vmag).ravel())[::-1][:args.n_genes]
    a = a[:, keep].copy()
    a.var["scHopfield_used"] = True
    if "sigmoid" not in a.layers:
        sch.pp.fit_all_sigmoids(a, genes=a.var["scHopfield_used"].values)
        sch.pp.compute_sigmoid(a)
    print(f"{args.name}: {a.shape} device={dev}", flush=True)

    sch.inf.fit_interactions(a, cluster_key=args.cluster_key, w_scaffold=None,
                             skip_all=True, w_threshold=1e-12, seed=0)
    sch.tl.compute_energies(a, cluster_key=args.cluster_key)
    sch.tl.compute_jacobians(a, cluster_key=args.cluster_key, device=dev)
    sch.tl.compute_jacobian_stats(a)
    print("energy + jacobian done", flush=True)

    obs = a.obs
    order = list(obs[args.cluster_key].astype(str).value_counts().index)
    present = [c for c in order if (obs[args.cluster_key].astype(str) == c).sum() >= 20]
    stats = {}
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    lab = obs[args.cluster_key].astype(str)
    for ax, col, title, yl in [
        (axes[0], "jacobian_eig1_real", "Local stability (leading eig, Re)", "leading Jacobian eig (Re)"),
        (axes[1], "jacobian_positive_evals", "Instability count", "# positive real eigenvalues"),
        (axes[2], "energy_total", "Energy landscape depth", "total Hopfield energy")]:
        if col in obs:
            data = [obs.loc[lab == c, col].values for c in present]
            ax.boxplot(data, showfliers=False)
            ax.set_xticklabels(present, rotation=40, ha="right", fontsize=7)
            ax.set_ylabel(yl); ax.set_title(title)
            if col == "jacobian_eig1_real":
                ax.axhline(0, color="crimson", lw=0.8, ls="--")
    for c in present:
        m = lab == c
        stats[c] = {k: float(np.nanmedian(obs.loc[m, k].values))
                    for k in ["jacobian_eig1_real", "jacobian_positive_evals", "energy_total"] if k in obs}
    fig.suptitle(f"scHopfield energy + stability: {args.name}", y=1.03, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{args.out}/{args.name}.png", dpi=140, bbox_inches="tight")
    json.dump(stats, open(f"{args.out}/{args.name}.json", "w"), indent=2)
    print(f"wrote {args.out}/{args.name}.png", flush=True)
    for c in present:
        s = stats[c]
        print(f"  {c[:26]:26s} eig1={s.get('jacobian_eig1_real', float('nan')):+.3f} "
              f"pos={s.get('jacobian_positive_evals', float('nan')):.1f} "
              f"E={s.get('energy_total', float('nan')):.1f}", flush=True)


if __name__ == "__main__":
    main()
