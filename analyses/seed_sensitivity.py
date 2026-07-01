"""Seed-sensitivity experiment for scHopfield GRN inference.

Purpose
-------
Quantify the reproducibility of the scaffold-guided GRN inference path, which is
the stochastic part of scHopfield (random weight init + Adam over shuffled
mini-batches). We demonstrate:

  1. UNSEEDED: two independent fits differ (the bug the collaborators hit).
  2. SEEDED (same seed): two fits are identical on CPU and quantify residual
     CUDA nondeterminism on GPU.
  3. CROSS-SEED: variance across 5 different seeds bounds how much the inferred
     network depends on the random draw.

Metrics per run pair: Pearson correlation and relative Frobenius distance of the
flattened interaction matrix W_all; Spearman rank correlation of per-gene
out-strength centrality (sum_j |W_ij|); and, when available, correlation of the
per-cell total Hopfield energy.

Usage
-----
    .venv/bin/python analyses/seed_sensitivity.py \
        --data data/Pancreas/endocrinogenesis_day15.h5ad \
        --n-genes 300 --n-epochs 300 --device cuda --out benchmark_results/seed_sensitivity

Falls back to a controlled synthetic system if the dataset lacks the required
layers, so the determinism machinery is always exercised.
"""
from __future__ import annotations

import argparse
import json
import os
from itertools import combinations

import numpy as np


def log(msg: str) -> None:
    print(msg, flush=True)


def build_synthetic_adata(n_cells=300, n_genes=200, noise=0.15, seed=0):
    """A controlled Hopfield-like dataset so the inference path always runs.

    Defaults to an UNDER-DETERMINED regime (n_genes > n_cells/type) that mirrors
    real scHopfield usage: hundreds of genes, a few hundred cells per cell type.
    In this regime the fit has many near-equivalent W minima, so unseeded runs
    diverge while seeded runs are reproducible.
    """
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(seed)
    # ground-truth sparse signed network
    W = rng.normal(0, 1, (n_genes, n_genes))
    mask = rng.random((n_genes, n_genes)) < 0.1
    W = W * mask
    gamma = rng.uniform(0.5, 1.5, n_genes).astype(np.float32)
    # expression drawn positive; velocity from the Hopfield rhs with noise
    X = rng.gamma(2.0, 1.0, (n_cells, n_genes)).astype(np.float32)
    k = np.median(X, axis=0) + 1e-3
    n_hill = 4.0
    sig = X**n_hill / (X**n_hill + k**n_hill)
    V = (sig @ W.T - gamma[None, :] * X).astype(np.float32)
    V += rng.normal(0, noise, V.shape).astype(np.float32)

    adata = ad.AnnData(X=X)
    adata.layers["Ms"] = X
    adata.layers["velocity_S"] = V
    adata.var["gamma"] = gamma
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    adata.obs["cell_type"] = pd.Categorical(
        rng.choice(["A", "B"], size=n_cells)
    )
    adata.uns["_synthetic_W_true"] = W
    return adata


def load_or_synth(path, n_genes, n_cells_cap):
    import anndata as ad

    if path and os.path.exists(path):
        adata = ad.read_h5ad(path)
        log(f"Loaded {path}: {adata.shape}")
        log(f"  layers: {list(adata.layers.keys())}")
        log(f"  obs: {list(adata.obs.columns)[:12]}")
        log(f"  var: {list(adata.var.columns)[:12]}")
        have_ms = "Ms" in adata.layers
        have_vel = "velocity_S" in adata.layers
        have_gamma = "gamma" in adata.var
        log(f"  Ms={have_ms} velocity_S={have_vel} gamma={have_gamma}")
        if have_ms and have_vel and have_gamma:
            # subset to n_genes with highest velocity magnitude, cap cells
            vmag = np.asarray(np.abs(adata.layers["velocity_S"]).mean(0)).ravel()
            keep = np.argsort(vmag)[::-1][:n_genes]
            adata = adata[:, keep].copy()
            if adata.n_obs > n_cells_cap:
                idx = np.random.RandomState(0).choice(
                    adata.n_obs, n_cells_cap, replace=False
                )
                adata = adata[idx].copy()
            log(f"  using real data subset: {adata.shape}")
            return adata, "real"
        log("  dataset missing required layers -> synthetic fallback")
    else:
        log(f"No dataset at {path} -> synthetic fallback")
    # under-determined synthetic: n_genes genes, ~150 cells/type (2 types)
    return build_synthetic_adata(n_cells=300, n_genes=min(n_genes, 200)), "synthetic"


def fit_W(adata_in, scaffold, seed, n_epochs, device, cluster_key):
    """Fit interactions on a copy; return (W_all, I_all)."""
    import scHopfield as sch

    adata = adata_in.copy()
    # ensure sigmoid params + layer exist
    genes = list(range(adata.n_vars))
    sch.pp.fit_all_sigmoids(adata, genes=genes)
    sch.pp.compute_sigmoid(adata)
    sch.inf.fit_interactions(
        adata,
        cluster_key=cluster_key,
        w_scaffold=scaffold,
        n_epochs=n_epochs,
        device=device,
        seed=seed,
        skip_all=False,
    )
    W = np.asarray(adata.varp["W_all"], dtype=np.float64)
    I = np.asarray(adata.var["I_all"].values, dtype=np.float64)
    energy = None
    try:
        import scHopfield as sch2  # noqa
        sch.tl.compute_energies(adata, cluster_key=cluster_key)
        energy = np.asarray(adata.obs["energy_total"].values, dtype=np.float64)
    except Exception as e:  # energy is a nice-to-have
        log(f"    (energy skipped: {e})")
    return W, I, energy


def pair_metrics(Wa, Wb):
    a, b = Wa.ravel(), Wb.ravel()
    pear = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
    rel_fro = float(np.linalg.norm(Wa - Wb) / (np.linalg.norm(Wa) + 1e-12))
    # out-strength centrality rank stability
    ca = np.abs(Wa).sum(1)
    cb = np.abs(Wb).sum(1)
    from scipy.stats import spearmanr
    sp = float(spearmanr(ca, cb).correlation)
    return {"W_pearson": pear, "W_rel_frobenius": rel_fro, "centrality_spearman": sp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/Pancreas/endocrinogenesis_day15.h5ad")
    ap.add_argument("--n-genes", type=int, default=300)
    ap.add_argument("--n-cells-cap", type=int, default=2000)
    ap.add_argument("--n-epochs", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="benchmark_results/seed_sensitivity")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    import torch

    dev = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    log(f"device requested={args.device} -> using {dev}; cuda_available={torch.cuda.is_available()}")

    adata, kind = load_or_synth(args.data, args.n_genes, args.n_cells_cap)
    cluster_key = "cell_type" if "cell_type" in adata.obs else adata.obs.columns[0]
    N = adata.n_vars
    scaffold = np.ones((N, N), dtype=np.float32)  # free (all-edges) scaffold: exercises stochastic path

    results = {"dataset": kind, "shape": list(adata.shape), "device": dev,
               "n_epochs": args.n_epochs, "runs": {}, "pairs": {}}

    def run(tag, seed):
        log(f"[fit] {tag} seed={seed}")
        W, I, E = fit_W(adata, scaffold, seed, args.n_epochs, dev, cluster_key)
        results["runs"][tag] = {"seed": seed,
                                "W_sparsity": float((W == 0).mean()),
                                "W_absmean": float(np.abs(W).mean())}
        return W, I, E

    # 1. unseeded twice
    Wu1, _, Eu1 = run("unseeded_a", None)
    Wu2, _, Eu2 = run("unseeded_b", None)
    results["pairs"]["unseeded_a_vs_b"] = pair_metrics(Wu1, Wu2)

    # 2. same-seed twice (determinism check)
    Ws1, _, Es1 = run("seed0_a", 0)
    Ws2, _, Es2 = run("seed0_b", 0)
    results["pairs"]["seed0_a_vs_b"] = pair_metrics(Ws1, Ws2)

    # 3. cross-seed variance
    cross = {}
    Ws = {0: Ws1}
    for s in [x for x in args.seeds if x != 0]:
        Ws[s], _, _ = run(f"seed{s}", s)
    for s1, s2 in combinations(sorted(Ws), 2):
        cross[f"{s1}_vs_{s2}"] = pair_metrics(Ws[s1], Ws[s2])
    results["pairs"]["cross_seed"] = cross
    # summary of cross-seed
    pear = [v["W_pearson"] for v in cross.values()]
    results["cross_seed_W_pearson_mean"] = float(np.nanmean(pear))
    results["cross_seed_W_pearson_std"] = float(np.nanstd(pear))

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log("\n=== SUMMARY ===")
    log(f"dataset={kind} {adata.shape} device={dev}")
    log(f"unseeded a-vs-b  W_pearson={results['pairs']['unseeded_a_vs_b']['W_pearson']:.4f}")
    log(f"seed0    a-vs-b  W_pearson={results['pairs']['seed0_a_vs_b']['W_pearson']:.6f} (want ~1.0)")
    log(f"cross-seed W_pearson={results['cross_seed_W_pearson_mean']:.4f} +/- {results['cross_seed_W_pearson_std']:.4f}")
    log(f"wrote {args.out}/results.json")


if __name__ == "__main__":
    main()
