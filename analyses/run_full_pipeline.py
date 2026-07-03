"""Reproducible end-to-end scHopfield pipeline across multiple datasets.

Runs the SAME pipeline on every dataset and stores the fitted data and figures in
a uniform directory layout, so the whole method (preprocessing -> scaffold/GRN
fit -> energy -> Jacobian stability -> network drivers -> in-silico perturbation)
can be reproduced with one command:

    .venv/bin/python analyses/run_full_pipeline.py                 # all datasets
    .venv/bin/python analyses/run_full_pipeline.py --only pancreas # one dataset
    .venv/bin/python analyses/run_full_pipeline.py --device cpu --n-genes 200

Every step is a public ``scHopfield`` call (see ``sch.run_pipeline``); nothing here
is bespoke. Outputs land under ``benchmark_results/pipeline/<dataset>/``:

    adata_fitted.h5ad      fitted GRN + energies + Jacobian eigenvalues
    summary.json           per-cluster stability/energy medians + top drivers
    energy_stability.png   per-cluster energy / leading-eigenvalue / instability
    top_drivers.png        strongest GRN regulators (out-strength)
    perturbation_impact.png in-silico KO impact of the top driver, per cluster

and a combined ``benchmark_results/pipeline/pipeline_summary.json`` +
``README.md`` index across all datasets.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scHopfield as sch

OUT_ROOT = "benchmark_results/pipeline"
DYN = "/home/bernaljp/Documents/DynamiSC/Data"

# CellOracle base GRNs used as prior-knowledge scaffolds (species-appropriate).
MOUSE_GRN = "data/hematopoiesis/networks/mouse_scATAC_atlas.parquet"
HUMAN_GRN = "data/human_promoter_base_GRN.parquet"

# name -> config. `prepare` = run velocity/sigmoid preprocessing first;
# `base_grn` = species-appropriate base GRN to build the scaffold from. Every
# dataset is scaffold-guided so all fits use the same (penalized, L1-bias) estimator.
DATASETS = [
    dict(name="hematopoiesis", path="data/hematopoiesis/base_preprocessed.h5ad",
         cluster_key="paul15_clusters", species="mouse", prepare=False,
         base_grn="data/hematopoiesis/base_GRN.parquet"),
    dict(name="pancreas", path="data/Pancreas/pancreas_scvelo_ready.h5ad",
         cluster_key="clusters", species="mouse", prepare=False, base_grn=MOUSE_GRN),
    dict(name="murine_nc", path="data/generalize/murine_nc.h5ad",
         cluster_key="celltype_update", species="mouse", prepare=False, base_grn=MOUSE_GRN),
    dict(name="human_limb", path="data/generalize/human_limb.h5ad",
         cluster_key="leiden_R_celltype", species="human", prepare=False, base_grn=HUMAN_GRN),
    dict(name="schwann", path=f"{DYN}/schwann.h5ad",
         cluster_key="location", species="mouse", prepare=True, base_grn=MOUSE_GRN),
]

MODEL_UNS_KEYS = ("models", "jacobian_eigenvectors_temp")


def _clean_for_write(adata):
    """Drop non-h5ad-serializable trained torch models before saving."""
    uns = adata.uns.get("scHopfield", {})
    for k in MODEL_UNS_KEYS:
        uns.pop(k, None)
        adata.uns.pop(k, None)


def _present_clusters(adata, cluster_key, min_cells=20):
    lab = adata.obs[cluster_key].astype(str)
    order = list(lab.value_counts().index)
    return [c for c in order if (lab == c).sum() >= min_cells]


def _top_drivers(adata, cluster, n=12):
    """Rank regulators by out-strength (column L1 norm of |W|) for one cluster."""
    W = np.asarray(adata.varp[f"W_{cluster}"])
    strength = np.abs(W).sum(axis=0)  # W[target, regulator] -> sum over targets
    genes = np.asarray(adata.var_names)
    order = np.argsort(strength)[::-1][:n]
    return pd.DataFrame({"gene": genes[order], "out_strength": strength[order]})


def _plot_energy_stability(adata, cluster_key, present, path, title):
    obs = adata.obs
    lab = obs[cluster_key].astype(str)
    panels = [("energy_total", "Energy landscape depth", "total Hopfield energy"),
              ("jacobian_eig1_real", "Local stability (leading eig, Re)", "leading Jacobian eig (Re)"),
              ("jacobian_positive_evals", "Instability count", "# positive real eigenvalues")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, (col, ttl, yl) in zip(axes, panels):
        if col not in obs:
            ax.set_visible(False)
            continue
        data = [obs.loc[lab == c, col].values for c in present]
        ax.boxplot(data, showfliers=False)
        ax.set_xticklabels(present, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel(yl)
        ax.set_title(ttl)
        if col == "jacobian_eig1_real":
            ax.axhline(0, color="crimson", lw=0.8, ls="--")
    fig.suptitle(title, y=1.03, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_top_drivers(drivers, path, title):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    d = drivers.iloc[::-1]
    ax.barh(d["gene"], d["out_strength"], color="#3b6ea5")
    ax.set_xlabel("GRN out-strength (sum |W| over targets)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_perturbation(impact, path, title):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    impact = impact.sort_values(ascending=False)
    ax.bar(range(len(impact)), impact.values, color="#c0605a")
    ax.set_xticks(range(len(impact)))
    ax.set_xticklabels(impact.index, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("mean |delta_X| after KO")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def run_one(cfg, device, n_genes, seed):
    name = cfg["name"]
    out = f"{OUT_ROOT}/{name}"
    os.makedirs(out, exist_ok=True)
    print(f"\n{'='*70}\n{name}  ({cfg['species']})\n{'='*70}", flush=True)

    adata = ad.read_h5ad(cfg["path"])
    cluster_key = cfg["cluster_key"]

    # velocity/sigmoid preprocessing first, so the scaffold can be built over the
    # top-velocity genes (needed for datasets fit from raw counts, e.g. schwann).
    prepared = False
    if cfg["prepare"]:
        sch.pp.prepare_dataset(adata)
        prepared = True
        print(f"prepared: {adata.shape}", flush=True)

    # prior-knowledge scaffold from a species-appropriate base GRN
    scaffold_arg = None
    scaffold_info = None
    if cfg.get("base_grn") and os.path.exists(cfg["base_grn"]):
        if "sigmoid" not in adata.layers:
            adata.var["scHopfield_used"] = True
        adata_sub = sch.workflows.select_top_velocity_genes(adata, n_genes) \
            if n_genes and n_genes < adata.n_vars else adata
        base = pd.read_parquet(cfg["base_grn"])
        scaffold, ntf, nedge = sch.inf.build_scaffold(adata_sub, base, return_stats=True)
        scaffold_arg = scaffold.values.T
        scaffold_info = {"n_tfs": ntf, "n_edges": nedge}
        adata = adata_sub
        n_genes = None  # already subset
        print(f"scaffold: {ntf} TFs, {nedge} edges", flush=True)

    adata = sch.run_pipeline(
        adata, cluster_key=cluster_key, prepare=cfg["prepare"] and not prepared,
        n_top_genes=n_genes, scaffold=scaffold_arg,
        fit_kwargs=(dict(n_epochs=600, batch_size=128, learning_rate=0.1,
                         reconstruction_regularization=100, bias_regularization=1,
                         refit_gamma=True, use_plateau_scheduler=True,
                         plateau_patience=100, plateau_factor=0.1, drop_last=True,
                         include_neighbors=True, neighbor_fraction=0.2)
                    if scaffold_arg is not None else None),
        device=device, seed=seed, copy=False, verbose=True,
    )

    present = _present_clusters(adata, cluster_key)
    largest = present[0]

    # --- per-cluster energy / stability summary ---
    obs = adata.obs
    lab = obs[cluster_key].astype(str)
    stab = {}
    for c in present:
        m = lab == c
        stab[c] = {k: float(np.nanmedian(obs.loc[m, k].values))
                   for k in ["energy_total", "jacobian_eig1_real",
                             "jacobian_positive_evals", "jacobian_trace"] if k in obs}

    # --- network drivers (out-strength) ---
    drivers = _top_drivers(adata, largest, n=12)

    # --- in-silico perturbation: KO the top driver, measure per-cluster impact ---
    perturb = {"target": None, "impact_per_cluster": {}}
    try:
        target = drivers["gene"].iloc[0]
        ko = sch.dyn.simulate_perturbation(
            adata, perturb_condition={target: 0.0}, cluster_key=cluster_key,
            n_propagation=3, verbose=False)
        dX = np.asarray(ko.layers["delta_X"])
        mag = np.abs(dX).mean(axis=1)
        impact = pd.Series({c: float(np.nanmean(mag[(lab == c).values])) for c in present})
        perturb = {"target": str(target),
                   "impact_per_cluster": {k: float(v) for k, v in impact.items()}}
        _plot_perturbation(impact, f"{out}/perturbation_impact.png",
                           f"{name}: in-silico KO of {target} (per-cluster impact)")
    except Exception as exc:
        print(f"  perturbation step skipped: {type(exc).__name__}: {exc}", flush=True)

    # --- figures ---
    _plot_energy_stability(adata, cluster_key, present, f"{out}/energy_stability.png",
                           f"scHopfield energy + stability: {name}")
    _plot_top_drivers(drivers, f"{out}/top_drivers.png",
                      f"{name}: top GRN regulators ({largest})")

    # --- persist data + summary ---
    _clean_for_write(adata)
    adata.write(f"{out}/adata_fitted.h5ad")
    summary = {
        "name": name, "species": cfg["species"], "path": cfg["path"],
        "n_cells": int(adata.n_obs), "n_genes": int(adata.n_vars),
        "cluster_key": cluster_key, "clusters": present,
        "scaffold": scaffold_info, "seed": seed,
        "pipeline_steps": adata.uns.get("scHopfield_pipeline", {}).get("steps", []),
        "stability_by_cluster": stab,
        "top_drivers": drivers.to_dict("records"),
        "perturbation": perturb,
    }
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=2)
    print(f"  wrote {out}/ (adata_fitted.h5ad, summary.json, 3 figures)", flush=True)
    return summary


def write_index(summaries):
    combined = {s["name"]: s for s in summaries}
    json.dump(combined, open(f"{OUT_ROOT}/pipeline_summary.json", "w"), indent=2)
    lines = ["# scHopfield end-to-end pipeline results", "",
             "Generated by `analyses/run_full_pipeline.py` (one pipeline, many datasets).",
             "Each dataset runs preprocessing -> GRN fit -> energy -> Jacobian stability ->",
             "network drivers -> in-silico perturbation via `sch.run_pipeline`.", "",
             "| Dataset | Species | Cells | Genes | Clusters | Fit | Top driver |",
             "|---|---|---|---|---|---|---|"]
    for s in summaries:
        fit = f"scaffold ({s['scaffold']['n_edges']} edges)" if s.get("scaffold") else "pseudoinverse"
        lines.append(f"| {s['name']} | {s['species']} | {s['n_cells']} | {s['n_genes']} | "
                     f"{len(s['clusters'])} | {fit} | {s['perturbation'].get('target','-')} |")
    lines += ["", "## Per-dataset outputs", ""]
    for s in summaries:
        lines += [f"### {s['name']}",
                  f"- `{s['name']}/adata_fitted.h5ad` fitted GRN + energies + Jacobian eigenvalues",
                  f"- `{s['name']}/energy_stability.png`, `top_drivers.png`, `perturbation_impact.png`",
                  f"- `{s['name']}/summary.json`", ""]
    open(f"{OUT_ROOT}/README.md", "w").write("\n".join(lines))
    print(f"\nwrote {OUT_ROOT}/pipeline_summary.json + README.md", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="run a single dataset by name")
    ap.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    ap.add_argument("--n-genes", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", default=None,
                    help="output root (default: benchmark_results/pipeline)")
    args = ap.parse_args()

    if args.out_root:
        global OUT_ROOT
        OUT_ROOT = args.out_root

    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_ROOT, exist_ok=True)
    sch.set_seed(args.seed)
    print(f"device={args.device} n_genes={args.n_genes} seed={args.seed}", flush=True)

    todo = [d for d in DATASETS if (args.only is None or d["name"] == args.only)]
    summaries = []
    for cfg in todo:
        if not os.path.exists(cfg["path"]):
            print(f"skip {cfg['name']} (missing {cfg['path']})", flush=True)
            continue
        try:
            summaries.append(run_one(cfg, args.device, args.n_genes, args.seed))
        except Exception as exc:
            import traceback
            print(f"FAILED {cfg['name']}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
    if summaries:
        write_index(summaries)


if __name__ == "__main__":
    main()
